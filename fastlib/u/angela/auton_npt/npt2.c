/*
   File:        npt2.c
   Author:      Alexander Gray
   Description: Faster N-point computation I: 
                known-results caching
                distance caching
                early-stopping
                recursive structured sampling
*/

#include "npt.h"
#include "npt2.h"
#include "npt3.h"

extern int    Next_n;
extern bool   Draw_joiners;
extern bool   Use_Npt2;           /* AG */
extern bool   Use_Npt3;           /* AG */
extern bool   Use_MC;             /* AG */
extern int    Projection;         /* AG */
extern double Eps;                /* AG */
extern double Sig;                /* AG */
extern double Force_p;            /* AG */
extern double Union_p;            /* AG */
extern double Nsamples_block;     /* AG */
extern double Datafrac_crit;      /* AG */
extern double Rerrfrac_crit;      /* AG */
extern int    Num_to_expand;      /* AG */
extern int    Start_secs;         /* AG */
extern double ntuples_seen_so_far;
extern double total_ntuples;
extern double num_not_worth_it_prunes;

double Avg_num_tries = 0.0;

double fast_npt2(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                 bool use_symmetry,bool use_permutes,knode **kns,
                 double thresh_ntuples,double connolly_thresh,
                 double *lobound,double *hibound, 
                 dyv *wlobound, dyv *whibound,dyv *wresult,dyv *wsum,dyv *wsumsq,
                 imat *permutation_cache,int depth,
                 imat *known_ndpairs,dym *known_dists, ivec **maps, int *starts)
{
  //bool do_weights = (wresult && wlobound && whibound && kns[0]->sum_weights);
  int n = matcher_n(ma);
  double result = -7.777e77; /* Set to a wild value so we'll detect a problem
                                if some path through the code neglects to set
                                result to a value */
  dyv *wtemp_result = NULL;
  dyv *wtemp_sum = NULL;
  dyv *wtemp_sumsq = NULL;
  int i,j;
  bool answer_is_zero = FALSE; /* Will be set to TRUE if we prove an
                                  EXCLUDE exists */

  bool all_subsume = TRUE; /* Set to TRUE until proven otherwise. This
                              will eventually hold the answer to the question:
                              "do all pairs of hrects satisfy SUBSUME status?" */
  /* Note, SUBSUME and EXCLUDE are described and defined in matcher.h */

  bool all_leaves = TRUE; /* Set to TRUE until proven otherwise. Will
                             eventually answer "are all the nodes in the 
                             tuple kdtree leaf nodes? */
                
  int num_subsumes[MAX_N]; /* Eventually, num_subsumes[i] will 
                              contain the number of other nodes that
                              kns[i] subsumes. */

  bool recursed = FALSE; /* Flag used only for deciding when to printf
                            stuff */

  double ntuples = total_num_ntuples(n,(use_symmetry || use_permutes),kns);
            /* The maximum possible increment to the count that could be
               obtained from this knode tuple. Note that if the matcher
               is symmetric and the nodes are in the wrong order,
               this will be zero; use_permutations also forces the ordering
               so the parameter sent is the OR of those */

  //dyv *weighted_ntuples = (do_weights) ? 
  //  mk_weighted_total_ntuples(n,(use_symmetry || use_permutes),kns) : NULL;

  double datafrac = ntuples_seen_so_far / total_ntuples;
  double errfrac = compute_errfrac(*lobound,*hibound);

  //bool not_worth_it = (ntuples <= thresh_ntuples);
  //bool not_worth_it = ((ntuples <= thresh_ntuples) || 
  //                    (errfrac * datafrac <= Eps)) &&
  //                    (*lobound > 0) && (ntuples_seen_so_far > 0);
  bool not_worth_it = Use_MC ? ((ntuples == 0.0) || (errfrac <= Eps))
                             : ((ntuples <= thresh_ntuples) || (errfrac <= Eps));

  ivec *permute_status = (permutation_cache) ? 
    mk_constant_ivec(imat_rows(permutation_cache),SUBSUME) : NULL;

  /* a clunky way to figure out which node may be best to split on next. */
  imat *num_incons = (permutation_cache) ? 
    mk_zero_imat(imat_rows(permutation_cache),n) : NULL;

  /* PRUNING **************/
  if ( not_worth_it ) {
    if (ntuples > 0) num_not_worth_it_prunes++;
    if ( Verbosity >= 1.0 )
      printf("thresh_ntuples = %g, this_ntuples = %g so I am cutting.\n",
             thresh_ntuples,ntuples);
  }
  else if ( connolly_thresh > 0.0 ) {
    bool all_diameters_below_connolly_thresh = TRUE;
    int i;
    for ( i = 0 ; all_diameters_below_connolly_thresh && i < n ; i++ ) {
      double diameter = hrect_diameter(kns[i]->hr);
      if ( diameter > connolly_thresh )
        all_diameters_below_connolly_thresh = FALSE;
    }
    if ( all_diameters_below_connolly_thresh )
      not_worth_it = TRUE;
  }/***********************/
  
  /* This if block only used for reporting */
  //if (Verbosity >= 0.5) 
    if ( Next_n < 1000000000 && Num_pt_dists + Num_hr_dists > Next_n )
    {
      printf("%9d dists. ",Num_pt_dists + Num_hr_dists);
      printf("lo = %9.5e, hibound = %9.5e ",*lobound,*hibound);
      if ( errfrac < 1e4 ) printf("(errfrac %9.4f)",errfrac);
      else                 printf("(errfrac %9g)",errfrac);
      printf(" datafrac %9.4f",datafrac);
      printf("\n");
      //Next_n *= 2;
      Next_n += 10000000;
    }

  for ( i = 0 ; !not_worth_it && i < n ; i++ ) num_subsumes[i] = 0;

  /* AWM note to self: You can't quit this loop as soon as you
     realize answer_is_zero because you are trying to accumulate the
     high bound. 

     AWM another note to self. I don't believe the above comment any
     more but I haven't had time to think it through or test it out.
  */
  /* MAIN LOOP ***************************************************/
  for ( i = 0 ; !answer_is_zero && !not_worth_it && i < n ; i++ )
  {
    knode *kni = kns[i];
    if (!knode_is_leaf(kni)) all_leaves = FALSE;

    /* Now we're putting the redundant-comparisons checking into this 
       implementation of the n-point algorithm.  AG */
    for ( j = i+1 ; !answer_is_zero && j < n ; j++ )
    { 
      //////////////////////////////////////////////////////////////////////////
      // Only do a pair-test if its result is unknown.
      // Store the result of a pair-test in known_ndpairs.
      //////////////////////////////////////////////////////////////////////////
      knode *knj = kns[j];
      int status = imat_ref(known_ndpairs,i,j);

      if (use_symmetry || !use_permutes)
      {
        if (status == UNKNOWN) {
          status = matcher_test_hrect_pair(ma,kni->hr,knj->hr,i,j);
          imat_set(known_ndpairs,i,j,status);
        }
        if (status == EXCLUDE) answer_is_zero = TRUE;
        else if ( status == SUBSUME ) {
          num_subsumes[i] += 1; num_subsumes[j] += 1;
        }
        else all_subsume = FALSE;
      }
      else 
      {
        if (status == UNKNOWN) {
          status = matcher_permute_test_hrect_pair(ma,kni->hr,knj->hr,i,j,
                                                   permute_status,num_incons,
                                                   permutation_cache);
          imat_set(known_ndpairs,i,j,status);
        }
        if (status == EXCLUDE) answer_is_zero = TRUE;
      }
    }
  }

  /* some extra cleanup to find the real state. first we figure out if we have
     a subsume for any of the permutations.  the next part is stranger.  we're
     just trying to wedge the results into the same logic later for choosing
     which node to split on.  if a permutation was inconclusive then every
     node who was inconclusive in that permutation gets a checkmark.  in this
     case "getting a checkmark" means setting num_subsumes to what is expected
     later in the old way of doing the checking. 
  */
  if (!use_symmetry && use_permutes && !answer_is_zero)
  {
    all_subsume = FALSE;
    /* if this n-tuple subsumes under even one permutation, it counts - AG */
    for (i=0;i<ivec_size(permute_status);i++)
      if (ivec_ref(permute_status,i) == SUBSUME) all_subsume = TRUE;
    if (!all_subsume)
      for (i=0;i<ivec_size(permute_status);i++)
	if (ivec_ref(permute_status,i) == INCONCLUSIVE)
	  for (j=0;j<imat_cols(num_incons);j++)
	    if (imat_ref(num_incons,i,j) > 0)
	      num_subsumes[j] = n-1;
  }

  /* This if block only concerned with animations - TOOK THIS OUT */

  /* Okay, now we're in a position to go ahead and prune if we can */
  if ( not_worth_it )
  {
    result = 0.0;  /* and don't bother updating wresult */
    ntuples_seen_so_far += ntuples; // AG
    //if ( Verbosity >= 1.0 )
    //  printf("Number of possible matches zero or below thresh. Prune\n");
    ///* Note we should NOT decrease hi bound in this case */
  }
  else if ( answer_is_zero )
  {
    result = 0.0;  /* and don't bother updating wresult */
    //if ( Verbosity >= 1.0 )
    //{
    //  int kk;
    //  printf("nodes: "); 
    //  for (kk=0;kk<n;kk++) 
    //    printf("%3d:%3d  ",kns[kk]->lo_index,kns[kk]->hi_index); printf("\n");
    //  printf("Impossible to match. Reduce hibound by %g\n",ntuples);
    //  if (weighted_ntuples) {
    //    dyv *test_wresult = mk_zero_dyv(dyv_size(wresult));
    //    special_weighted_symmetric_debugging_test(xs,ws,ma,kns,test_wresult);
    //    fprintf_oneline_dyv(stdout,"weighted hibound reduced by",
    //                        weighted_ntuples,"\n");
    //    if (!dyv_equal(test_wresult,weighted_ntuples)) {
    //      printf("fast_npt: Dyvs not equal!!\n");
    //      fprintf_oneline_dyv(stdout,"test_dyv",test_wresult,"\n");
    //      really_wait_for_key();
    //    }
    //  }
    //}
    *hibound -= ntuples;
    ntuples_seen_so_far += ntuples; // AG
    //if (do_weights) 
    //{
    //  dyv_subtract(whibound,weighted_ntuples,whibound);
    //  printf("%20.10f decrease in whibound\n",dyv_ref(weighted_ntuples,0));
    //}
  }
  else if ( all_subsume )
  {
    result = ntuples;
    //if ( Verbosity >= 1.0 )
    //  printf("All tuples match. Prune. Increases lobound by %g\n",ntuples);
    *lobound += ntuples;
    ntuples_seen_so_far += ntuples; // AG
    //if (do_weights) 
    //{
    //  dyv_plus(wresult,weighted_ntuples,wresult);
    //  dyv_plus(wlobound,weighted_ntuples,wlobound);
    //}
    //if (wsum) {
    //  dyv *tmp = mk_weighted_sum_ntuples(n, (use_symmetry || use_permutes), 
    //                                     FALSE, kns);
    //  dyv_plus(wsum,tmp,wsum);
    //  free_dyv(tmp);
    //}
    //if (wsumsq)
    //{
    //  dyv *tmp = mk_weighted_sum_ntuples(n, (use_symmetry || use_permutes), 
    //                                     TRUE, kns);
    //  dyv_plus(wsumsq,tmp,wsumsq);
    //  free_dyv(tmp);
    //}
  }
  else 
  {
    /* Sampling *********************/
    bool happy = FALSE;

    if ( Use_MC )    
    {
      double s2 = real_square(Sig), p = 0.0, sd = 0.0;
      double nsamples = 100.0, last_nsamples = 0.0, last_nmatches = 0.0;

      while ( nsamples < 0.5 * ntuples )
      {
        double nmatches = sample_npt(ms,xs,ws,ma,kns,use_symmetry,use_permutes,
                                     permutation_cache,wtemp_result,wtemp_sum,
                                     wtemp_sumsq,known_ndpairs,known_dists,maps,
                                     starts,nsamples - last_nsamples);
        p = (last_nmatches + nmatches + s2/2.0) / (nsamples + s2);

        if ( sqrt(nsamples*p*(1.0-p)) < 3.0 ) break;

        sd = Sig * sqrt(p*(1.0-p)/(nsamples+s2));

        if ( (sd <= Eps*p) && (p-sd >= 0) && (p+sd <= 1) ) {
          result = p * ntuples;
          *lobound += result - sd*ntuples;
          *hibound += result + sd*ntuples - ntuples;
          ntuples_seen_so_far += ntuples;
          //printf("p=%#3.2g, est=%g, ntuples=%g, sd=%g, nsamples=%g\n",
          //       p,result,ntuples,sd,nsamples);
          happy = TRUE; break;
        } 
        else {
          last_nmatches = nmatches; last_nsamples = nsamples;
          nsamples = real_max( compute_nsamples(p,Sig,Eps), 1.1 * last_nsamples);
        }
      }
    }/*******************************/

    if ( happy == FALSE )
    {
      /* We're going to recurse by finding one element of kns (call it kns[i]
         and calling fast_npt twice, with kns[i] replaced by kns[i]->left and
         kns[i] -> right. But what should i be? We're going to use the 
         following (debatable) strategy:
               Choose it to be the biggest non-leaf, non-all-subsuming node
               available. "biggest" means most points. Non-leaf is because
               we can't recurse on leaves. Non-all-subsuming is because
               it would be a waste to recurse on an all-subsuming node since
               it is the non-all-subsuming nodes that need to get smaller
               if we are going to be able to do a subsume-based prune. 
      
	       If all nodes are leaves we'll do it the slow way
      
               If all non-leaf nodes are subsuming, split the largest
               non-leaf node. (Andrew had a lot of code for doing something
               cleverer in this case, involving doing the slow thing over
               the leaf nodes only and multiplying by a factor derived from
               the non-leaf-but-subsume nodes. He abandonded it because it
               was too complicated to factor in permutation-overcounting-prevent
               issues. But it would be easy to reinstitute for non-symmetric
               searches). */
      
      int split_index = -1;
      int split_index_num_points = -77;
      int min_num_subsumes = n;
      
      for ( i = 0 ; i < n ; i++ )
      {
        double ns = num_subsumes[i];
        int size = kns[i]->num_points;
        if ( !knode_is_leaf(kns[i]) ) {
          if (ns < min_num_subsumes) {
            split_index = i; split_index_num_points = size; 
            min_num_subsumes = ns;
          } else 
          if ((ns == min_num_subsumes) && (size > split_index_num_points)) {
            split_index = i; split_index_num_points = size; 
            min_num_subsumes = ns;
          }      
        }
      }
      
      //for ( i = 0 ; i < n ; i++ )
      //{
      //  bool subsume_all = num_subsumes[i] == n-1;
      //  if ( !subsume_all && !knode_is_leaf(kns[i]) )
      //  {
      //    int num_points = kns[i]->num_points;
      //    if ( split_index < 0 || num_points > split_index_num_points )
      //    {
      //      split_index = i;
      //      split_index_num_points = num_points;
      //    }
      //  }
      //}
      //
      ///* The only way that split_index could be undefined is if all 
      //   non-all-subsuming knodes were leaves */
      //
      //if ( split_index < 0 ) {
      //  /* We failed to find a non-subsuming non-leaf. Now we'll be happy
      //     with the largest non-leaf whether it sumbsumes or not. */
      //
      //  for ( i = 0 ; i < n ; i++ ) {
      //    if ( !knode_is_leaf(kns[i]) ) {
      //      int num_points = kns[i]->num_points;
      //      if ( split_index < 0 || num_points > split_index_num_points ) {
      //        split_index = i;
      //        split_index_num_points = num_points;
      //      }
      //    }
      //  }
      //}
        
      if ( split_index < 0 )
      {
        /* All the nodes are leaves */
        ivec *rowsets[MAX_N];
      
        for ( i = 0 ; i < n ; i++ )
        {
          if ( !knode_is_leaf(kns[i]) ) my_error("no way jose");
          rowsets[i] = kns[i] -> rows;
        }
      
        //if (do_weights) wtemp_result = mk_zero_dyv(dyv_size(wresult));
        //if (wsum) wtemp_sum = mk_zero_dyv(dyv_size(wsum));
        //if (wsumsq) wtemp_sumsq = mk_zero_dyv(dyv_size(wsumsq));
      
        if (use_symmetry || !use_permutes)
          result = slow_npt2(ms,xs,ws,ma,use_symmetry,rowsets,wtemp_result,
                             wtemp_sum,wtemp_sumsq,known_ndpairs,known_dists);
        else
          result=slow_permute_npt2(ms,xs,ws,ma,rowsets,permutation_cache,
                                   wtemp_result,wtemp_sum,wtemp_sumsq,
                                   known_ndpairs,known_dists);
      
        *lobound += result;
        *hibound -= (ntuples - result);
        ntuples_seen_so_far += ntuples; // AG
      
        //if (do_weights) {
        //  dyv_plus(wlobound,wtemp_result,wlobound);
        //  dyv_subtract(whibound,weighted_ntuples,whibound);
        //  dyv_plus(whibound,wtemp_result,whibound);
        //  dyv_plus(wresult,wtemp_result,wresult);
        //  
        //  if (Verbosity >= 1.0) {
        //    dyv *test_wresult = mk_zero_dyv(dyv_size(wresult));
        //    special_weighted_symmetric_debugging_test(xs,ws,ma,kns,
        //                                              test_wresult);
        //    fprintf_oneline_dyv(stdout,"found",wtemp_result,"\n");
        //    fprintf_oneline_dyv(stdout,"reduce hibound by (minus found)",
        //                        weighted_ntuples,"\n");
        //    if (!dyv_equal(test_wresult,weighted_ntuples)) {
        //      printf("fast_npt: Dyvs not equal!!\n");
        //      fprintf_oneline_dyv(stdout,"test_dyv",test_wresult,"\n");
        //      really_wait_for_key();
        //    }
        //  }
        //  free_dyv(wtemp_result);
        //}
        //if (wsum) {
        //  dyv_plus(wsum,wtemp_sum,wsum);
        //  free_dyv(wtemp_sum);
        //}
        //if (wsumsq) {
        //  dyv_plus(wsumsq,wtemp_sumsq,wsumsq);
        //  free_dyv(wtemp_sumsq);
        //}
      }
      else /* There's someone waiting to be splitted... */
      {
        knode *parent = kns[split_index];
        knode *child1 = parent->left;
        knode *child2 = parent->right;
        imat *ko_copy;
      
        recursed = TRUE;
        //if ( !Do_rectangle_animation && Verbosity >= 1.0 )
        //{
        //  printf("About to recurse. lobound=%g, hibound=%g\n",
        //         *lobound,*hibound);
        //  wait_for_key();
        //}
      
        result = 0.0;
        kns[split_index] = child1;
      
        /////////////////////////////////////////////////////////////////////////
        known_ndpairs = prepare_known_ndpairs_matrix(known_ndpairs,split_index);
        ko_copy = mk_copy_imat(known_ndpairs);
        /////////////////////////////////////////////////////////////////////////
      
        result += fast_npt2(ms,xs,ws,ma,use_symmetry,use_permutes,kns,
                            thresh_ntuples,connolly_thresh,lobound,hibound,
                            wlobound,whibound,wresult,wsum,wsumsq,
                            permutation_cache,depth+1,
                            ko_copy,known_dists,maps,starts);
      
        kns[split_index] = child2;
        free_imat(ko_copy);
      
        result += fast_npt2(ms,xs,ws,ma,use_symmetry,use_permutes,kns,
                            thresh_ntuples,connolly_thresh,lobound,hibound,
                            wlobound,whibound,wresult,wsum,wsumsq,
                            permutation_cache,depth+1,
                            known_ndpairs,known_dists,maps,starts);
      
        kns[split_index] = parent;
      
        /////////////////////////////////////////////////////////////////////////
        // here we could combine the info from recursing on both halves:
        //   if left was S and right was S, set total-node result to S
        //   if left was E and right was E, set total-node result to E
        //   any other case, set total-node result to I
        //   if any are U, error since they should be computed by this point
        /////////////////////////////////////////////////////////////////////////
      }
    }
  }

  //if ( !recursed && Verbosity >= 1.0 )
  //{
  //  printf("lobound = %g, hibound = %g\n",*lobound,*hibound);
  //  wait_for_key();
  //}

  //if (weighted_ntuples) free_dyv(weighted_ntuples);
  if (permute_status) free_ivec(permute_status);
  if (num_incons) free_imat(num_incons);

  return result;
}

/* To prepare for replacing a node in the set with one of its children in a
   recursive call.  Set the row and column of split index s appropriately:
     if entry was S, it stays S
     if entry was I, make it U
     if entry was E, error since the appearance of an E means no recursion
     if entry was U, error since it should have been computed by this point
*/
imat *prepare_known_ndpairs_matrix(imat *ko,int s)
{
  int i, n = imat_rows(ko);

  // row s
  for ( i = s+1 ; i < n ; i++ ) {
    int val = imat_ref(ko,s,i);
    switch(val) {
      case SUBSUME:      break;
      case INCONCLUSIVE: imat_set(ko,s,i,UNKNOWN); break;
      default:           my_error("whatchoo talkin bout, willis?"); break;
    }
  }
  // column s
  for ( i = 0 ; i < s ; i++ ) {
    int val = imat_ref(ko,i,s);
    switch(val) {
      case SUBSUME:      break;
      case INCONCLUSIVE: imat_set(ko,i,s,UNKNOWN); break;
      default:           my_error("whatchoo talkin bout, willis?"); break;
    }
  }

  return ko;
}

////////////////////////////////////////////////////////////////////////////////
// This incorporates known_ndpairs:
//   if there is an S between two nodes, none of the distances between rows
//     from those two nodes need to be checked.
// Also incorporates known_dists:
//   if the value is already stored in known_dists, don't compute that distance.
////////////////////////////////////////////////////////////////////////////////
double slow_npt2_helper(mapshape *ms,dym **xs,dym **ws, matcher *ma,
                        bool use_symmetry,int k,int *row_indexes,ivec **rowsets,
                        dyv *wresult,dyv *wsum, dyv *wsumsq, 
                        imat *known_ndpairs,dym *known_dists)
{
  int n = matcher_n(ma);
  int i;
  ivec *rowset_k = rowsets[k];
  double result = 0.0;
  bool cutoff = FALSE;

  /* The following loop iterates over all choices for the k'th
     tuple-member. What's cutoff all about? Well, if the matcher
     is symmetric and if two of the
     rowsets are from the same kdnode then we only want to count
     tuples in which the index in the first rowset is strictly
     less that the index in the second rowset. This prevents
     double-counting. So IF it's a symmetric matcher and
     rowsets[k] comes from the same knode
     as any rowset[j] for j < k then cutoff is set to TRUE (and the 
     loop stops early) as soon as index i becomes greater than 
     row_indexes[j]. */
  for ( i = 0 ; !cutoff && i < ivec_size(rowset_k) ; i++ )
  {
    int row = ivec_ref(rowset_k,i);
    int j;
    bool ok = TRUE;

    /* The following loop finds out if setting the k'th tuple-member
       to be the row'th row of x causes a conflict with any of the 
       earlier-indexed tuple-members, and if so it sets ok to FALSE
       and exits the loop. Furthermore, we're also monitoring for
       cutoff, as described in the outer loop. */
    for ( j = 0 ; !cutoff && ok && j < k ; j++ )
    {
      int row_index_j = row_indexes[j];
      int row_j = ivec_ref(rowsets[j],row_index_j);
      bool k_and_j_rows_from_same_knode = 
        ((long)(rowsets[k])) == ((long)(rowsets[j]));
      bool match = FALSE;

      cutoff = use_symmetry && 
	       k_and_j_rows_from_same_knode && row_index_j <= i;

      if (!cutoff) {
        if (imat_ref(known_ndpairs,j,k) == SUBSUME) match = TRUE; 
        else match = matcher_test_point_pair(ma,NONE,NONE,
                                             xs[k],xs[j],row,row_j,k,j);
        //else match = matcher_test_point_pair2(ma,xs[k],xs[j],row,row_j,k,j,
        //                                      known_dists);
      }

      ok = !cutoff && match;

      //if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) ) {
      //  if ( ok )
      //    ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
      //                             (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
      //  ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
      //  ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
      //}
      //if ( !cutoff && ok && Draw_joiners )
      //  ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
    }

    /* Now, we will do more work ONLY IF there's no constraint violation,
       and no double-counting... */
    if ( !cutoff && ok )
    {
      if ( k == n-1 )
      {
        result += 1.0; /* base case of recursion */
        //if (wresult || wsum || wsumsq) {
        //  int wi,ki;
        //  double wsofar;
        //
        //  row_indexes[k] = i;  /* just for the loop below */
        //  for (wi=0;wi<dyv_size(wresult);wi++) 
        //  {
        //    double temp;
        //    wsofar = 1.0;
	    //
        //    for (ki=0;ki<n;ki++)
        //    {
        //      temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
        //      wsofar *= temp;
        //      if (wsum) dyv_increment(wsum,wi,temp);
        //      if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
        //    }
        //    dyv_increment(wresult,wi,wsofar);
        //  }
        //  row_indexes[k] = -77; /* prevents accidental uses */
        //}
      }
      else 
      {
        row_indexes[k] = i; /* recursive case */
        result += slow_npt2_helper(ms,xs,ws,ma,use_symmetry,k+1,row_indexes,
                                   rowsets,wresult,wsum,wsumsq,known_ndpairs,
                                   known_dists);
        row_indexes[k] = -77; /* Just to prevent anyone accidently using
                                 row_indexes[k] again */
      }
    }
  }

  return result;
}

double slow_npt2(mapshape *ms,dym **xs,dym **ws,matcher *ma,bool use_symmetry,
                 ivec **rowsets,dyv *wresult, dyv *wsum,dyv *wsumsq, 
                 imat *known_ndpairs,dym *known_dists)
{
  int rows[MAX_N];
  int i;
  double result;

  if ( matcher_n(ma) > MAX_N ) my_error("MAX_N too small");

  for ( i = 0 ; i < matcher_n(ma) ; i++ ) rows[i] = -77;

  result = slow_npt2_helper(ms,xs,ws,ma,use_symmetry,0,rows,rowsets,wresult,
                            wsum,wsumsq,known_ndpairs,known_dists);
  return result;
}
  
double slow_permute_npt2_helper(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                                int k,int *row_indexes,ivec **rowsets,
                                imat *permutation_cache,ivec *permutes_ok,
                                dyv *wresult,dyv *wsum,dyv *wsumsq,
                                imat *known_ndpairs,dym *known_dists)
{
  int n = matcher_n(ma);
  int i;
  ivec *rowset_k = rowsets[k];
  double result = 0.0;
  bool cutoff = FALSE;
  ivec *permutes_ok_copy = mk_copy_ivec(permutes_ok);

  /* The following loop iterates over all choices for the k'th
     tuple-member. What's cutoff all about? Well, if two of the
     rowsets are from the same kdnode then we only want to count
     tuples in which the index in the first rowset is strictly
     less that the index in the second rowset. This prevents
     double-counting. So if rowsets[k] comes from the same knode
     as any rowset[j] for j < k then cutoff is set to TRUE (and the 
     loop stops early) as soon as index i becomes greater than 
     row_indexes[j]. */
  for ( i = 0 ; !cutoff && i < ivec_size(rowset_k) ; i++ )
  {
    int row = ivec_ref(rowset_k,i);
    int j;
    bool ok = TRUE;

    /* The following loop finds out if setting the k'th tuple-member
       to be the row'th row of x causes a conflict with any of the 
       earlier-indexed tuple-members, and if so it sets ok to FALSE
       and exits the loop. Furthermore, we're also monitoring for
       cutoff, as described in the outer loop. */
    copy_ivec(permutes_ok,permutes_ok_copy);
    for ( j = 0 ; !cutoff && ok && j < k ; j++ )
    {
      int row_index_j = row_indexes[j];
      int row_j = ivec_ref(rowsets[j],row_index_j);
      bool k_and_j_rows_from_same_knode = 
        ((long)(rowsets[k])) == ((long)(rowsets[j]));
      bool match = FALSE;

      cutoff = k_and_j_rows_from_same_knode && row_index_j <= i;

      if (!cutoff) {
        if (imat_ref(known_ndpairs,j,k) == SUBSUME) match = TRUE;
        else match = matcher_permute_test_point_pair(ma,NONE,NONE,
                                                     xs[k],xs[j],row,row_j,
                                                     k,j,permutation_cache,
                                                     permutes_ok_copy);
        //else match = matcher_permute_test_point_pair2(ma,xs[k],xs[j],row,row_j,
        //                                              k,j,permutation_cache,
        //                                              permutes_ok_copy,
        //                                              known_dists);
      }

      ok = !cutoff && match;

      //if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) )
      //{
      //  if ( ok )
      //    ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
      //                             (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
      //  ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
      //  ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
      //}
      //if ( !cutoff && ok && Draw_joiners )
      //  ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
    }

    /* Now, we will do more work ONLY IF there's no constraint violation,
       and no double-counting... */
    if ( !cutoff && ok )
    {
      if ( k == n-1 )
      {
        result += 1.0; /* base case of recursion */
        //if (wresult || wsum || wsumsq)
        //{
        //  int wi,ki;
        //  double wsofar;
        //  
        //  row_indexes[k] = i;  /* just for the loop below */
        //  for (wi=0;wi<dyv_size(wresult);wi++)
        //  {
        //    double temp;
        //    wsofar = 1.0;
        //
        //    for (ki=0;ki<n;ki++)
	    //    {
        //      temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
        //      wsofar *= temp;
        //      if (wsum) dyv_increment(wsum,wi,temp);
        //      if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
        //    }
        //    dyv_increment(wresult,wi,wsofar);
        //  }
        //  row_indexes[k] = -77; /* prevents accidental uses */
        //}
      }
      else
      {
        row_indexes[k] = i; /* recursive case */
        result += slow_permute_npt2_helper(ms,xs,ws,ma,k+1,row_indexes,rowsets,
                                           permutation_cache,permutes_ok_copy,
                                           wresult,wsum,wsumsq,known_ndpairs,
                                           known_dists);
        row_indexes[k] = -77; /* Just to prevent anyone accidently using
                                 row_indexes[k] again */
      }
    }
  }
  free_ivec(permutes_ok_copy);

  return result;
}

/* Same comments as the helper function above, except "with k == 0": i.e.
   we are searching through all sets of points, finding which ones match. 

   Just like slow_npt except that it tries all permutations of the template
   and always has use_symmetry = TRUE (which is why its dropped from the
   argument list.

   The idea is to count asymmetric templates (such as non-equilateral
   triangles) in the same way that symmetric ones are now counted.  This
   means that a match to an equilateral triangle will only add 1 to the
   count rather than 6 for each matching triple, and an isoceles triangle
   will only add 1 to the count rather than 2 for each matching triple.

   A note on the results of this vs slow_npt:
   1. The symmetric case (e.g. equilateral triangle).  This function will
      produce the same answer as slow_npt, but it will take longer.  Both
      functions are equally smart about only considering tuples of points 
      whose indices are in order.  This function will compare to all 6
      (identical) permutations of the template, while slow_npt will not.
   2. The asymmetric case.  This function will return a different count
      than slow_npt (as described above).  It is unclear whether it will
      take more or less time.  It gains time by only considering tuples in
      order (as in the symmetric case), but loses it by testing each tuple
      against all permutations of the template.
*/
double slow_permute_npt2(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
			ivec **rowsets,imat *permutation_cache,dyv *wresult,
			dyv *wsum, dyv *wsumsq,imat *known_ndpairs,dym *known_dists)
{
  int rows[MAX_N];
  int i;
  double result;
  int num_permutes = 1;
  ivec *permutes_ok;

  if ( matcher_n(ma) > MAX_N ) my_error("MAX_N too small");

  for (i=matcher_n(ma); i>1; i--) num_permutes *= i;
  permutes_ok = mk_constant_ivec(num_permutes,1);

  for ( i = 0 ; i < matcher_n(ma) ; i++ ) rows[i] = -77;

  result = slow_permute_npt2_helper(ms,xs,ws,ma,0,rows,rowsets,
                                    permutation_cache,permutes_ok,wresult,
                                    wsum,wsumsq,known_ndpairs,known_dists);

  free_ivec(permutes_ok);
  
  return result;
}

/******************************************************************************/
///////////////////
////////// SAMPLING
///////////////////
/******************************************************************************/

double compute_nsamples(double p, double sig, double eps)
{
  //return floor( real_square(s/eps) * (1.0/p - 1.0) );
  return floor( real_square( (sig*0.5)/(eps*p) ) );
}

double sample_npt(mapshape *ms, dym **xs,dym **ws, matcher *ma, knode **kns,
                  bool use_symmetry,bool use_permutes,imat *permutation_cache,
                  dyv *wresult,dyv *wsum, dyv *wsumsq,
                  imat *known_ndpairs,dym *known_dists,ivec **maps,int *starts,
                  double nsamples)
{
  int i,s, n = matcher_n(ma); double nmatches = 0.0;
  ivec *rowsets[MAX_N]; int rowset_indexes[MAX_N];

  //printf("sampling from nodeset: (%d,%d) (%d,%d) (%d,%d)\n",
  //       kns[0]->lo_index,kns[0]->hi_index,kns[1]->lo_index,kns[1]->hi_index,
  //       kns[2]->lo_index,kns[2]->hi_index);

  for ( i = 0 ; i < n ; i++ ) {
    int j; bool done = FALSE;
    for ( j = 0 ; j < i ; j++ ) {
      if (kns[j] == kns[i]) { // see if a rowset for knode has already been made
        add_to_ivec( rowsets[j],-777 );
        rowsets[i] = rowsets[j];
        rowset_indexes[i] = ivec_size( rowsets[i] ) - 1;
        done = TRUE; break;
      }
    }
    if (!done) { // otherwise make a rowset for the knode
      rowsets[i] = mk_ivec_1(-777); 
      rowset_indexes[i] = 0;
    }
  }

  for ( s = 0 ; s < nsamples ; s++ ) 
  {
    double match = -1; 
    //bool valid = FALSE; int num_tries = 1;
    //while (!valid) {
    //  bool start_over = FALSE;
    //  for ( i = 0 ; i < n ; i++ ) {
    //    int row = int_random(kns[i]->num_points);
    //    int dym_row = ivec_ref(kns[i]->rows,row);
    //    int virtual_index = ivec_ref(maps[i],dym_row);
    //    ivec_set(rowsets[i], 0, dym_row);
    //    //printf("dym_row: %d, virtual index: %d\n", dym_row,virtual_index);
    //  
    //    if ((use_symmetry || !use_permutes) && (i>0)) {
    //      int last_dym_row = ivec_ref(rowsets[i-1], 0);
    //      int last_virtual_index = ivec_ref(maps[i-1],last_dym_row);
    //      if (virtual_index < last_virtual_index) { 
    //        /* printf("fuck\n"); */ start_over = TRUE; break; }
    //    }
    //  }
    //  if (start_over == FALSE) { /* printf("YES\n"); */ valid = TRUE; }
    //  else num_tries++;
    //}
    //Avg_num_tries += num_tries;

    //int virtual_indexes[MAX_N], dym_row;
    //for ( i = 0 ; i < n ; i++ ) {
    //  if ((use_symmetry || !use_permutes) && (i>0)) {
    //    virtual_indexes[i] = virtual_indexes[i-1] + 
    //      int_random(kns[i]->hi_index - virtual_indexes[i-1]);
    //  } else {
    //    virtual_indexes[i] = kns[i]->lo_index + int_random(kns[i]->num_points);
    //  }
    //  dym_row = ivec_ref(maps[i],virtual_indexes[i]);
    //  ivec_set(rowsets[i], 0, dym_row);
    //  //printf("dym_row: %d, virtual index: %d\n", dym_row,virtual_index);
    //}

    bool valid = FALSE; int num_tries = 1;
    int virtual_indexes[MAX_N], dym_row;
    while (!valid) {
      bool start_over = FALSE;
      for ( i = 0 ; i < n ; i++ ) {
        //if ((use_symmetry || !use_permutes) && (i>0) && (kns[i]==kns[i-1])) {
        //  virtual_indexes[i] = virtual_indexes[i-1] + 
        //    int_random(kns[i]->hi_index - virtual_indexes[i-1]);
        //} else
        //  virtual_indexes[i]=kns[i]->lo_index + int_random(kns[i]->num_points);
        virtual_indexes[i] = kns[i]->lo_index + int_random(kns[i]->num_points);
        dym_row = ivec_ref(maps[i],virtual_indexes[i]-starts[i]);
        ivec_set(rowsets[i], rowset_indexes[i], dym_row);

        if (i>0) {
          if (virtual_indexes[i] < virtual_indexes[i-1]) { 
            start_over = TRUE; break; }
        }
      }
      if (start_over == FALSE) { valid = TRUE; }
      else num_tries++;
    }
    Avg_num_tries += num_tries;

    /* check for a match */
    if (use_symmetry || !use_permutes) {
      if (known_ndpairs != (imat *)NULL && known_dists != (dym *)NULL)
        match = slow_npt2(ms,xs,ws,ma,use_symmetry,rowsets,
                          wresult,wsum,wsumsq,known_ndpairs,known_dists);
      else
        match = slow_npt(ms,xs,ws,ma,use_symmetry,NONE,NONE,
                         rowsets,wresult,wsum,wsumsq);
    } else {
      if (known_ndpairs != (imat *)NULL && known_dists != (dym *)NULL)
        match = slow_permute_npt2(ms,xs,ws,ma,rowsets,permutation_cache,
                                  wresult,wsum,wsumsq,known_ndpairs,known_dists);
      else
        match = slow_permute_npt(ms,xs,ws,ma,NONE,NONE,rowsets,permutation_cache,
                                 wresult,wsum,wsumsq);
    }
    if (match == 1.0) nmatches += 1;
  }
  
  for ( i = 0 ; i < n ; i++ ) {
    if (rowset_indexes[i]==0) free_ivec(rowsets[i]);
  }

  return nmatches;
}

/* note: the values stored in array are assumed to represent the *upper ends* of
   the sub-ranges, ie. bin_i = (l,u], where u = array[i].
   patterned after find_sivec_insert_index() in amdmex. */
int bsearch_dyv(dyv *array, double value)
{
  int size = dyv_size(array), result;

  if ( size == 0 ) result = 0;
  else {
    double loval = dyv_ref(array,0);

    if ( value <= loval ) result = 0;
    else {
      int lo = 0, hi = size-1;
      double hival = dyv_ref(array,hi);

      if ( value > hival ) result = size;
      else {
        while ( hi > lo + 1 ) {
          int mid = (lo + hi) / 2;
          double midval = dyv_ref(array,mid);

          if ( midval < value ) { lo = mid; loval = midval; }
          else { hi = mid; hival = midval; }
        }
        if ( loval == value ) result = lo;
        else result = hi;
      }
    }
  }
  return result;
}

/* sample uniformly from a union of nodesets.
   note: nsamples_block is assumed to divide nsamples */
double sample_npt_union0(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
                         sheap *fsh, bool use_symmetry,bool use_permutes,
                         imat *permutation_cache,
                         dyv *wresult,dyv *wsum, dyv *wsumsq,
                         imat *known_ndpairs,dym *known_dists,ivec **maps,
                         int *starts, double nsamples)
{
  double nmatches, ntuples_so_far; int i, nsamples_block = 10;
  dyv *bins = mk_dyv(sheap_size(fsh)); 

  /* set up for sampling by proportion of total ntuples, using binary search */
  ntuples_so_far = 0.0;
  for (i=0; i<sheap_size(fsh); i++) {
    snodeset *sns = (snodeset*)(((sheap_elt*)(fsh->hp->data[i]))->data);
    ntuples_so_far += sns->ntuples;
    dyv_set(bins,i,ntuples_so_far);
  }

  nmatches = 0.0; i = 0;
  while (i*nsamples_block < nsamples) {

    double x = range_random(0.0,ntuples_so_far);
    int b = bsearch_dyv(bins, x);
    snodeset *sns = (snodeset*)(((sheap_elt*)(fsh->hp->data[b]))->data);
    nmatches += sample_npt(ms,xs,ws,ma,sns->kns,use_symmetry,
                           use_permutes,permutation_cache,
                           //wtemp_result,wtemp_sum,wtemp_sumsq,
                           NULL,NULL,NULL,NULL,NULL,maps,starts, 
                           nsamples_block);
    i++;
  }

  free_dyv(bins);
  return nmatches;
}

dyv *mk_compute_proportions(sheap *fsh, double ntuples_total)
{
  int i; dyv *proportions = mk_dyv(sheap_size(fsh)); 

  for (i=0; i<sheap_size(fsh); i++) {
    snodeset *sns = (snodeset*)(((sheap_elt*)(fsh->hp->data[i]))->data);
    dyv_set(proportions, i, sns->ntuples / ntuples_total);
  }

  return proportions;
}

double sample_npt_union(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
                        sheap *fsh, dyv *proportions,
                        bool use_symmetry,bool use_permutes,
                        imat *permutation_cache,
                        dyv *wresult,dyv *wsum, dyv *wsumsq,
                        imat *known_ndpairs,dym *known_dists,ivec **maps,
                        int *starts, double nsamples)
{
  double nmatches; int i;

  nmatches = 0.0; 
  for (i=0; i<sheap_size(fsh); i++) {
    snodeset *sns = (snodeset*)(((sheap_elt*)(fsh->hp->data[i]))->data);
    nmatches += sample_npt(ms,xs,ws,ma,sns->kns,use_symmetry,
                           use_permutes,permutation_cache,
                           //wtemp_result,wtemp_sum,wtemp_sumsq,
                           NULL,NULL,NULL,NULL,NULL,maps,starts, 
                           ceil(nsamples * dyv_ref(proportions,i)));
  }
  return nmatches;
}

/******************************************************************************/
////////////////
////////// MULTI 
////////////////
/******************************************************************************/

nouts *mk_multi_run_npt2(twinpack *tp,params *ps,string_array *matcher_strings)
{
  nouts *ns = mk_empty_nouts();
  int i;
  for ( i = 0 ; i < string_array_size(matcher_strings) ; i++ )
  {
    char *matcher_string = string_array_ref(matcher_strings,i);
    matcher *ma = mk_matcher_from_string(ps->n,tp->dp_data->mps->metric,
					 matcher_string);
    nout *no = mk_run_npt_from_twinpack(tp,ps,ma,NULL);
    //nout *no = mk_run_npt_from_twinpack(tp,ps,ma);
    add_to_nouts(ns,no);
    free_nout(no);
    free_matcher(ma);
  }
  return ns;
}

bool matcher_test_point_pair2(matcher *ma,dym *x1,dym *x2,int row1,int row2,
                              int tuple_index_1,int tuple_index_2,
                              dym *known_dists)
{
  bool matches;
  double known_dsqd, dsqd, dsqd_hi, dsqd_lo = 7e77;

  if (known_dists != NULL) {
    known_dsqd = dym_ref(known_dists,row1,row2);
    if (known_dsqd != UNKNOWN) dsqd = known_dsqd;
    else dym_set(known_dists,row1,row2,
                 dsqd = row_metric_dsqd(x1,x2,ma->metric,row1,row2));
  } else dsqd = row_metric_dsqd(x1,x2,ma->metric,row1,row2);

  if ( ma -> compound )
  {
    dsqd_hi = dym_ref(ma->compound_hi,tuple_index_1,tuple_index_2);
    if ( ma -> between )
      dsqd_lo = dym_ref(ma->compound_lo,tuple_index_1,tuple_index_2);
  }
  else
  {
    dsqd_hi = ma -> dsqd_hi;
    if ( ma -> between )
      dsqd_lo = ma -> dsqd_lo;
  }

  matches = dsqd <= dsqd_hi;
  if ( ma->between && matches )
    matches = dsqd >= dsqd_lo;

  return matches;
}

bool matcher_permute_test_point_pair2(matcher *ma,dym *x1,dym *x2,
                                      int row1,int row2,
                                      int pt_tuple_index_1,int pt_tuple_index_2,
                                      imat *permutation_cache,
                                      ivec *permutes_ok,dym *known_dists)
{
  int i;
  bool matches;
  double known_dsqd, dsqd, dsqd_hi, dsqd_lo = 7e77;
  bool any_matches = FALSE;
  int template_tuple_index_1,template_tuple_index_2;

  if (known_dists != NULL) {
    known_dsqd = dym_ref(known_dists,row1,row2);
    if (known_dsqd != UNKNOWN) dsqd = known_dsqd;
    else dym_set(known_dists,row1,row2,
                 dsqd = row_metric_dsqd(x1,x2,ma->metric,row1,row2));
  } else dsqd = row_metric_dsqd(x1,x2,ma->metric,row1,row2);

  /* make sure the cache is ready */
  if (permutation_cache && imat_cols(permutation_cache) != matcher_n(ma))
  {
    free_imat(permutation_cache);
    permutation_cache = NULL;
  }
  if (!permutation_cache) 
    permutation_cache = mk_permutation_cache(matcher_n(ma));

  /* check all permutations */
  for (i=0;i<ivec_size(permutes_ok);i++)
  {
    if (!ivec_ref(permutes_ok,i)) continue;

    /* check where these points are in the i'th permutation */
    template_tuple_index_1 = imat_ref(permutation_cache,i,pt_tuple_index_1);
    template_tuple_index_2 = imat_ref(permutation_cache,i,pt_tuple_index_2);

    if ( ma -> compound )
    {
      dsqd_hi = dym_ref(ma->compound_hi,
			template_tuple_index_1,template_tuple_index_2);
      if ( ma -> between )
	dsqd_lo = dym_ref(ma->compound_lo,
			  template_tuple_index_1,template_tuple_index_2);
    }
    else
    {
      dsqd_hi = ma -> dsqd_hi;
      if ( ma -> between ) dsqd_lo = ma -> dsqd_lo;
    }

    matches = dsqd <= dsqd_hi;
    if (ma->between && matches) matches = dsqd >= dsqd_lo;

    if (matches) any_matches = TRUE;
    else         ivec_set(permutes_ok,i,FALSE);
  }

  return any_matches;
}

/* call this with root of kd-tree, map ivec already allocated, and 
   *curr_index = 0 */
void create_virtual_index_to_dym_row_map(knode *kn, ivec *map, int *curr_index,
                                         int start_index)
{
  if (knode_is_leaf(kn)) {
    int i;
    for (i=0; i<kn->num_points; i++) {
      int dym_row = ivec_ref(kn->rows, i);
      ivec_set(map, *curr_index - start_index, dym_row);
      //printf("virtual: %d  dym_row: %d\n",*curr_index,dym_row);
      (*curr_index)++;
    }
  } else {
    create_virtual_index_to_dym_row_map(kn->left,map,curr_index,start_index);
    create_virtual_index_to_dym_row_map(kn->right,map,curr_index,start_index);
  }
}

/* call this with root of kd-tree, map ivec already allocated, and 
   *curr_index = 0 */
void create_dym_row_to_virtual_index_map(knode *kn, ivec *map, int *curr_index)
{
  if (knode_is_leaf(kn)) {
    int i;
    for (i=0; i<kn->num_points; i++) {
      int dym_row = ivec_ref(kn->rows, i);
      ivec_set(map, dym_row, *curr_index);
      //printf("dym_row: %d  virtual: %d\n",dym_row,*curr_index);
      (*curr_index)++;
    }
  } else {
    create_dym_row_to_virtual_index_map(kn->left,map,curr_index);
    create_dym_row_to_virtual_index_map(kn->right,map,curr_index);
  }
}
