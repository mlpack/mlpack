/*
   File:        npt3.c
   Author:      Alexander Gray
   Description: Faster N-point computation II:
                priority search
                geometric stratified sampling
*/

//For now, this .c is just included by npt.c, since this shouldn't stay a 
//separate module.  Otherwise we'd need the following.
//#include "npt2.h"
//extern int Next_n;

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
extern double Avg_num_tries;

int Num_expansions = 0;

double fast_npt3(sheap *sh,
                 mapshape *ms,dym **xs,dym **ws,matcher *ma,
                 bool use_symmetry,bool use_permutes,
                 double thresh_ntuples,double connolly_thresh,
                 double *lobound,double *hibound, 
                 dyv *wlobound, dyv *whibound,dyv *wresult,dyv *wsum,dyv *wsumsq,
                 imat *permutation_cache, ivec **maps, int *starts)
{
 sheap *ssh = mk_sheap(); 

 while(!sheap_is_empty(sh)) {

  nodeset *ns = (nodeset*)get_from_sheap(sh);
  knode *kns[MAX_N];

  //bool do_weights = (wresult && wlobound && whibound && ns->kns[0]->sum_weights);
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

  int num_excludes[MAX_N]; /* Eventually, num_excludes[i] will 
                              contain the number of other nodes that
                              kns[i] excludes. */
  int total_num_excludes = 0;

  //bool recursed = FALSE; /* Flag used only for deciding when to printf stuff */

  //double ntuples = total_num_ntuples(n,(use_symmetry || use_permutes),ns->kns);
  double ntuples = ns->ntuples;
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

  //     double a = printf("bounds: [%g, %g] errfrac = %g\n",*lobound,*hibound,
  //                       errfrac);
  //     double nsamples = 1000000;
  //     double nmatches = sample_npt(ms,xs,ws,ma,ns->kns,use_symmetry,
  //                                  use_permutes,permutation_cache,
  //                                  //wtemp_result,wtemp_sum,wtemp_sumsq,
  //                                  NULL,NULL,NULL,NULL,NULL,maps,starts,
  //                                  nsamples);
  //     double p = nmatches/nsamples;
  //     double sd = sqrt(p*(1.0-p))/sqrt(nsamples);
  //     printf("p = %g, sd = %g\n",p,sd);
  //     *lobound += p*ntuples - Sig*sd*ntuples;
  //     *hibound += p*ntuples + Sig*sd*ntuples - ntuples;
  //     return (*hibound - *lobound)/2;

  Num_expansions++;

  if (errfrac <= Eps) { printf("errfrac reached!\n"); break; }

  ///////////// delete this pq entry
  for (i=0; i<n; i++) kns[i] = ns->kns[i];
  free_nodeset(ns,n);

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
      printf(" datafrac %9.8f",datafrac);
      printf("\n");
      //Next_n *= 2;
      Next_n += 10000000;
    }

  for ( i = 0 ; !not_worth_it && i < n ; i++ ) { 
    num_subsumes[i] = 0; num_excludes[i] = 0; 
  }

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

    /* Took out redundancy checking because the memory cost of storing all state
       is too high.  AG */
    for ( j = i+1 ; j < n ; j++ )
    { 
      knode *knj = kns[j];
      int status;

      if (use_symmetry || !use_permutes)
      {
        status = matcher_test_hrect_pair(ma,kni->hr,knj->hr,i,j);
        
        if (status == EXCLUDE) {
          answer_is_zero = TRUE;
          num_excludes[i] += 1; num_excludes[j] += 1;
        }
        else if ( status == SUBSUME ) {
          num_subsumes[i] += 1; num_subsumes[j] += 1;
        }
        else all_subsume = FALSE;
      }
      else 
      {
        status = matcher_permute_test_hrect_pair(ma,kni->hr,knj->hr,i,j,
                                                 permute_status,
                                                 num_incons,
                                                 permutation_cache);
        if (status == EXCLUDE) {
          answer_is_zero = TRUE;
          num_excludes[i] += 1; num_excludes[j] += 1;
        }
      }
    }
  }

  for ( i = 0 ; i < n ; i++ ) total_num_excludes += num_excludes[i];

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

//  if (0) {
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
//{}}
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
    //int min_num_subsumes = n;
    //
    //for ( i = 0 ; i < n ; i++ )
    //{
    //  double ns = num_subsumes[i];
    //  int size = kns[i]->num_points;
    //  if ( !knode_is_leaf(kns[i]) ) {
    //    if (ns < min_num_subsumes) {
    //      split_index = i; split_index_num_points = size; 
    //      min_num_subsumes = ns;
    //    } else 
    //    if ((ns == min_num_subsumes) && (size > split_index_num_points)) {
    //      split_index = i; split_index_num_points = size; 
    //      min_num_subsumes = ns;
    //    }      
    //  }
    //}
    
    for ( i = 0 ; i < n ; i++ )
    {
      bool subsume_all = num_subsumes[i] == n-1;
      if ( !subsume_all && !knode_is_leaf(kns[i]) )
      {
        int num_points = kns[i]->num_points;
        if ( split_index < 0 || num_points > split_index_num_points )
        {
          split_index = i;
          split_index_num_points = num_points;
        }
      }
    }
    
    /* The only way that split_index could be undefined is if all 
       non-all-subsuming knodes were leaves */
    
    if ( split_index < 0 ) {
      /* We failed to find a non-subsuming non-leaf. Now we'll be happy
         with the largest non-leaf whether it sumbsumes or not. */
    
      for ( i = 0 ; i < n ; i++ ) {
        if ( !knode_is_leaf(kns[i]) ) {
          int num_points = kns[i]->num_points;
          if ( split_index < 0 || num_points > split_index_num_points ) {
            split_index = i;
            split_index_num_points = num_points;
          }
        }
      }
    }
      
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
        result = slow_npt(ms,xs,ws,ma,use_symmetry,NONE,NONE,rowsets,
                          wtemp_result,wtemp_sum,wtemp_sumsq);
      else
        result = slow_permute_npt(ms,xs,ws,ma,NONE,NONE,rowsets,
                                  permutation_cache,
                                  wtemp_result,wtemp_sum,wtemp_sumsq);
    
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
      bool stop_expanding = (Num_to_expand > 0) ? 
        (Num_expansions >= Num_to_expand) : (datafrac > Datafrac_crit);
      //bool stop_expanding = (Num_to_expand > 0) ? 
      //  (Num_expansions >= Num_to_expand) : (errfrac < Rerrfrac_crit);

      if (stop_expanding) {
        double nsamples = 100;
        double nmatches = sample_npt(ms,xs,ws,ma,kns,use_symmetry,use_permutes,
                              permutation_cache,
                              //wtemp_result,wtemp_sum,wtemp_sumsq,
                              NULL,NULL,NULL,NULL,NULL,maps,starts,nsamples);
        enqueue_snodeset(ssh,kns,n,ntuples,nmatches,nsamples,REGULAR);
        //recursed = TRUE;
      }
      else
      {
        //if ( !Do_rectangle_animation && Verbosity >= 1.0 )
        //{
        //  printf("About to recurse. lobound=%g, hibound=%g\n",
        //         *lobound,*hibound);
        //  wait_for_key();
        //}
  
        kns[split_index] = child2;
      
        //ntuples = total_num_ntuples(n,use_symmetry,kns);
        ntuples = total_num_ntuples(n,(use_symmetry || use_permutes),kns);
        //ntuples = total_num_ntuples(n,use_permutes,kns);
        //ntuples = total_num_ntuples(n,(use_symmetry || !use_permutes),kns);
        //ntuples = total_num_ntuples(n,use_permutes,kns);
        //ntuples = total_num_ntuples_assymmetric(n,kns);
        if (ntuples != 0) enqueue_nodeset(sh,kns,n,ntuples);
      
        kns[split_index] = child1;
      
        //ntuples = total_num_ntuples(n,use_symmetry,kns);
        ntuples = total_num_ntuples(n,(use_symmetry || use_permutes),kns);
        //ntuples = total_num_ntuples(n,use_permutes,kns);
        //ntuples = total_num_ntuples(n,(use_symmetry || !use_permutes),kns);
        //ntuples = total_num_ntuples(n,use_permutes,kns);
        //ntuples = total_num_ntuples_assymmetric(n,kns);
        if (ntuples != 0) enqueue_nodeset(sh,kns,n,ntuples);
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
 }
 printf("Num expansions: %d\n",Num_expansions);

 /* NOW SAMPLE FROM THE REMAINING NODESETS */
 {
   double global_ntuples = 0.0, global_nsamples = 0.0;
   double global_estimate = 0.0, global_variance = 0.0, global_sd = 0.0;
   double global_sum_sd = 0.0, last_global_sum_sd = 0.0;
   double hard_lobound = *lobound, hard_hibound = *hibound;
   double errfrac = compute_errfrac(*lobound,*hibound);
   int orig_num_nodesets = sheap_size(ssh), i;
   double smallfries_ntuples  = 0.0, bigcheeses_ntuples  = 0.0;
   double smallfries_nmatches = 0.0, bigcheeses_nmatches = 0.0;
   double smallfries_nsamples = 0.0, bigcheeses_nsamples = 0.0;
   dyv *smallfries_props = NULL, *bigcheeses_props = NULL;
   sheap *smallfries = mk_sheap(), *bigcheeses = mk_sheap();
   sheap *regulars = mk_sheap();

   //for (i=0; i<sheap_size(ssh); i++) {
   //  snodeset *sns = (snodeset*)(((sheap_elt*)(ssh->hp->data[i]))->data);
   //  double p = sns->nmatches / sns->nsamples;
   //  printf("ntuples = %g, nmatches = %g, nsamples = %g, p = %g (%s)\n",
   //         sns->ntuples,sns->nmatches,sns->nsamples,p,
   //         sns->type==REGULAR ? "reg" : 
   //         (sns->type==SMALLFRIES ? "SMALLFRIES" : "BIGCHEESES")); 
   //}

   /* First separate out the smallfries and bigcheeses */
   if (Union_p > -1) {
     for (i=0; i<orig_num_nodesets; i++) {
       snodeset *sns = (snodeset*)get_from_sheap(ssh);
       double p = sns->nmatches / sns->nsamples;
       if (p <= Union_p) {
         add_to_sheap(smallfries,(void *)sns,0);
         smallfries_ntuples += sns->ntuples; 
         smallfries_nmatches += sns->nmatches; 
         smallfries_nsamples += sns->nsamples;
       }
       else if (p >= 1.0 - Union_p) {
         add_to_sheap(bigcheeses,(void *)sns,0);
         bigcheeses_ntuples += sns->ntuples; 
         bigcheeses_nmatches += sns->nmatches; 
         bigcheeses_nsamples += sns->nsamples;
       }
       else add_to_sheap(regulars,(void *)sns,0);
     }
     free_sheap(ssh); ssh = regulars;
     if (smallfries_ntuples > 0) {
       add_to_sheap(ssh,(void *)mk_snodeset(NULL,matcher_n(ma),
                                            smallfries_ntuples,
                                            smallfries_nmatches,
                                            smallfries_nsamples,SMALLFRIES),0);
       smallfries_props = mk_compute_proportions(smallfries,smallfries_ntuples);
     }
     if (bigcheeses_ntuples > 0) {
       add_to_sheap(ssh,(void *)mk_snodeset(NULL,matcher_n(ma),
                                            bigcheeses_ntuples,
                                            bigcheeses_nmatches,
                                            bigcheeses_nsamples,BIGCHEESES),0);
       bigcheeses_props = mk_compute_proportions(bigcheeses,bigcheeses_ntuples);
     }
   }

   /* Do initial scan to set the sampling proportions */
   global_ntuples = 0.0; global_nsamples = 0.0; global_sum_sd = 0.0;
   for (i=0; i<sheap_size(ssh); i++) {
     snodeset *sns = (snodeset*)(((sheap_elt*)(ssh->hp->data[i]))->data);
     double prior_sd = sqrt(Force_p*(1.0-Force_p)) / sqrt(sns->nsamples);
     double p = sns->nmatches / sns->nsamples;
     if (p == 0) global_sum_sd += sns->ntuples / ( 2.0 * sqrt(sns->nsamples) );
     else global_sum_sd += sns->ntuples * real_max(prior_sd, 
                                 ( sqrt(p*(1.0-p)) / sqrt(sns->nsamples) ));
     global_ntuples += sns->ntuples; global_nsamples += sns->nsamples;
     //  printf("ntuples = %g, nmatches = %g, nsamples = %g, p = %g (%s)\n",
     //         sns->ntuples,sns->nmatches,sns->nsamples,p,
     //         sns->type==REGULAR ? "reg" : 
     //         (sns->type==SMALLFRIES ? "SMALLFRIES" : "BIGCHEESES")); 
     //if (i>=4) { printf("...printing top 5 only\n"); break; }
   }
   printf("Number of nodesets sampled from = %d\n",sheap_size(ssh));
   printf("  Compiled into smallfries: %d\n",sheap_size(smallfries));
   printf("  Compiled into bigcheeses: %d\n",sheap_size(bigcheeses));
   printf("BOUNDS: %g [%g, %g] errfrac = %g at %d secs\n",
          real_max(0.0,(*lobound+*hibound)/2.0),real_max(0.0,*lobound),*hibound,
          errfrac,global_time()-Start_secs);
   if (errfrac <= Eps) return (*hibound - *lobound)/2.0;

   /* Iteratively allocate numbers-of-samples and do stratified sampling */
   do {
     global_estimate = 0.0; global_variance = 0.0; global_sd = 0.0;
     last_global_sum_sd = global_sum_sd; global_sum_sd = 0.0;

     for (i=sheap_size(ssh)-1; i>=0; i--) {

       /* Get old sampling statistics from snodeset */
       snodeset *sns = (snodeset*)(((sheap_elt*)(ssh->hp->data[i]))->data);
       double ntuples = sns->ntuples;
       double last_nsamples = sns->nsamples, last_nmatches = sns->nmatches;
       //double prior_sd = sqrt(Force_p*(1.0-Force_p)) / sqrt(last_nsamples+s2);
       //double last_p = (last_nmatches + s2/2.0) / (last_nsamples + s2);
       //double last_sd = real_max(prior_sd, sqrt(last_p*(1.0-last_p)))
       //                 / sqrt(last_nsamples + s2);
       double last_p = last_nmatches / last_nsamples;
       double last_sd  = (last_p == 0.0) ? 1.0/(2.0*sqrt(last_nsamples)) :
         real_max(( sqrt(Force_p*(1.0-Force_p))/sqrt(last_nsamples) ),
                  ( sqrt(last_p*(1.0-last_p)) / sqrt(last_nsamples) ));

       /* Compute number of samples using stdev fraction */
       //double neyman_optimal_fraction = ntuples/global_ntuples;
       double neyman_optimal_fraction = (ntuples*last_sd)/last_global_sum_sd;
       double nsamples = ceil(neyman_optimal_fraction * Nsamples_block);
       //double nsamples = ceil(Nsamples_block/sheap_size(ssh));
       //double nsamples = ceil(Nsamples_block*(ntuples/global_ntuples));

       /* Sample */
       double nmatches = sns->type == REGULAR ? 
                         sample_npt(ms,xs,ws,ma,sns->kns,use_symmetry,
                                    use_permutes,permutation_cache,
                                    //wtemp_result,wtemp_sum,wtemp_sumsq,
                                    NULL,NULL,NULL,NULL,NULL,maps,starts,
                                    nsamples) :
                         (sns->type == SMALLFRIES ? 
                          sample_npt_union(ms,xs,ws,ma,
                                    smallfries,smallfries_props,
                                    use_symmetry,use_permutes,permutation_cache,
                                    //wtemp_result,wtemp_sum,wtemp_sumsq,
                                    NULL,NULL,NULL,NULL,NULL,maps,starts,
                                    nsamples) :
                          sample_npt_union(ms,xs,ws,ma,
                                    bigcheeses,bigcheeses_props,
                                    use_symmetry,use_permutes,permutation_cache,
                                    //wtemp_result,wtemp_sum,wtemp_sumsq,
                                    NULL,NULL,NULL,NULL,NULL,maps,starts,
                                    nsamples));

       /* Update to get new sampling statistics */
       double new_nsamples = last_nsamples + nsamples;
       double new_nmatches = last_nmatches + nmatches;
       //double new_p = (new_nmatches + s2/2.0) / (new_nsamples + s2);
       //double new_sd = sqrt(new_p*(1.0-new_p)) / sqrt(new_nsamples + s2);
       double new_p = new_nmatches / new_nsamples;
       double new_sd = (new_p == 0.0) ? 1.0/(2.0*sqrt(new_nsamples)) :
         real_max(( sqrt(Force_p*(1.0-Force_p))/sqrt(new_nsamples) ),
                  ( sqrt(new_p*(1.0-new_p))/sqrt(new_nsamples) ));

       /* Store new sampling statistics in snodeset */
       sns->nmatches = new_nmatches; sns->nsamples = new_nsamples; 

       //printf("  ntuples = %g, nmatches = %g, nsamples = %g, p = %g (%s)\n",
       //       sns->ntuples,sns->nmatches,sns->nsamples,new_p,
       //       sns->type==REGULAR ? "reg" : 
       //       (sns->type==SMALLFRIES ? "SMALLFRIES" : "BIGCHEESES")); 

       /* Add to global estimate and variance */
       global_estimate += new_p * ntuples;
       global_sum_sd += new_sd * ntuples;
       global_variance += (new_sd * ntuples) * (new_sd * ntuples);
       global_nsamples += nsamples;

       /* Update global bounds incrementally - this could allow us to stop
          early; but we shouldn't be wasting much time on worthless snodesets
          anyway */
       //if ( (p-sd >= 0.0) && (p+sd <= 1.0) ) {
       //  double result = p * ntuples;
       //  if ( (last_p-last_sd >= 0.0) && (last_p+last_sd <= 1.0) ) {
       //    *lobound -= (last_result - last_sd*ntuples);
       //    *hibound -= (last_result + last_sd*ntuples - ntuples);
       //  }
       //  *lobound += (result - sd*ntuples);
       //  *hibound += (result + sd*ntuples - ntuples);
       //  global_p += result;
       //  global_nsamples += nsamples;
       //} else {
       //  global_p += last_result;
       //  global_nsamples += last_nsamples;
       //}

       //printf("nodeset %d:\n",i);
       //printf("  last_nmatches = %g, nmatches = %g, new_nmatches = %g\n",
       //       last_nmatches,nmatches,new_nmatches);
       //printf("  last_nsamples = %g, nsamples = %g, new_nsamples = %g\n",
       //       last_nsamples,nsamples,new_nsamples);
       //printf("  last_p = %g, new_p = %g\n\n",last_p,new_p);

     }

     global_sd = sqrt( global_variance ); 
     *lobound = hard_lobound + (global_estimate - Sig*global_sd);
     *hibound = hard_hibound + (global_estimate + Sig*global_sd - 
                                global_ntuples);
     errfrac = compute_errfrac(*lobound,*hibound);
     printf("BOUNDS: %g [%g, %g] errfrac = %g at %d secs\n",
            real_max(0.0,(*lobound+*hibound)/2.0),real_max(0.0,*lobound),
            *hibound,errfrac,global_time()-Start_secs);

     Nsamples_block *= 2;

   } while (errfrac > Eps);
   //} while (0);

   /* final diagnostic printout */
   //for (i=0; i<sheap_size(ssh); i++) {
   //  snodeset *sns = (snodeset*)(((sheap_elt*)(ssh->hp->data[i]))->data);
   //  double p = (sns->nmatches) / (sns->nsamples);
   //  printf("ntuples = %g, nmatches = %g, nsamples = %g, p = %g\n",
   //         sns->ntuples,sns->nmatches,sns->nsamples,p); 
   //}

   // note: this is the effective p we sampled from, not including hard_lobound
   printf("global_estimate: %g\n",global_estimate);
   printf("global_p: %g\n",global_estimate/global_ntuples);
   printf("total number of samples used: %g\n",global_nsamples);
   printf("average number of tries per sample: %g\n",
          Avg_num_tries/global_nsamples);

   while(!sheap_is_empty(ssh)) {
     snodeset *sns = (snodeset*)get_from_sheap(ssh);
     free_snodeset(sns,matcher_n(ma));
   }
   if (smallfries_ntuples > 0) {
     while(!sheap_is_empty(smallfries)) {
       snodeset *sns = (snodeset*)get_from_sheap(smallfries);
       free_snodeset(sns,matcher_n(ma));
     }
     free_sheap(smallfries); free_dyv(smallfries_props);
   }
   if (bigcheeses_ntuples > 0) {
     while(!sheap_is_empty(bigcheeses)) {
       snodeset *sns = (snodeset*)get_from_sheap(bigcheeses);
       free_snodeset(sns,matcher_n(ma));
     }
     free_sheap(bigcheeses); free_dyv(bigcheeses_props);
   }
 }

 return (*hibound - *lobound)/2.0;
}

/* Given a sum-of-squares about a mean m_1, compute the sum-of-squares of those
   N points about a new mean m_2, without needing to sum over the original 
   points. */
double shift_mean_of_sumsq(double sumsq1, double m1, double m2, double N)
{
  double sumsq2 = sumsq1 + (m1-m2)*(2*m1 - N*m1 - N*m2);
  return sumsq2;
}

nodeset *mk_nodeset(knode **kns, int n, double ntuples)
{
  int i;
  nodeset *ns = AM_MALLOC(nodeset);
  ns->ntuples = ntuples;
  ns->kns = AM_MALLOC_ARRAY(knode*,n);
  for (i=0; i<n; i++) ns->kns[i] = kns[i];
  return ns;
}

snodeset *mk_snodeset(knode **kns, int n, double ntuples, double nmatches,
                      double nsamples, char type)
{
  int i;
  snodeset *sns = AM_MALLOC(snodeset);
  sns->ntuples = ntuples; sns->nmatches = nmatches; sns->nsamples = nsamples;
  sns->type = type;
  if (type == REGULAR) {
    sns->kns = AM_MALLOC_ARRAY(knode*,n); 
    for (i=0; i<n; i++) sns->kns[i] = kns[i];
  }
  return sns;
}

void free_nodeset(nodeset *ns,int n)
{
  AM_FREE_ARRAY(ns->kns,knode*,n);
  AM_FREE(ns,nodeset);
}

void free_snodeset(snodeset *sns,int n)
{
  if (sns->type == REGULAR) AM_FREE_ARRAY(sns->kns,knode*,n);
  AM_FREE(sns,snodeset);
}

//for use with fibheaps
//int compare_nodesets(void * x, void * y)
//{
//  double a = ((nodeset*)x)->ntuples * ((nodeset*)x)->p * -1.0;
//  double b = ((nodeset*)y)->ntuples * ((nodeset*)y)->p * -1.0;
//  if (a < b) return -1;
//  if (a == b) return 0;
//  return 1;
//}

void enqueue_nodeset(sheap *sh,knode **kns,int n,double ntuples)
{
  double h = ntuples * -1.0; bool bad = FALSE;

  /* enforce the node ordering - but ttn() should handle this */
  //int i;
  //for ( i = 0 ; i < n ; i++ ) 
  //  if ((i>0) && (kns[i]->hi_index < kns[i-1]->lo_index)) { bad=TRUE; break; }

  //printf("enqueueing nodeset: (%d,%d) (%d,%d) (%d,%d)\n",
  //       kns[0]->lo_index,kns[0]->hi_index,kns[1]->lo_index,kns[1]->hi_index,
  //       kns[2]->lo_index,kns[2]->hi_index);

  if (!bad) 
    add_to_sheap(sh, (void *)mk_nodeset(kns,n,ntuples), h);
    //fh_insert(fh, (void *)mk_nodeset(kns,n,ntuples));
  else printf("heeeey now!!\n");
}

void enqueue_snodeset(sheap *sh,knode **kns,int n,double ntuples, 
                      double nmatches, double nsamples, char type)
{
  double h = 0.0; bool bad = FALSE;

  /* enforce the node ordering - but ttn() should handle this */
  //int i;
  //for ( i = 0 ; i < n ; i++ ) 
  //  if ((i>0) && (kns[i]->hi_index < kns[i-1]->lo_index)) { bad=TRUE; break; }

  //printf("enqueueing SNODESET: (%d,%d) (%d,%d) (%d,%d)\n",
  //       kns[0]->lo_index,kns[0]->hi_index,kns[1]->lo_index,kns[1]->hi_index,
  //       kns[2]->lo_index,kns[2]->hi_index);

  if (!bad) 
    add_to_sheap(sh,(void *)mk_snodeset(kns,n,ntuples,nmatches,nsamples,type),h);
    //fh_insert(fh, (void *)mk_snodeset(kns,n,ntuples,nmatches,nsamples,type));
  else printf("heeeey now!!\n");
}

/* Gets the minimum priority item from the heap, also popping it - but 
   the item still needs to be freed later. */
void *get_from_sheap(sheap *sh)
{
  if (sheap_is_empty(sh)) return NULL;
  else {
    void *item = sheap_data_with_minimum_priority(sh);
    sheap_remove_minimum_priority(sh);
    return item;
  }
}
  
bool sheap_is_empty(sheap *sh)
{
  return (sheap_size(sh)==0);
}

