/*
   File:        npt.c
   Author:      Andrew W. Moore
   Created:     Wed May 17 12:25:12 EDT 2000
   Description: Fast N-point computation

   Copyright 2000, the Auton Lab
*/

#include "npt.h"
#include "npt2.h"
#include "npt3.h"
#include "projnpt.h"
#include <assert.h>

bool   Draw_joiners = FALSE;
bool   Use_Npt2 = FALSE;   /* AG */
bool   Use_Npt3 = FALSE;   /* AG */
bool   Use_MC = FALSE;     /* AG */
int    Projection = NONE;  /* AG */
int    Projmethod = NONE;  /* AG */
double Eps;                /* AG */
double Sig;                /* AG */
double Force_p;            /* AG */
double Union_p;            /* AG */
double Nsamples_block;     /* AG */
double Datafrac_crit;      /* AG */
double Rerrfrac_crit;      /* AG */
int    Num_to_expand;      /* AG */
int    Start_secs;         /* AG */
/* This global variable is merely used for deciding when to next
   print out something to the screen */
int Next_n;
bool Do_rectangle_animation = FALSE;
knode *Old_animated_kns[MAX_N];
bool Old_active = FALSE;
/* Merely used in animating... */
mrkd *Mr_root;
double ntuples_seen_so_far = 0.0;  /* AG
                                      this is incremented upon an exclude,
                                      all-subsume, or all-leaf */
double total_ntuples = 0.0;
double num_not_worth_it_prunes = 0.0;

typedef unsigned long long result_t;

/* Used by the "rdraw t" command line option, but not relevant to
   the basic algorithm.... */
void draw_knodes_boxes(mapshape *ms,knode **kns,int n,int agcol)
{
  int i;
  ag_set_pen_color(agcol);
  for ( i = 0 ; i < n ; i++ )
  {
    ms_hrect_box(ms,kns[i]->hr);
    ms_mark(ms,hrect_middle_ref(kns[i]->hr,0),hrect_middle_ref(kns[i]->hr,1),
	    DOT_MARKTYPE);
  }
}

/* Used by the "rdraw t" command line option, but not relevant to
   the basic algorithm.... */

/* Used by the "rdraw t" command line option, but not relevant to
   the basic algorithm.... */
void rectangle_animate(mapshape *ms,knode **kns,int n,int agcol)
{
  int i;
  if ( Old_active )
    draw_knodes_boxes(ms,Old_animated_kns,n,AG_WHITE);
  draw_knodes_boxes(ms,kns,n,agcol);
  for ( i = 0 ; i < n ; i++ )
  {
    Old_animated_kns[i] = kns[i];
    Old_active = TRUE;
  }
}

/* Used by the "high verbosity" animation code, but not relevant to
   the true function... */
void draw_lettered_knode_points(mapshape *ms,knode *kn,dym *x,char *s)
{
  if ( knode_is_leaf(kn) )
  {
    int i;
    for ( i = 0 ; i < kn -> num_points ; kn++ )
    {
      int row = ivec_ref(kn->rows,i);
      ms_point_in_dym_colored(ms,x,row,DOT_MARKTYPE,AG_YELLOW);
      ms_point_in_dym_colored_string(ms,x,row,s,AG_BLACK);
      s[0] += 1;
    }
  }
  else
  {
    draw_lettered_knode_points(ms,kn->left,x,s);
    draw_lettered_knode_points(ms,kn->right,x,s);
  }
}

void draw_lettered_mrkd_points(mapshape *ms,mrkd *mr,dym *x)
{
  char s[100];
  s[0] = 'a';
  s[1] = '\0';
  draw_lettered_knode_points(ms,mr->root,x,s);
}

ivec *mk_rowset(knode *kn)
{
  if (kn->rows) return mk_copy_ivec(kn->rows);
  else
  {
    ivec *lrows = mk_rowset(kn->left);
    ivec *rrows = mk_rowset(kn->right);
    ivec *both = mk_append_ivecs(lrows,rrows);
    free_ivec(lrows);
    free_ivec(rrows);
    return both;
  }
}

/* This function is not general in any way and should not be reused */
void special_weighted_symmetric_debugging_test(dym **xs,dym **ws,matcher *ma,
					       knode **kns,dyv *wresult)
{
  int i,j,k;
  int n = matcher_n(ma);
  dyv *acc = mk_zero_dyv(dyv_size(wresult));
  dyv *tmp = mk_zero_dyv(dyv_size(wresult));
  ivec *r0 = mk_rowset(kns[0]);
  ivec *r1 = mk_rowset(kns[1]);
  ivec *r2 = mk_rowset(kns[2]);

  if (n != 3)
  {
    printf("special_weighted_symmetric_debugging_test: only works for n=3\n");
    really_wait_for_key();
    return;
  }

  zero_dyv(wresult);
  for (i=0;i<ivec_size(r0);i++)
    for (j=0;j<ivec_size(r1);j++)
      for (k=0;k<ivec_size(r2);k++)
	if (((kns[0]->lo_index+i) < (kns[1]->lo_index+j)) && 
	    ((kns[1]->lo_index+j) < (kns[2]->lo_index+k)))
	{
	  int row0 = ivec_ref(r0,i);
	  int row1 = ivec_ref(r1,j);
	  int row2 = ivec_ref(r2,k);
	  copy_dym_row_to_dyv(ws[0],acc,row0);
	  copy_dym_row_to_dyv(ws[1],tmp,row1);
	  dyv_mult(acc,tmp,acc);
	  copy_dym_row_to_dyv(ws[2],tmp,row2);
	  dyv_mult(acc,tmp,acc);
	  dyv_plus(wresult,acc,wresult);
	}
  free_ivec(r0);
  free_ivec(r1);
  free_ivec(r2);
  free_dyv(acc);
  free_dyv(tmp);
}

/* 
 * ANG ~> Rewritting slow_permute_npt_helper in a non-recursive manner.
 *
 * First stage for getting rid of recursion is to embedd everything in a big
 * loop. The following iterative backtracking algorithm applies:
 *
 * while (k >= 0) {
 * 	while ( !ok(k_th_point) && can_try_again(k) ) {
 * 		find_new_point(k);
 * 	}
 * 	if ( ok(k_th_point) ) {
 * 		if (k == (n-1) ) {
 * 			increase_count();
 * 		}
 * 		else {
 * 			k++;
 * 			reset(k_th_point);
 * 		}
 * 	}
 * 	else {
 * 		k--;
 * 	} 	
 * }
 * 
 * For each k we assume that the previous k-1 points have been selected in a
 * non-conflicting manner and that they all satisfy the matcher. Thus, we
 * proceed to select a new point from the k-th leaf in a non-colflicting
 * fashion. We stop trying either when we find a 'good' point or when we have
 * tried all the points.
 * 
 * If we managed to find a 'good' point we check to see if this is in fact an
 * n-tuple. If so we need to update our counts from the unweighted and for the
 * weighted cases. If not we simply proceed to select a new point from the next
 * knode.
 * 
 * If at some point we have run out of possible points in a knode our only
 * solution is to ge to the previous knode and try to find a new point there.
 * 
 * When all the possible n-tuples of points have been tested we cannot find any
 * more new knodes and k will be decreased below the minimum possible level.
 * This is how we check that the matching process is finished.
 */

double iterative_slow_npt_helper(mapshape *ms,dym **xs,dym **ws, matcher *ma, 
                       bool use_symmetry, int projection, int projmethod,
                       int *row_indexes,ivec **rowsets,dyv *wresult,
                       dyv *wsum, dyv *wsumsq)

{
  int n = matcher_n(ma);
  int k = 0;
  double result = 0.0;

	total_num_iterative_base_cases += 1.0;
	/* Initializing the first point */
	row_indexes[k] = -1;

	while (k >= 0) {
		int new_point_is_ok = 0;
		int can_try_new_point = 1;
		ivec *rowset_k = rowsets[k];

		while ( !new_point_is_ok && can_try_new_point ) {
			if ((row_indexes[k] + 1) >= ivec_size(rowset_k)) {
				/* We cannot try a new point. */
				can_try_new_point = 0;
			}
			else {					
				int current_row;
		    int j, i;

				/* Get the  next point and be optimistic about it */
				row_indexes[k] += 1;
				i = row_indexes[k];
				new_point_is_ok = 1;
//				printf("got here 1\n");
				current_row = ivec_ref(rowset_k, i);
//				printf("got here 2\n");
				
				/* Test it for conflicts against all previously set nodes */
		    for (j = 0; new_point_is_ok && j < k; j++) {
					ivec *rowset_j = rowsets[j];
		      int row_index_j = row_indexes[j];
		      int row_j = ivec_ref(rowset_j,row_index_j);
		      bool k_and_j_rows_from_same_knode = ((long)(rowset_k)) ==	((long)(rowset_j));

					if (row_index_j <= i && k_and_j_rows_from_same_knode && use_symmetry) {
						/* The k-th and j-th points are from the same leaf and they are not
						 * in order. This is a conflict! */
						new_point_is_ok = 0;
						/* This might also warrant setting can_try_new_point = 0 but only if
						 * the points are in some sort of order. I'm not sure about that
						 * though so it's best to forget about it for the time being. It 
						 * should only be a minor time increase if there is some order and 
						 * it will lead to bad results if not. */
						//can_try_new_point = 0;
					}
		      else {
						/* The points are either in order of from different knodes. */
					  if (!matcher_test_point_pair(ma,projection,projmethod,
               xs[k],xs[j],current_row,row_j,k,j))	{
							/* However, if they don't match this is still a conflict. */
							new_point_is_ok = 0;
						}
						/* Stuff about animations I won't care about now. */
/*
		      if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) ) {
    		    if ( new_point_is_ok ) ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
                               (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
		        ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
		        ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
		      }
		      if ( !cutoff && ok && Draw_joiners ) ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
*/

					}
				}
			}
		}

		if ( new_point_is_ok ) {
			if (k == (n-1) ) { /* We got a matching n-tuple. Time to count. */
				int tmp;
				
	      result += 1.0; 
				printf("Iterative base case method found a match. All results are increased.\n");
				/* Extra debugging info */
				printf("The n-tuple was:  ");
				for (tmp=0; tmp<n; tmp++) {
					printf(" %d ",row_indexes[tmp]);
				}
				printf("\n");
					
				if (wresult || wsum || wsumsq) {
				  int wi,ki;
				  double wsofar;

				  for (wi=0;wi<dyv_size(wresult);wi++) {
				    double temp;
				    wsofar = 1.0;

				    for (ki=0;ki<n;ki++) {
				      temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
				      wsofar *= temp;
				      if (wsum) dyv_increment(wsum,wi,temp);
				      if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
				    }
				    dyv_increment(wresult,wi,wsofar);
				  }
				}
			}
			else { /* We got a partial matching n-tuple. Time to get a new point. */
				k++;
				row_indexes[k] = -1;
			}
		}
		else { /* We cannot possibly get a new point without changing some of the
							previous ones */
			k--;	
		}
	}
	
  return result;
}

/* Consider all tuples in which

    ...the first k elements are such that tuple-member j (for j < k) is fixed
       to be the datapoint associated with the rowindex[j]'th row
      of x.

    ...the remaining n-k elements may be chosen according to this recipe:

         tuple member j (for k <= j < n) may be any member it wishes out
         of the set of rows stored inside rowsets[j].

   Then...

     This function returns the number of tuples obeying the above rules
     that "match" the n-point correlation predicate.

     Note that is we have a symmetric (i.e. scalar) matcher 
     (see matcher.h for what this means) this function does not
     double-count (or in general n-factorially-overcount) points:
     it only counts tuples of points in which their virtual kdtree labels
     are in strictly ascending order.


   mapshape *ms : Is merely used for animating
   dym *x: the q'th row of x contains the q'th datapoint
   matcher *ma : Info for the n-point match predicate
   int k: Defines which tuple-members are fixed (those with indices < k)
   row_indexes[] : (array of size n, of which only elements 0 through k-1
                      are relevant on entry) giving the indexes in
                      rowsets for the fixed tuple-members
    rowsets[] : (array of size n). rowsets[q] gives the set of rows in
                  x available to choose between while searching.

     This function ASSUMES that the knodes from which the rowsets were taken
     WERE in increasing order of label. Two adjacent knodes (in rowsets)
     are either identical, or from different knodes in which the 
     earlier-indexed knode has labels strictly lower than the later-indexed
     knode. (Observe a general fact that two leaf knodes must either
     by identical or have an empty set of labels that they share).

   Implementation: Recursive (induction on k).
*/                  
double slow_npt_helper(mapshape *ms,dym **xs,dym **ws, matcher *ma, 
                       bool use_symmetry, int projection, int projmethod,
                       int k,int *row_indexes,ivec **rowsets,dyv *wresult,
                       dyv *wsum, dyv *wsumsq)
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

      cutoff = use_symmetry && 
	       k_and_j_rows_from_same_knode && row_index_j <= i;

      ok = !cutoff && matcher_test_point_pair(ma,projection,projmethod,
                                                xs[k],xs[j],row,row_j,k,j);

      if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) )
      {
        if ( ok )
          ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
                                   (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
        ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
        ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
      }

      if ( !cutoff && ok && Draw_joiners )
        ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
    }

    /* Now, we will do more work ONLY IF there's no constraint violation,
       and no double-counting... */
    if ( !cutoff && ok )
    {
    	if ( k == n-1 )
      {
        result += 1.0; /* base case of recursion */
/*
				printf("Base case of recursion. We found a matching %d-tuple.\n",n);
				printf("The unweighted count is %g.\n",result);
*/			
        if (wresult || wsum || wsumsq)
        {
          int wi,ki;
          double wsofar;
          
          row_indexes[k] = i;  /* just for the loop below */
          for (wi=0;wi<dyv_size(wresult);wi++)
	    	  {
          	double temp;
            wsofar = 1.0;
	    
            for (ki=0;ki<n;ki++)
			      {
              temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
              wsofar *= temp;
              if (wsum) dyv_increment(wsum,wi,temp);
              if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
    	      }
            dyv_increment(wresult,wi,wsofar);

						printf("The %d-th weighted result is %g.\n",wi,wsofar);
	        }
          row_indexes[k] = -77; /* prevents accidental uses */
  	    }
			}
      else
      {
        row_indexes[k] = i; /* recursive case */
        result += slow_npt_helper(ms,xs,ws,ma,use_symmetry,projection,projmethod,
                                  k+1,row_indexes,rowsets,wresult,wsum,wsumsq);
        row_indexes[k] = -77; /* Just to prevent anyone 
                                 accidently using row_indexes[k] again */
      }
    }
  }

  return result;
}

/* Same comments as the helper function above, except "with k == 0": i.e.
   we are searching through all sets of points, finding which ones match. */
double slow_npt(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                bool use_symmetry,int projection,int projmethod,ivec **rowsets,
                dyv *wresult, dyv *wsum,dyv *wsumsq)
{
  int rows[MAX_N];
  int i;
  double result;

  if ( matcher_n(ma) > MAX_N ) my_error("MAX_N too small");

	if (iterative && 0) {
		result = iterative_slow_npt_helper(ms,xs,ws,ma,use_symmetry,projection,projmethod,
		                         rows,rowsets,wresult,wsum,wsumsq);
	}
	else {
	  for ( i = 0 ; i < matcher_n(ma) ; i++ ) rows[i] = -77;
	  result = slow_npt_helper(ms,xs,ws,ma,use_symmetry,projection,projmethod,
		                         0,rows,rowsets,wresult,wsum,wsumsq);
	}
                           
  return result;
}

/* 
 * ANG ~> Rewritting slow_permute_npt_helper in a non-recursive manner.
 *
 * First stage for getting rid of recursion is to embedd everything in a big
 * loop. The following iterative backtracking algorithm applies:
 *
 * while (k >= 0) {
 * 	while ( !ok(k_th_point) && can_try_again(k) ) {
 * 		find_new_point(k);
 * 	}
 * 	if ( ok(k_th_point) ) {
 * 		if (k == (n-1) ) {
 * 			increase_count();
 * 		}
 * 		else {
 * 			k++;
 * 			reset(k_th_point);
 * 		}
 * 	}
 * 	else {
 * 		k--;
 * 	} 	
 * }
 * 
 * For each k we assume that the previous k-1 points have been selected in a
 * non-conflicting manner and that they all satisfy the matcher. Thus, we
 * proceed to select a new point from the k-th leaf in a non-colflicting
 * fashion. We stop trying either when we find a 'good' point or when we have
 * tried all the points.
 * 
 * If we managed to find a 'good' point we check to see if this is in fact an
 * n-tuple. If so we need to update our counts from the unweighted and for the
 * weighted cases. If not we simply proceed to select a new point from the next
 * knode.
 * 
 * If at some point we have run out of possible points in a knode our only
 * solution is to ge to the previous knode and try to find a new point there.
 * 
 * When all the possible n-tuples of points have been tested we cannot find any
 * more new knodes and k will be decreased below the minimum possible level.
 * This is how we check that the matching process is finished.
 */

double iterative_slow_permute_npt_helper(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                               int projection,int projmethod,
                               int *row_indexes,ivec **rowsets,
                               imat *permutation_cache,ivec *permutes_ok,
                               dyv *wresult,dyv *wsum,dyv *wsumsq)
{
  int n = matcher_n(ma);
  int k = 0;
  double result = 0.0;

	total_num_iterative_base_cases += 1.0;
	/* Initializing the first point */
	row_indexes[k] = -1;

	while (k >= 0) {
		int new_point_is_ok = 0;
		int can_try_new_point = 1;
		ivec *rowset_k = rowsets[k];
	  ivec *permutes_ok_copy = mk_copy_ivec(permutes_ok);

		while ( !new_point_is_ok && can_try_new_point ) {
			if ((row_indexes[k] + 1) >= ivec_size(rowset_k)) {
				/* We cannot try a new point. */
				can_try_new_point = 0;
			}
			else {					
				int current_row;
		    int j, i;

				/* Get the  next point and be optimistic about it */
				row_indexes[k] += 1;
				i = row_indexes[k];
				new_point_is_ok = 1;
//				printf("got here 1\n");
				current_row = ivec_ref(rowset_k, i);
//				printf("got here 2\n");
		    copy_ivec(permutes_ok,permutes_ok_copy);				
				
				/* Test it for conflicts against all previously set nodes */
		    for (j = 0; new_point_is_ok && j < k; j++) {
					ivec *rowset_j = rowsets[j];
		      int row_index_j = row_indexes[j];
		      int row_j = ivec_ref(rowset_j,row_index_j);
		      bool k_and_j_rows_from_same_knode = ((long)(rowset_k)) ==	((long)(rowset_j));

					if (row_index_j >= i && k_and_j_rows_from_same_knode) {
						/* The k-th and j-th points are from the same leaf and they are not
						 * in order. This is a conflict! */
						new_point_is_ok = 0;
						/* This might also warrant setting can_try_new_point = 0 but only if
						 * the points are in some sort of order. I'm not sure so I'll forget
						 * about it for the time being. */
						can_try_new_point = 0;
					}
		      else {
						/* The points are either in order of from different knodes. */
					  if (!matcher_permute_test_point_pair(ma,projection,projmethod,
               xs[k],xs[j],current_row,row_j,k,j,permutation_cache,permutes_ok_copy))	{
							/* However, if they don't match this is still a conflict. */
							new_point_is_ok = 0;
						}
						/* Stuff about animations I won't care about now. */
/*
		      if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) ) {
    		    if ( new_point_is_ok ) ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
                               (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
		        ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
		        ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
		      }
		      if ( !cutoff && ok && Draw_joiners ) ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);
*/

					}
				}
			}
		}

		if ( new_point_is_ok ) {
			if (k == (n-1) ) { /* We got a matching n-tuple. Time to count. */
				int tmp_row, tmp_col;
				
	      result += 1.0; 
				printf("Iterative base case method found a match. All results are increased.\n");
				/* Extra debugging info */
				printf("The n-tuple was:  ");
				for (tmp_row = 0; tmp_row < n; tmp_row++) {
					for (tmp_col = 0; tmp_col < dym_cols(xs[k]); tmp_col++) {
						printf("  %f ", dym_ref(xs[tmp_row],row_indexes[tmp_row],tmp_col));
					}
					printf("\n");
				}
	
				if (wresult || wsum || wsumsq) {
				  int wi,ki;
				  double wsofar;

				  for (wi=0;wi<dyv_size(wresult);wi++) {
				    double temp;
				    wsofar = 1.0;

				    for (ki=0;ki<n;ki++) {
				      temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
				      wsofar *= temp;
				      if (wsum) dyv_increment(wsum,wi,temp);
				      if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
				    }
				    dyv_increment(wresult,wi,wsofar);
				  }
				}
			}
			else { /* We got a partial matching n-tuple. Time to get a new point. */
				k++;
				row_indexes[k] = -1;
			}
		}
		else { /* We cannot possibly get a new point without changing some of the
							previous ones */
			k--;	
		}
	}
	
  return result;
}
/* Just like slow_npt_helper except that it tries all permutations of the
   template and always has use_symmetry = TRUE (which is why its dropped
   from the argument list.


Consider all tuples in which

    ...the first k elements are such that tuple-member j (for j < k) is fixed
       to be the datapoint associated with the rowindex[j]'th row
      of x.

    ...the remaining n-k elements may be chosen according to this recipe:

         tuple member j (for k <= j < n) may be any member it wishes out
         of the set of rows stored inside rowsets[j].

   Then...

     This function returns the number of tuples obeying the above rules
     that "match" the n-point correlation predicate.

     Note that is we have a symmetric (i.e. scalar) matcher 
     (see matcher.h for what this means) this function does not
     double-count (or in general n-factorially-overcount) points:
     it only counts tuples of points in which their virtual kdtree labels
     are in strictly ascending order.


   mapshape *ms : Is merely used for animating
   dym **xs: the q'th row of x contains the q'th datapoint
   dym **ws: same as xs, but containing the weights on the data point
   matcher *ma : Info for the n-point match predicate
   int k: Defines which tuple-members are fixed (those with indices < k)
   row_indexes[] : (array of size n, of which only elements 0 through k-1
                      are relevant on entry) giving the indexes in
                      rowsets for the fixed tuple-members
   rowsets[] : (array of size n). rowsets[q] gives the set of rows in
                  x available to choose between while searching.
   permutes_ok : an ivec indicating which permutations of the template are
                 still possible matches given the row_indexes already in
                 place.  If the i'th element is 1, then the i'th permutation
                 is still possible (0 if not).
   wresult: the weighted result (contains sum of products of weights for each
            template match).
   wsum: ... sum of sums of weights ...
   wsumsq: ... sum of sums of squared weights ...
            
     This function ASSUMES that the knodes from which the rowsets were taken
     WERE in increasing order of label. Two adjacent knodes (in rowsets)
     are either identical, or from different knodes in which the 
     earlier-indexed knode has labels strictly lower than the later-indexed
     knode. (Observe a general fact that two leaf knodes must either
     by identical or have an empty set of labels that they share).

   Implementation: Recursive (induction on k).  
*/
double slow_permute_npt_helper(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                               int projection,int projmethod,int k,
                               int *row_indexes,ivec **rowsets,
                               imat *permutation_cache,ivec *permutes_ok,
                               dyv *wresult,dyv *wsum,dyv *wsumsq)
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

      cutoff = (k_and_j_rows_from_same_knode && (row_index_j <= i));

      ok = !cutoff && matcher_permute_test_point_pair(ma,projection,projmethod,
                                                      xs[k],xs[j],row,row_j,k,j,
                                                      permutation_cache,
                                                      permutes_ok_copy);

      if ( !Do_rectangle_animation && (Verbosity >= 1.0 && ms != NULL) ) {
        if ( ok ) ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,
                               (cutoff)?AG_BLUE:(ok)?AG_RED:AG_GREEN);
        ms_point_in_dym_colored(ms,xs[k],row,DOT_MARKTYPE,AG_BLACK);
        ms_point_in_dym_colored(ms,xs[j],row_j,DOT_MARKTYPE,AG_BLACK);
      }

      if ( !cutoff && ok && Draw_joiners ) ms_line_between_dym_rows(ms,xs[k],xs[j],row,row_j,AG_BLUE);

			/* Debugging info */
			/*
			if (cutoff && k==1) {
				printf("Cutoff is true for k=%d and we have: \n",k);
				fprintf_ivec(stdout, "rowsets[k] = ", rowsets[k], "\n");
				fprintf_ivec(stdout, "rowsets[j] = ", rowsets[j], "\n");
			}
			*/

    }

    /* Now, we will do more work ONLY IF there's no constraint violation, and no double-counting... */
		if (cutoff || !ok) {
/*			
 			printf("Something went wrong and there's a constraint violation at k = %d.\n", k);
			fprintf_imat(stdout,"permutation cache ", permutation_cache, "\n"); 
			fprintf_ivec(stdout,"permutes ok ", permutes_ok_copy, "\n");
*/
		}
		else
//    if ( !cutoff && ok )
    {
      if ( k == n-1 ) {
				int tmp;

        result += 1.0; /* base case of recursion */
				/*
				printf("Base case of recursion. All results are increased.\n");
//			Extra debugging info 
				printf("The n-tuple was:  ");
				for (tmp=0; tmp<n-1; tmp++) {
					printf(" %d ",row_indexes[tmp]);
				}
				printf(" %d \n", i);
				*/
			
				if (wresult || wsum || wsumsq) {
				  int wi,ki;
				  double wsofar;

				  row_indexes[k] = i;  /* just for the loop below */
				  for (wi=0;wi<dyv_size(wresult);wi++) {
				    double temp;
				    wsofar = 1.0;

				    for (ki=0;ki<n;ki++) {
				      temp = dym_ref(ws[ki],ivec_ref(rowsets[ki],row_indexes[ki]),wi);
				      wsofar *= temp;
				      if (wsum) dyv_increment(wsum,wi,temp);
				      if (wsumsq) dyv_increment(wsumsq,wi,temp*temp);
				    }
				    dyv_increment(wresult,wi,wsofar);
				  }
				  row_indexes[k] = -77; /* prevents accidental uses */
				}
			}
      else {
        row_indexes[k] = i; /* recursive case */
/*				
				if (k > 0) {
					printf("Recursive base-case will be called for the %d'th node.\n",k+1);
				}
*/
        result += slow_permute_npt_helper(ms,xs,ws,ma,projection,projmethod,
                                          k+1,row_indexes,rowsets,
                                          permutation_cache,permutes_ok_copy,
                                          wresult,wsum,wsumsq);
																					
        row_indexes[k] = -77; /* Just to prevent anyone 
                                 accidently using row_indexes[k] again */
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
double slow_permute_npt(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
                        int projection, int projmethod, ivec **rowsets,
                        imat *permutation_cache,dyv *wresult,
                        dyv *wsum, dyv *wsumsq)
{
  int rows[MAX_N];
  int i;
  double result;
  int num_permutes = 1;
  ivec *permutes_ok;

  if ( matcher_n(ma) > MAX_N ) my_error("MAX_N too small");

  for (i=matcher_n(ma); i>1; i--) num_permutes *= i;
  permutes_ok = mk_constant_ivec(num_permutes,1);


	if ( iterative ) {
	  result = iterative_slow_permute_npt_helper(ms,xs,ws,ma,projection,projmethod,
             rows,rowsets,permutation_cache,permutes_ok,wresult,wsum,wsumsq);
	
	}
	else {
	  for ( i = 0 ; i < matcher_n(ma) ; i++ ) rows[i] = -77;
	  result = slow_permute_npt_helper(ms,xs,ws,ma,projection,projmethod,
                                   0,rows,rowsets,
                                   permutation_cache,permutes_ok,wresult,
                                   wsum,wsumsq);
	}

	/* Debugging info */
//	printf("Base case matching in slow_permute_npt is done with result = %g.\n",result);
//	fprintf_ivec(stdout, "Current possible permutations ", permutes_ok, "\n");

  free_ivec(permutes_ok);
  
  return result;
}
  
/* Returns the combinatoric function "n choose m". Implementation cost
   is O(m). Conceivably could be worth accelrating by means of a lookup
   table. */
double n_choose_m(int n,int m)
{
  double result = 1.0;
  int i;

  if ( m < 0 || m > n ) my_error("n_choose_m");

  if ( m > n/2 ) m = n - m;

  for ( i = 0 ; i < m ; i++ ) result *= (n-i) / (double) (i+1);
  return result;
}

/* Returns 0 (instead of aborting) if m is -ve or > n */
double careful_n_choose_m(int n,int m)
{
  return ( m < 0 ) ? 0 : ( m > n ) ? 0 : n_choose_m(n,m);
}

/* This is sort of like n_choose_m except that "n" is the number of points
   stored in the knode and each m-tuple is weighted by the product of the
   weights on the data points.  Now for the bad news.  It is difficult to
   do this in general.  The method used is to compute the result when the
   same point is allowed to appear multiple times in an m-tuple, then try
   to correct for the overcounting.

   In the case m=2:
   Subtract the sum of squared weights since each x_i^2 will appear in the
   initial result, but should not be in an n choose m result.
   Lastly, divide by m!, the number of times legit tuples get counted.

   In the case m=3:
   Subtract the following quantity from the initial result:
   3 * sumsq * sum - 2 * sumcube
   This is a little more complicated.  sumsq * sum will produce lots of terms
   of the form:  x_i^2 * x_j.  These all represent overcounts.  In fac the
   real overcount is x_i * x_i * x_j + x_i * x_j * x_i + x_j * x_i * x_i.
   Therefore, we want to subtract 3 times that amount.  Except:  that product
   produces a few terms of the type x_i^3.  These are only single overcounts,
   so two of them have to be removed again from the correction.  Hence the
   2 * sumcube.  
   Lastly, divide by m!, the number of times legit tuples get counted.
   Confusing?  Yes, but it works if you write it all out.

   The other cases can be handled by successively more complicated methods
   and with more cached information in the knode.  For now, this function
   just returns -1 and an invalid result if you pass it m>3.  
*/
int weighted_n_choose_m(knode *kn, int m, dyv *result)
{
  switch(m)
  {
    case 0: 
    {
      constant_dyv(result,1.0);
      break;
    }
    case 1:
    {
      copy_dyv(kn->sum_weights,result);
			/*			
	 		printf("m=1 and we just computed a weighted contribution\n");
			fprintf_dyv(stdout,"The result is ",result,"\n");
			*/
    	break;
    }
    case 2:
    {
      dyv_mult(kn->sum_weights,kn->sum_weights,result);
      dyv_subtract(result,kn->sumsq_weights,result);
      dyv_scalar_mult(result,0.5,result);
			/*
  		printf("m=2 and we just computed a weighted contribution\n");
			fprintf_dyv(stdout,"The result is ",result,"\n");
			*/
  	  break;
    }
    case 3:
    {
      dyv *acc = mk_dyv(dyv_size(result));
      dyv_mult(kn->sum_weights,kn->sum_weights,result);
      dyv_mult(kn->sum_weights,result,result);
      dyv_mult(kn->sumsq_weights,kn->sum_weights,acc);
      dyv_scalar_mult(acc,3.0,acc);
      dyv_subtract(result,acc,result);
      copy_dyv(kn->sumcube_weights,acc);
      dyv_scalar_mult(acc,2.0,acc);
      dyv_plus(result,acc,result);
      dyv_scalar_mult(result,1.0/6.0,result);
      free_dyv(acc);
			/*
			printf("m=3 and we just computed a weighted contribution\n");
			fprintf_dyv(stdout,"The result is ",result,"\n");
			*/
      break;
    }
    default:
    {
			printf("n was too big or small. Can't compute weighted contribution.\n");
      return -1;
    }
  }
  return 0;
}

/* Returns TRUE if and only if "b" is a descendant of "a", AND
   "a" owns at least one other datapoint that is not owned by "b".

   Implemented by looking at the lo_index and hi_index labels. */
bool as_indexes_strictly_surround_bs(knode *a,knode *b)
{
  return (a->lo_index < b->lo_index && a->hi_index >= b->hi_index) ||
         (a->lo_index <= b->lo_index && a->hi_index > b->hi_index);
}

/* Read the documentation of ttn (below) first. 

   This function simply takes as input

    (b,n,kns[0],{kns[1] ... ,kns[i], ... kns[n-1]} , i)

   and returns

    ttn(b,n,kns[0],{kns[1] ... ,kns[i]->left, ... kns[n-1]})
     +
    ttn(b,n,kns[0],{kns[1] ... ,kns[i]->right, ... kns[n-1]})
*/
extern double ttn(int b,int n,knode **kns);

double two_ttn(int b,int n,knode **kns,int i)
{

  double result = 0.0;
  knode *kni = kns[i];
  if ( knode_is_leaf(kni) ) my_error("Can't happen");
  kns[i] = kni->left;
  result += ttn(b,n,kns);
  kns[i] = kni->right;
  result += ttn(b,n,kns);
  kns[i] = kni;
  return result;
}

/* Let q == b-n.

   Returns the number of distinct q-tuples of strictly sorted labels such that
   the first label is from knode kns[b], the second from kns[b+1] and the
   q'th from kns[b+q-1] == kns[n-1].
   
   A q-tuple of labels ( lab1 , lab2 , ... lab[q] ) is strictly sorted
   if and only if forall i in {1,2,..q-1} lab_i < lab_[i+1].

   This number is the maximum possible number of tuples chosen from
   this set of knodes that could possibly match a symmetric (scalar) matcher.
   If the knodes are out of order the answer will come back zero.

   Note, this problem is the same as "how many ways could you place four
   X's on the diagram below such that there is one X per line and each X
   is on a dot, and all X's must be to the right of the X above them. I (AWM)
   believe this is hard to do efficiently (O(n)), though dynamic programming
   could do it easily in O(number of dots). The below implementation is faster 
   than O(number of dots), I think. It exploits the kdtree structure
   ingeniously. 

   ......................
   .........................................................
            .............
                                   .........................

   Wondering how the recursion implemented actually works?  Don't waste your 
   time looking here, but you might check out the comments about it in the 
   sum weighted version of this function, mk_weighted_sum_ttn().
*/

double ttn(int b,int n,knode **kns)
{
  knode *bkn = kns[b];
  double result;

  if ( b == n-1 )
    result = (double) bkn->num_points;
  else
  {
    int j;
    bool conflict = FALSE;
    bool simple_product = TRUE;

    result = (double) bkn->num_points;

    for ( j = b+1 ; j < n && !conflict ; j++ )
    {
      knode *knj = kns[j];
      if ( bkn->lo_index >= knj->hi_index )
        conflict = TRUE;
      else if ( kns[j-1]->hi_index > knj->lo_index )
        simple_product = FALSE;
    }

    if ( conflict ) result = 0.0;
    else if ( simple_product )
    {
      for ( j = b+1 ; j < n ; j++ )
        result *= kns[j]->num_points;
    }
    else
    {
      int jdiff = -1; /* undefined... will eventually point to the
                          lowest j > b such that kns[j] is different from
                          bkn */
      
      for ( j = b+1 ; jdiff < 0 && j < n ; j++ )
      {
        knode *knj = kns[j];
        if ( bkn->lo_index != knj->lo_index ||
             bkn->hi_index != knj->hi_index )
          jdiff = j;
      }
      
      if ( jdiff < 0 )
        result = careful_n_choose_m(bkn->num_points,n-b);
      else
      {
        knode *dkn = kns[jdiff];
        if ( dkn->lo_index >= bkn->hi_index )
        {
          result = careful_n_choose_m(bkn->num_points,jdiff-b);
          if ( result > 0.0 )
            result *= ttn(jdiff,n,kns);
        }
        else if ( as_indexes_strictly_surround_bs(bkn,dkn) )
          result = two_ttn(b,n,kns,b);
        else if ( as_indexes_strictly_surround_bs(dkn,bkn) )
          result = two_ttn(b,n,kns,jdiff);
        else
          my_error("osdncocsndsn");
      }
    }
  }
  return result;
}

/* Iterative ttn-like counter.. probably very slow */
double iterative_ttn(int n, knode **kns) {
	double result = 0.0;
	int i;

	for (i=0;i<n;) {
		bool conflict = FALSE;
		bool simple_product = TRUE;
		knode *kni = kns[i];
		int j;
		double tmp_result = (double) kni->num_points;

		/* Test for conflicts and overlapping knodes */
		for (j=i+1;j<n && !conflict; j++) {
	 		if (kni->lo_index >= kns[j]->hi_index) conflict = TRUE;
			else if (kns[j-1]->hi_index > kns[j]->lo_index) simple_product = FALSE;
		}

		if (conflict) { /* bad ordering... not counting anything */
			return 0.0;
		}
    else {
			if (simple_product) { /* non-overlapping knodes... easy to give a result */
				
				for (j=i+1;j<n;j++) { /* compute the contribution of the last knodes */
					tmp_result *= kns[j]->num_points;
				}
				
				/* Combine this with the contribution of previous knodes */
				if (i==0)	result = tmp_result; /* no previous knodes => this is it */
				else result *= tmp_result; 				

				/* Since we accounted for all knodes we're done */
				return result;
			}
			else { /* some knodes overlap or are identical */
				int first_overlap = -1;    
				
				j = i+1;
	      do {
	        if (kni->lo_index != kns[j]->lo_index || 
							kni->hi_index !=	kns[j]->hi_index ) {
						first_overlap = j;
					}
					j++;
				}
				while (j < n && first_overlap < 0);
      
	      if (first_overlap < 0) { /* all remaining knodes are identical */
					tmp_result = careful_n_choose_m(kni->num_points,n-i);

					/* Combine this with the contribution of the previous knodes */
					if (i==0) result = tmp_result; /* no previous knodes => this is it */
					else result *= tmp_result;
					
					/* Since we accounted for all knodes we're done */
					return result;
				}
				
	      else { /* we identified the first overlapping knode */
	        knode *kno = kns[first_overlap];
	        if ( kno->lo_index >= kni->hi_index ) {
						/* Computing the contribution of knodes i -> first_overlap-1 */
						tmp_result = careful_n_choose_m(kni->num_points,first_overlap-i);

						/* Only do more work if it's needed to get the answer */
						if (tmp_result > 0.0) {					
							/* Combine this with the contribution of the previous knodes */
							if (i==0) result = tmp_result;
							else result *= tmp_result;
							
							/* Jump the current knode (i) to the overlapping knode */
							i = first_overlap;
						}
					}
					
	        else { /* one of the overlapping knodes is a parent of the other */
						
						if ( as_indexes_strictly_surround_bs(kni,kno) ) { /* i is bigger */
							knode *tmp_kn = kns[i];
							
							/* computing the count for the left child */
							kns[i] = tmp_kn->left;
							result = iterative_ttn(n,kns);
							/* computing the count for the right child */
							kns[i] = tmp_kn->right;
							result += iterative_ttn(n,kns);
							/* restoring the knode list */
							kns[i] = tmp_kn;

							/* We're done */
							return result;
						}
		        else { /* i is not bigger */
							if ( as_indexes_strictly_surround_bs(kno,kni) ) { /* i is smaller */
								knode *tmp_kn = kns[first_overlap];

								/* computing the count for the left child */
								kns[first_overlap] = tmp_kn->left;
								result = iterative_ttn(n,kns);
								/* copmuting the count for the right child */
								kns[first_overlap] = tmp_kn->right;
								result += iterative_ttn(n,kns);
								/* restoring the knode list */
								kns[first_overlap] = tmp_kn;

								/* We're done */
								return result;
							}
				      else my_error("osdncocsndsn"); /* i is in fact the same as
																								first_overlap => ERROR */
						}
					}
				}
			}
		}
	}	

	abort();
	return result;
}


/* Just used for checking ttn (a slow implementation thereof) */
double vst(imat *constraints,int k,int n,int *vals)
{
  double result = 0;

  if ( k == n )
  {
    int i;
    bool ok = TRUE;
    for ( i = 1 ; ok && i < n ; i++ )
      ok = vals[i-1] < vals[i];
    if ( ok )
      result = 1.0;
  }
  else
  {
    int start = imat_ref(constraints,k,0);
    int i;
    if ( k > 0 ) start = int_max(start,vals[k-1]+1);
    for ( i = start ; i < imat_ref(constraints,k,1) ; i++ )
    {
      vals[k] = i;
      result += vst(constraints,k+1,n,vals);
    }
  }
  return result;
}

void weighted_vst()
{
}

/* Just used for checking ttn (a slow implementation thereof) */
double very_slow_ttn(imat *constraints)
{
  int size = imat_rows(constraints);
  int vals[MAX_N];
  return vst(constraints,0,size,vals);
}


/* A version of total_num_ntuples_symmetric(...) declared below
   that does some very very slow checking to ensure that ttn is
   implemented correctly */
double careful_total_num_ntuples_symmetric(int n,knode **kns)
{
  double result;
  double slowres;
  imat *im = mk_imat(n,2);
  int i;
  for ( i = 0 ; i < n ; i++ )
  {
    imat_set(im,i,0,kns[i]->lo_index);
    imat_set(im,i,1,kns[i]->hi_index);
  }

	result = ttn(0,n,kns);
//  slowres = very_slow_ttn(im);
//
//  if ( -1 < (slowres-result) || (slowres-result) > 1 )
//  {
//	  pimat(im);
//    printf("Counting problem!\nslow = %f and fast = %f\n",slowres, result);
//    my_breakpoint();
//    (void) ttn(0,n,kns);
//    my_error("slow and fast ttn disagree");
//  }

  free_imat(im);

  return result;
}

double fast_total_num_ntuples_symmetric(int n,knode **kns)
{
  double result;

	if (iterative) result = iterative_ttn(n,kns);
	else result = ttn(0,n,kns);

  return result;
}

/* Returns the number of distinct n-tuples of strictly sorted labels such that
   the first label is from knode kns[0], the second from kns[1] and the
   n'th from kns[n-1].
   
   A n-tuple of labels ( lab1 , lab2 , ... lab[n] ) is strictly sorted
   if and only if forall i in {1,2,..n-1} lab_i < lab_[i+1].

   This number is the maximum possible number of tuples chosen from
   this set of knodes that could possibly match a symmetric (scalar) matcher.
   If the knodes are out of order the answer will come back zero.
*/
double total_num_ntuples_symmetric(int n,knode **kns)
{
  double result = fast_total_num_ntuples_symmetric(n,kns);
	
#ifdef NEVER
#ifndef AMFAST
  if ( result < 5000.0 )
    (void) careful_total_num_ntuples_symmetric(n,kns);
#endif
#endif

	return result;
}

/* If we're using a compound matcher, it's easy. It's always
   the product of the number of points in the knodes because all
   orderings (permutations) can be counted. */
double total_num_ntuples_assymmetric(int n,knode **kns)
{
  int i;
  double result = 1.0;
  for ( i = 0 ; i < n ; i++ )
  { 
    knode *kni = kns[i];
    result *= (double) (kni->num_points);
  }
  return result;
}

/* If all points in all the knodes matched what is the maximum possible
   count? */
/* WARNING For this function to work, the lo_index and hi_index
   fields of all the knodes should be set correctly */
double total_num_ntuples(int n,bool use_symmetry,knode **kns)
{
  double result;

  if ( use_symmetry ) {
/*
 * if (kns[n-1]->num_points < 1000) {
			result = careful_total_num_ntuples_symmetric(n,kns);
		}
		else {
*/
			result = total_num_ntuples_symmetric(n,kns);
//		}
//		printf("Using symmetric counter\n");
	}
  else {
    abort(); // gfb
		result = total_num_ntuples_assymmetric(n,kns);
//		printf("Using asymmetric counter\n");
	}

  return result;
}

/* Read the documentation of ttn (below) first. 

   This function simply takes as input

    (b,n,kns[0],{kns[1] ... ,kns[i], ... kns[n-1]} , i)

   and returns

    ttn(b,n,kns[0],{kns[1] ... ,kns[i]->left, ... kns[n-1]})
     +
    ttn(b,n,kns[0],{kns[1] ... ,kns[i]->right, ... kns[n-1]})
*/
dyv *mk_weighted_two_ttn(int b,int n,knode **kns,int i)
{
	/* Just declaring the prototype for mk_weighted_ttn. */
  dyv *mk_weighted_ttn(int b,int n,knode **kns);
  dyv *result,*tmp_result;

  knode *kni = kns[i];

  if ( knode_is_leaf(kni) ) my_error("mk_weighted_two_ttn: Can't happen");
  kns[i] = kni->left;
  result = mk_weighted_ttn(b,n,kns);
  kns[i] = kni->right;
  tmp_result = mk_weighted_ttn(b,n,kns);
  dyv_plus(result,tmp_result,result);
  kns[i] = kni;
  free_dyv(tmp_result);
  return result;
}

/* see comments for ttn.  this is the weighted version */
dyv *mk_weighted_ttn(int b,int n,knode **kns) {
  knode *bkn = kns[b];
  dyv *result = NULL;
  int ret;
  int wsize = dyv_size(bkn->sum_weights);
  
  if (b == n-1) result = mk_copy_dyv(bkn->sum_weights);
  else
  {
    int j;
    bool conflict = FALSE;
    bool simple_product = TRUE;

    for ( j = b+1 ; j < n && !conflict ; j++ )
    {
      knode *knj = kns[j];
      if (bkn->lo_index >= knj->hi_index) conflict = TRUE;
      else if (kns[j-1]->hi_index > knj->lo_index) simple_product = FALSE;
    }

    if (conflict) { /* There is an ordering conflict so we return 0 */
      result = mk_zero_dyv(wsize);
			// printf("Ordering conflict\n");
    }
    else {
			if (simple_product) { /* There is no ordering conflict and no overlap */
	      result = mk_copy_dyv(kns[b]->sum_weights);
  	    for (j=b+1;j<n;j++) {
					dyv_mult(kns[j]->sum_weights,result,result);
				}
				/* We can safely compute the rest of the contribution as 
				 *		result = product (sum(W in knode j)) for all j>=b
				 */
  	  }
	    else { /* There is no conflict but there may be overlaps */
	      bool jdiff = -1; 
				/* undefined... will eventually point to the lowest 
           j > b such that kns[j] is different from bkn */
  	    for ( j = b+1 ; jdiff < 0 && j < n ; j++ ) {
	        knode *knj = kns[j];
	        if (bkn->lo_index != knj->lo_index || bkn->hi_index != knj->hi_index)	{ 
							jdiff = j;
					}
				}

	      if (jdiff < 0) { /* The rest of the knodes are copies of the current knode */
					result = mk_dyv(wsize);
					ret = weighted_n_choose_m(bkn,n-b,result); /* Compute the result in
																												one shot */
					if (ret < 0) { /* We couldn't compute the contribution */
					  /* note to self: instead of my_error, just randomly (or something
        	     smarter) split one of these nodes with two_ttn.  of course you
          	   need to check for leafness in that case */
					  printf("The following error can be fixed in the code with some effort.\n");
						printf("See the function weighted_n_choose_m and add additional	cases for higher n.\n"); 
						printf("Also add the required extra cached info to knodes.\n");
					  my_error("weighted_ttn: code does not support weighted npoint for n>3");
					}
				}
				else { /* There may be a few other copies of the current node */
					knode *dkn = kns[jdiff]; /* Consider the first different knode */
					if ( dkn->lo_index >= bkn->hi_index ) { /* There is no overlap with
																										 the current knode */
					  result = mk_dyv(wsize);
						
					 /* Compute the contribution of	the copies and recurse on the	first
						* different	knode */
					  ret = weighted_n_choose_m(bkn,jdiff-b,result); 
						if (ret < 0) { /* We couldn't compute the contribution */
						  printf("The following error can be fixed in the code with some effort.\n");
							printf("See the function weighted_n_choose_m and add additional	cases for higher n.\n"); 
							printf("Also add the required extra cached info to knodes.\n");
						  my_error("weighted_ttn: code does not support weighted npoint for n>3");
						}
						else { /* We computed a valid contribution */
							/* I don't understand why the next check is useful
						  if ((dyv_max(result) != 0.0) || (dyv_min(result) != 0.0)) {
						    dyv *tmp_result = mk_weighted_ttn(jdiff,n,kns);
						    dyv_mult(result,tmp_result,result);
						    free_dyv(tmp_result);
						  }
							*/
							dyv *tmp_result = mk_weighted_ttn(jdiff,n,kns);
							dyv_mult(result,tmp_result,result);
							free_dyv(tmp_result);
						}
					}
					else { /* The first different knode overlaps the current knode */
						if ( as_indexes_strictly_surround_bs(bkn,dkn) ) {
						  result = mk_weighted_two_ttn(b,n,kns,b); /* Split the current
																													knode and try again */
						}
						else { 
							if ( as_indexes_strictly_surround_bs(dkn,bkn) ) {
							  result = mk_weighted_two_ttn(b,n,kns,jdiff); /* Split the
																																first different
																																knode and try
																																again */
							}
							else { /* The first different knode overlaps the current knode but
											is neither a leaf or a parent of it ??? This shouldn't
											happen... ever. */
						  	my_error("osdncocsndsn");
			      	}
						}
					}
				}
			}
    }
  }
	
  return result;
}

/* If we're using a compound matcher, it's easy. It's always
   the product of the number of points in the knodes because all
   orderings (permutations) can be counted.
   PRE: result is a vector of 1.0s
*/
void weighted_ntuples_asymmetric(int n,knode **kns, dyv *result)
{
  int i;
  for ( i = 0 ; i < n ; i++ ) dyv_mult(kns[i]->sum_weights,result,result);
}

/* Just like total_num_ntuples, except it counts the weighted contribution
   of all possible ntuples -- returns sum of product of points matching the
   template (ie all points below these nodes).
*/
dyv *mk_weighted_total_ntuples(int n,bool use_symmetry,knode **kns)
{
  dyv *result = NULL;
  if (use_symmetry) {
		result = mk_weighted_ttn(0,n,kns);
//		printf("Using the symmetric weighted n-tuple counter\n");
	}
  else  {
    result = mk_constant_dyv(dyv_size(kns[0]->sum_weights),1.0);
    weighted_ntuples_asymmetric(n,kns,result);
//		printf("Using the asymmetric weighted n-tuple counter\n");
  }
  return result;
}

/* There are n knodes which all have non-overlapping indices.  We compute the
   sum (over all possible tuples taking one point from each knode) of the 
   sums of the weights of each point in each tuple.
*/
dyv *mk_weighted_sum_ntuples_asymmetric(int n,knode **kns,bool sq,int *count)
{
  int i,j,prod;
  dyv *result = mk_zero_dyv(dyv_size(kns[0]->sum_weights));
  dyv *temp = mk_zero_dyv(dyv_size(result));

  if (count) *count = 1;
  for (i=0;i<n;i++)
  {
    /* the number of times each element of this knode appears in a matched
       template is the product of the number of points in the other knodes
    */
    for (j=0,prod=1;j<n;j++) if (i!=j) prod *= kns[j]->num_points;
    if (sq) copy_dyv(kns[i]->sumsq_weights,temp);
    else    copy_dyv(kns[i]->sum_weights,temp);
    dyv_scalar_mult(temp,prod,temp);
    dyv_plus(result,temp,result);
    if (count) *count *= kns[i]->num_points;
  }
  free_dyv(temp);
  return result;
}

/* a copy of mk_weighted_two_ttn.  Just like the other versions except it
   calls mk_weighted_sum_ttn instead of mk_..._ttn.  count counts the number
   of pattern matches, much as the original ttn does.
   sq says if we're accumulating sums of squares of weights rather than just
   sums of weights.
*/
dyv *mk_weighted_sum_two_ttn(int b, int n, knode **kns, int i, bool sq,
			     int *count)
{
  int count_left = 0;
  int count_right = 0;
  dyv *mk_weighted_sum_ttn(int b,int n,knode **kns,bool sq,int *count);
  dyv *result,*tmp_result;

  knode *kni = kns[i];

  if ( knode_is_leaf(kni) ) my_error("mk_weighted_two_ttn: Can't happen");
  kns[i] = kni->left;
  result = mk_weighted_sum_ttn(b,n,kns,sq,&count_left);
  kns[i] = kni->right;
  tmp_result = mk_weighted_sum_ttn(b,n,kns,sq,&count_right);
  dyv_plus(result,tmp_result,result);
  *count = count_left + count_right;
  kns[i] = kni;
  free_dyv(tmp_result);
  return result;
}

/* See comments for ttn.  This is the weighted sum version.  There is a lot
   of strange recursion (not commented in the original form of ttn) and a lot
   of algebra to make this work just right.

   The basic idea of the recursion is as follows:
   You have a list of knodes making up a size n match to the pattern.
   You're currently considering only those appearing at or after index b
   (i.e. we're doing a modified kind of tail recursion).
   At any point you may discover one of the following:
   1. the indices in the nodes from b on are out of order and thus the actual
      number of correct matches is zero. (conflict == TRUE)
   2. all of the remaining nodes contain non-overlapping indices, thus the
      number of matches is a product of all the points below.  you can compute
      this in one shot and return immediately. (simple_product == TRUE)
   3. all of the remaining nodes are exactly the same.  you can compute the
      result in one shot and return immediately.
   4. some nodes are identical to the bth node, followed by some other 
      nodes that strictly follow the bth node.  you can immediately compute 
      the contribution of the identical ones and recurse to compute the 
      contribution of the remaining ones (which may also contain some nodes
      that match each other -- but they do not match the bth node).
   5. the next node that does not match the bth node is either an ancestor or
      a descendent of the bth node.  this is far too messy to handle on the 
      spot.  the solution is to call back to the ..._two_... function and ask 
      it to split the larger of the two nodes for another try.

   Now for the algebra:
   First, the basic idea for combining partial results:
   Suppose you are part-way through this recursion.  Some set of "current"
   nodes have been determined to make up count_current possible combinations
   and those count_current possible combinations have a total summed weight
   of weight_current.  With a recursive call, you have determined that the
   rest of the list of knodes make up count_rest possible combinations with
   a total summed weight of weight_rest.  Provided the "current" nodes are
   all strictly before the "rest" nodes (they are -- see notes on recursion)
   then the total count becomes count_total = count_current * count_rest.
   The summed weight between all patterns becomes
   weight_total = count_current * weight_rest + count_rest * weight_current.
   This equation is used to do number 2 in the recursion via a call to a
   function that computes the same in a slightly different manner.
   The equation is also used to combine the two portions in number 4.

   The algebra for step 3 in the recursion (and that part of step 4) is
   another special case.  If you have m copies of the same node with n points
   in the node and weight as the sum_weight of those n points then the
   count_total = n choose m and the 
   weight_total = (n choose m) * sum_weight * (m/n)
   Why is this?  there are n choose m patterns to be matched.  the average
   weight of the data points is sum_weight/n.  there are m points matched for
   each pattern thus we have to multiply by m.  we can use the "average weight"
   of the data points because every data point gets re-used exactly the same
   number of times.
   
   Note that "weight" is actually accumulated in the dyv that is returned
   (each element of the dyv is a different weight).  "count" is returned via
   a pointer in the argument list.  It is not necessary to initialize count
   to zero on the original call to this function.  It will be overwritten
   when the tail recursion hits the end anyway.
*/
dyv *mk_weighted_sum_ttn(int b, int n, knode **kns, bool sq, int *count)
{
  knode *bkn = kns[b];
  dyv *result = NULL;
  int wsize = dyv_size(bkn->sum_weights);
  
  if (b == n-1) 
  {
    if (sq) result = mk_copy_dyv(bkn->sumsq_weights);
    else    result = mk_copy_dyv(bkn->sum_weights);
    *count = bkn->num_points;
  }
  else
  {
    int j;
    bool conflict = FALSE;
    bool simple_product = TRUE;

    for ( j = b+1 ; j < n && !conflict ; j++ )
    {
      knode *knj = kns[j];
      if (bkn->lo_index >= knj->hi_index) conflict = TRUE;
      else if (kns[j-1]->hi_index > knj->lo_index) simple_product = FALSE;
    }

    if (conflict)  /* case 1 */
    {
      result = mk_zero_dyv(wsize);
      *count = 0;
    }
    else if (simple_product)  /* case 2 */
    {
      result = mk_weighted_sum_ntuples_asymmetric(n-b,&(kns[b]),sq,count);
    }
    else
    {
      bool jdiff = -1; /* undefined... will eventually point to the lowest 
                          j > b such that kns[j] is different from bkn */

      for ( j = b+1 ; jdiff < 0 && j < n ; j++ )
      {
        knode *knj = kns[j];
        if (bkn->lo_index != knj->lo_index || bkn->hi_index != knj->hi_index)
	  jdiff = j;
      }

      if (jdiff < 0)  /* case 3 */
      {
	*count = my_irint(careful_n_choose_m(bkn->num_points,n-b));
	if (sq) result = mk_copy_dyv(bkn->sumsq_weights);
	else    result = mk_copy_dyv(bkn->sum_weights);
	dyv_scalar_mult(result,
			(double)(*count)*(double)(n-b)/(double)bkn->num_points,
			result);
      }
      else
      {
	knode *dkn = kns[jdiff];
	if ( dkn->lo_index >= bkn->hi_index )  /* case 4 */
	{
	  double count_cur = careful_n_choose_m(bkn->num_points,jdiff-b);
	  if (sq) result = mk_copy_dyv(bkn->sumsq_weights);
	  else    result = mk_copy_dyv(bkn->sum_weights);
	  dyv_scalar_mult(result,
			  count_cur*(double)(jdiff-b)/(double)bkn->num_points,
			  result);
	  if ((dyv_max(result) != 0.0) || (dyv_min(result) != 0.0))
	  {
	    int count_rest;
	    dyv *result_rest = mk_weighted_sum_ttn(jdiff,n,kns,sq,&count_rest);
	    dyv_scalar_mult(result_rest,count_cur,result_rest);
	    dyv_scalar_mult(result,(double)count_rest,result);
	    dyv_plus(result,result_rest,result);
	    free_dyv(result_rest);
	    *count = my_irint(count_cur) * count_rest;
	  }
	}
	else if ( as_indexes_strictly_surround_bs(bkn,dkn) ) /* case 5 */
	  result = mk_weighted_sum_two_ttn(b,n,kns,b,sq,count);
	else if ( as_indexes_strictly_surround_bs(dkn,bkn) ) /* case 5 */
	  result = mk_weighted_sum_two_ttn(b,n,kns,jdiff,sq,count);
	else
	  my_error("osdncocsndsn");
      }
    }
  }
  return result;
}

/* sq tells if you would like to return sum of squares of weights rather than
   just sum of weights 
*/
dyv *mk_weighted_sum_ntuples(int n, bool use_symmetry, bool sq, knode **kns)
{
  int count = 0;
  dyv *result = NULL;
  if (use_symmetry)  /* carefully count ordered matches */
    result = mk_weighted_sum_ttn(0,n,kns,sq,&count);
  else /* all cross products of points in knode are valid */
    result = mk_weighted_sum_ntuples_asymmetric(n,kns,sq,NULL);
  return result;
}

/* If I give you an estimate of some number V* as Vhat where
   Vhat = (lo + hi)/2, and assuming that lo <= V* <= hi,
   what is the largest fractional error I could make. I.E.,
   how big might

      | Vhat - V* |
      -------------
           V*

   be? */
double compute_errfrac(double lo,double hi)
{
  double maxerr = fabs(hi-lo)/2.0;
  return maxerr / real_max(1.0,lo);
}

dym *other_x(dym **xs,int n)
{
  dym *ox = NULL;
  int i;
  for ( i = 1 ; i < n && ox == NULL ; i++ )
  {
    if ( xs[i] != xs[0] )
      ox = xs[i];
  }
  return ox;
}

void draw_matcher_key(mapshape *ms,matcher *ma)
{
  double x  = 10.0;
  double y = 492.0;
  double msx;
  double msy;
  double msx2;
  double msy2;
  double x2;
  double y2;

  ag_set_pen_color(AG_BLACK);
  ag_print(x,y,matcher_describe_string(ma));

  if ( matcher_is_symmetric(ma) && !ma->between )
  {
    y -= 10.0;
  
    mapshape_agcoords_to_datapoint(ms,x,y,&msx,&msy);
    msx2 = msx + sqrt(ma->dsqd_hi) / dyv_ref(ma->metric,0);
    msy2 = msy;
    mapshape_datapoint_to_agcoords(ms,msx2,msy2,&x2,&y2);
    ag_set_pen_color(AG_BLUE);
    ag_arrow(x,y,x2,y2);
    ag_arrow(x2,y2,x,y);
  }
}

/* This function computes the n-point correlation recursively and quickly.

   It returns an approximate answer to this question:

    How many n-tuples of points (x0,x1,...x_n) can you find such that
    forall i, xi is in kdtree node kns[i] and which match the matching
    predicate? (If the matcher is symmetric (i.e. scalar) returns the
    answer without including redundant permutations).

   The answer will be exact if thresh_ntuples is 0.0

   If thresh_ntuples >= 1.0 then the answer may be approximate, but
   the true value V* is guaranteed to lie in the range 
        ( *lobound - result ) , ( *hibound - result ) 
   upon exit

PARAMETERS:

mapshape *ms:   Only used for animation (contains information on how
                to scale datapoints to draw them on the X-windows display)

dym **xs:       The data. The i'th row of x is the i'th datapoint. Note
                that the "labels" described above are not ordered according to
                the row in "x" but instead by the position in a depth first
                traversal in the kdtree, which may in general be a completely
                different order.

dym **ws;       Weight vectors for each data point.  The i'th row of ws is a
                vector of weights for the i'th row of xs.  Each element of the
                weight vector is treated independently.  It is fine for this
                to be NULL, thus indicating unweighted data.

matcher *ma     The structure representing the matching predicate in 
                question. The "n" of "n-point correlation" is stored in 
                here and is accessed with the matcher_n(ma) fucntion.

use_symmetry    assume a regular polygon when doing matching

use_permutes    check all permutations (but only check point that are ordered
                as in use_symmetry)

knode **kns     An array (of size MAX_N, which is > n) containing the
                current n-tuple of kdtree search nodes (the initial top
                level call has all of these pointing to the root node)

double thresh_ntuples: If you can prove that the current n-tuple of
                       kdtree nodes can produce at most no more than
                       thresh_ntuples matches, then don't bother to
                       go any further, simply return result 0. This 
                       can (if thresh_ntuples > 0) cause an approximate
                       incorrect result, but the bounds (see below) will
                       be valid.

double connolly_thresh: If all knodes have diameter <= connolly_thresh
                        then prune the computation (and count zero for
                        that set of nodes)

                        The bounds will be valid.

double *lobound: 
double *hibound:   On entry, these bound the result of the n-point value
                   over the entire tree, taking into account all knode-tuples
                   processed to date, but not including the current (kns)
                   knode tuples, and not including future tuples, awaiting
                   exploration by recursive calls above us in the stack.

                   On exit, we will have tightened these bounds so that
                   in addition to previously expanded knode-tuples they will
                   also account for the current kns (but will still not have 
                   accounted for those knode-tuples waiting expansion on
                   the calling stack above us).

dyv *wlobound:
dyv *whibound:
dyv *wresult:     Just like lobound, hibound, and the return value, except
                  they are weighted according to the weights given in the
                  data file.  These can all be NULL, meaning don't bother
                  to compute them.
dyv *sum:
dyv *sumsq:       where wresult accumulates products of weights of template
                  matched tuples, these compute sums and sums of squares of
                  weights of template matches

imat *permutation_cache: make one of these using mk_permutation_cache if 
                         you're not symmetric and you are using permutes

int depth:         just for people who want to make nicely formatted debugging
                   prints.          
   
*/
double fast_npt(mapshape *ms,dym **xs,dym **ws,matcher *ma,bool use_symmetry,
                bool use_permutes,knode **kns,
                double thresh_ntuples,double connolly_thresh,
                double *lobound,double *hibound, dyv *wlobound, dyv *whibound,
                dyv *wresult,dyv *wsum,dyv *wsumsq,
                imat *permutation_cache,int depth, 
                int projection, int projmethod)
{
  bool do_weights = (wresult && wlobound && whibound && kns[0]->sum_weights);
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

  dyv *weighted_ntuples = (do_weights) ? 
    mk_weighted_total_ntuples(n,(use_symmetry || use_permutes),kns):
    NULL;

/*
	dyv *weighted_ntuples = (do_weights) ? 
		mk_weighted_total_ntuples(n,use_symmetry,kns) :
			NULL;
*/	
/*	
  bool not_worth_it = (ntuples <= thresh_ntuples);
*/
	bool not_worth_it = FALSE;
	
	/* ANG ~> Maybe take this out for debugging */
  ivec *permute_status = (permutation_cache) ?  mk_constant_ivec(imat_rows(permutation_cache),SUBSUME) : NULL;

  /* a clunky way to figure out which node may be best to split on next. */
	/* ANG ~> Taking this out for debugging would be nice but we segfault :( */
  imat *num_incons = (permutation_cache) ? mk_zero_imat(imat_rows(permutation_cache),n) : NULL;

	/* ANG ~> counting all the possible ntuples to make sure the recursion is ok */
	int tmp_i;
	double tmp_product;
	
	for (tmp_i=0, tmp_product = 1.0;tmp_i<n;tmp_i++) {
		tmp_product *= kns[tmp_i]->num_points;	
	}
//	sum_total_ntuples += tmp_product;
	
	
	/* More debugging */
	if (ntuples <= 0) {
		/* The total number of ntuples is 0... We'll prune right away and count the
		 * number of times this fishy behaviour appears.
		 */
		total_num_missing_ntuples += 1;
		result = 0.0;
		
		sum_total_ntuples += tmp_product;
		return result;
	}
	
  if ( connolly_thresh > 0.0 ) {
    bool all_diameters_below_connolly_thresh = TRUE;
    int i;

    abort();
    
    for ( i = 0 ; all_diameters_below_connolly_thresh && i < n ; i++ ) {
      double diameter = hrect_diameter(kns[i]->hr);
      if ( diameter > connolly_thresh ) all_diameters_below_connolly_thresh = FALSE;
    }

    if ( all_diameters_below_connolly_thresh ) {
      not_worth_it = TRUE;
      printf("We are using the connoly threshold and we will prune because all the diameters are below the specified threshold.\n");
    }
  }

  /* This if block only used for reporting */
//  if (Verbosity >= 0.5)
//  {
//    if ( Next_n < 1000000000 && Num_pt_dists + Num_hr_dists > Next_n )
//    {
//      double ferr = compute_errfrac(*lobound,*hibound);
//      printf("%9d dists. ",Num_pt_dists + Num_hr_dists);
//      printf("lo = %9.5e, hibound = %9.5e ",*lobound,*hibound);
//      if ( ferr < 1e4 ) printf("(ferr %9.4f)",ferr);
//      else              printf("(ferr %9g)",ferr);
//      printf("\n");
//      Next_n *= 2;
//    }
//  }

  for ( i = 0 ; i < n ; i++ ) num_subsumes[i] = 0;

  /* AWM note to self: You can't quit this loop as soon as you
     realize answer_is_zero because you are trying to accumulate the
     high bound. 

     AWM another note to self. I don't believe the above comment any
     more but I haven't had time to think it through or test it out.
  */
  for ( i = 0 ; i < n ; i++ ) {
    knode *kni = kns[i];
    /* Same self note */
    /* Efficiency note: Some of the node-pair-tests will get repeated in
       a recursive call. Cure that problem sometime! 
       Alex (Gray) says he tried this in his implementation but wasn't 
       impressed. */
    for ( j = i+1 ; j < n ; j++ ) { 
      knode *knj = kns[j];
      int status;
      double min_dsqd_between_hrs = FLT_MAX;
      double max_dsqd_between_hrs = 0.0;

			/* Projecting the hrects if we need to */
      if (projection != NONE) {
        abort();
        min_and_max_dsqd_proj(ma->metric,kni->hr,knj->hr,
                              &min_dsqd_between_hrs, &max_dsqd_between_hrs,
                              projection,projmethod);
			}
			
	  	if (use_symmetry || !use_permutes) {
        assert(!233423);
				if (projection == NONE) {
					status = matcher_test_hrect_pair(ma,kni->hr,knj->hr,i,j);
				}
 	      else {
 	        abort();
          status = matcher_test_hrect_pair_proj(ma,kni->hr,knj->hr,i,j,
 	                                              min_dsqd_between_hrs,
   	                                            max_dsqd_between_hrs);
				}
				
        if (status == EXCLUDE) {
					answer_is_zero = TRUE;
					all_subsume = FALSE;
				}
        else {
					if ( status == SUBSUME ) {
						num_subsumes[i] += 1;
	          num_subsumes[j] += 1;
					}
	        else all_subsume = FALSE;
				}
        assert(!233423);
			}
			
      else {
        if (projection == NONE) {
          status = matcher_permute_test_hrect_pair(ma,kni->hr,knj->hr,i,j,
                                                   permute_status,
                                                   num_incons,
                                                   permutation_cache);
				}
        else {
          assert(!323);
          status = matcher_permute_test_hrect_pair_proj(ma,
                                                        kni->hr,knj->hr,i,j,
                                                        permute_status,
                                                        num_incons,
                                                        permutation_cache,
                                                        min_dsqd_between_hrs,
                                                        max_dsqd_between_hrs);
				}
				
        if (status == EXCLUDE) {
					answer_is_zero = TRUE;
					all_subsume = FALSE;
				}
				/* Added by ANG for debugging.  Obsolete now because we no longer use
				 * num_subsumes to determine which knode to split. */
				else {
					if (status == SUBSUME) {
						num_subsumes[i] += 1;
						num_subsumes[j] += 1;
					}
					else all_subsume = FALSE;
				}
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

  /* Okay, now we're in a position to go ahead and prune if we can */
  if ( not_worth_it ) { /* We can do an approximate exclude prune */
    result = 0.0;  /* and don't bother updating wresult */
    total_num_exclusions += 1;
    if ( Verbosity >= 0.0 )
      printf("Number of possible matches zero or below thresh. Prune without decreasing hibound!! \n");
    /* Note we should NOT decrease hi bound in this case */
		
		sum_total_ntuples += tmp_product;
    return result;
  }
  else { 
    if ( answer_is_zero ) { /* We can do an exact exclude prune */
      result = 0.0; /* We will have to decrease the highbound */
      total_num_exclusions += 1;
      /* Decrease the hibound and, eventually, the weighted hibound */
      *hibound -= ntuples;
      if (do_weights) {
        dyv_subtract(whibound,weighted_ntuples,whibound);
      }
			
			sum_total_ntuples += tmp_product;
      return result;
    }
    else {
      if ( all_subsume ) { /* We can do an exact include prune */
        result = ntuples;
        total_num_inclusions += 1;
//		    printf("All tuples match. Prune. Increases lobound by %g\n", ntuples);
        
        /* Increase the lobound and, eventually, the weighted lobound. Update
         * the weighted and unweighted results.
         */
        *lobound += ntuples;
        if (do_weights) {
          dyv_plus(wresult,weighted_ntuples,wresult);
          dyv_plus(wlobound,weighted_ntuples,wlobound);
        }
        if (wsum) {
          dyv *tmp = mk_weighted_sum_ntuples(n, (use_symmetry || use_permutes), FALSE, kns);
          dyv_plus(wsum,tmp,wsum);
          free_dyv(tmp);
        }
        if (wsumsq) {
          dyv *tmp = mk_weighted_sum_ntuples(n, (use_symmetry || use_permutes), TRUE, kns);
          dyv_plus(wsumsq,tmp,wsumsq);
          free_dyv(tmp);
        }
				
				sum_total_ntuples += tmp_product;							
        return result;
      }
      else { /* Can't prune so we need to recurse */
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

        /* Alternate, really stupid method for splitting. Pick the first
         * non-leaf knode. It's as simple as it gets though so it shouldn't
         * cause any problems.
         */
        i = 0;
        while (split_index < 0 && i < n) {
          if (!knode_is_leaf(kns[i])) {
            split_index = i;
            split_index_num_points = kns[i]->num_points;
          }
          i++;
        }

        
        if ( split_index < 0 ) { /* All the nodes are leaves. Do base-case. */
          ivec *rowsets[MAX_N];
          int base_case_error_flag = 0;

          total_num_base_cases += 1;

          for ( i = 0 ; i < n ; i++ ) {
            if ( !knode_is_leaf(kns[i]) ) my_error("no way jose");
            rowsets[i] = kns[i] -> rows;
          }
          
          if (do_weights) wtemp_result = mk_zero_dyv(dyv_size(wresult));
          if (wsum) wtemp_sum = mk_zero_dyv(dyv_size(wsum));
          if (wsumsq) wtemp_sumsq = mk_zero_dyv(dyv_size(wsumsq));

          if (use_symmetry || !use_permutes) {
            result = slow_npt(ms,xs,ws,ma,use_symmetry,projection,projmethod,rowsets,
                              wtemp_result,wtemp_sum,wtemp_sumsq);
            if (result < 0) {
              printf("The result was %g. Error. \n", result);
              exit(0);
            }
          }
          else {
            result = slow_permute_npt(ms,xs,ws,ma,projection,projmethod,rowsets,
                                    permutation_cache,
                                    wtemp_result,wtemp_sum,wtemp_sumsq);
            if (result <0) {
              printf("The result was %g. Error. \n", result);
              exit(0);
            }
          }

          /* Update all hi and lo bounds */
          if (result == 0) {
            base_case_error_flag = 1;
          }
          *lobound += result;
          *hibound -= (ntuples - result);
          if (do_weights) {
            dyv_plus(wlobound,wtemp_result,wlobound);
            dyv_subtract(whibound,weighted_ntuples,whibound);
            dyv_plus(whibound,wtemp_result,whibound);
            dyv_plus(wresult,wtemp_result,wresult);
            free_dyv(wtemp_result);
          }

          if (wsum) {
            dyv_plus(wsum,wtemp_sum,wsum);
            free_dyv(wtemp_sum);
          }
          if (wsumsq) {
            dyv_plus(wsumsq,wtemp_sumsq,wsumsq);
            free_dyv(wtemp_sumsq);
          }
		
        }
        else { /* There's someone waiting to be splitted... */					
          /* We will choose a single knode to split on */
          knode *parent = kns[split_index];
          knode *child1 = parent->right;
          knode *child2 = parent->left;
          double tmp_result;

          recursed = TRUE;
          total_num_recursions += 1;

          if (child1 == NULL || child2 == NULL) {
            printf("Bubu! We are trying to recurse but one of the children is	missing.\n");
            exit(0);
          }
          
          if ( !Do_rectangle_animation && Verbosity >= 1.0 )  {
            printf("About to recurse. lobound=%g, hibound=%g\n",*lobound,*hibound);
            wait_for_key();
          }

          result = 0.0;
          kns[split_index] = child1;

          tmp_result = fast_npt(ms,xs,ws,ma,use_symmetry,use_permutes,kns,
                               thresh_ntuples,connolly_thresh,
                               lobound,hibound,wlobound,whibound,
                               wresult,wsum,wsumsq,permutation_cache,depth+1,
                               projection,projmethod);

          if (tmp_result < 0) {
            printf("The result was %g. Error.\n", tmp_result);
            exit(0);
          }
          else {
            result += tmp_result;
          }

          kns[split_index] = child2;

          tmp_result = fast_npt(ms,xs,ws,ma,use_symmetry,use_permutes,kns,
                               thresh_ntuples,connolly_thresh,
                               lobound,hibound,wlobound,whibound,
                               wresult,wsum,wsumsq,permutation_cache,depth+1,
                               projection,projmethod);
          
          if (tmp_result < 0) {
            printf("The result was %g. Error.\n", tmp_result);
            exit(0);
          }
          else {
            result += tmp_result;
          }


          kns[split_index] = parent;
        }
      }
    }
  }
//	if ( !recursed && Verbosity >= 1.0 ) {
//	  printf("lobound = %g, hibound = %g\n",*lobound,*hibound);
//	  wait_for_key();
//	}
  if (weighted_ntuples) free_dyv(weighted_ntuples);
  if (permute_status) free_ivec(permute_status);
  if (num_incons) free_imat(num_incons);

  return result;
}




void twinpack_initialize_search_state(twinpack *tp,int n,
				      knode **kns,dym **xs,dym **ws)
{
  int i;
  for ( i = 0 ; i < n ; i++ )
  {
    datapack *dp = (tp->format[i] == 'd') ? tp->dp_data : tp->dp_random;
    kns[i] = dp->mr->root;
    xs[i] = dp->x;
    ws[i] = dp->w;
  }
}

bool twinpack_can_use_symmetry(twinpack *tp)
{
  return !tp->d_used || !tp->r_used;
}

bool can_use_symmetry(twinpack *tp,matcher *ma)
{
  return twinpack_can_use_symmetry(tp) && matcher_is_symmetric(ma);
}

/* This function computes the n-point correlation recursively and quickly.

   It returns an approximate answer to this question:

    How many n-tuples of points (x0,x1,...x_n) can you find
    which match the matching
    predicate? (If the matcher is symmetric (i.e. scalar) returns the
    answer without including redundant permutations).

   The answer will be exact if thresh_ntuples is 0.0 and connolly_thresh
   is 0.0
   
   If thresh_ntuples >= 1.0 then the answer may be approximate, but
   the true value V* is guaranteed to lie in the range 
         *lobound  , *hibound 
   upon exit

PARAMETERS:

mapshape *ms:   Only used for animation (contains information on how
                to scale datapoints to draw them on the X-windows display)

dym **xs:         The data. The i'th row of x is the i'th datapoint. Note
                that the "labels" described above are not ordered according to
                the row in "x" but instead by the position in a depth first
                traversal in the kdtree, which may in general be a completely
                different order.

matcher *ma     The structure representing the matching predicate in 
                question. The "n" of "n-point correlation" is stored in 
                here and is accessed with the matcher_n(ma) fucntion.


double thresh_ntuples: If you can prove that the current n-tuple of
                       kdtree nodes can produce at most no more than
                       thresh_ntuples matches, then don't bother to
                       go any further, simply return result 0. This 
                       can (if thresh_ntuples > 0) cause an approximate
                       incorrect result, but the bounds (see below) will
                       be valid.

double connolly_thresh: If all knodes have diameter <= connolly_thresh
                        then prune the computation (and count zero for
                        that set of nodes)

                        The bounds will be valid.

mrkd *mr: An mrkd (see mrkd.h) built from the dataset x (using the
same metric that is stored in matcher, and in which, when built,
the mrpars with which it was build specified has_points as TRUE. By
the way we recommend setting has_xxts and has_sums to FALSE because
they are not needed by npt, and take up a lot of memory)

double *lobound: 
double *hibound:   The entry values are ignored.

                   These are guaranteed to bound the true value of the
                   n-point function.
   
*/

void mrkd_fast_npt_with_thresh_ntuples(twinpack *tp,matcher *ma,
                                       matcher *ma2,
                                       double thresh_ntuples,
                                       double connolly_thresh,
                                       double *lobound,double *hibound,
                                       dyv *wlobound, dyv *whibound,
                                       dyv *wresult, dyv *wsum, dyv *wsumsq)
//void mrkd_fast_npt_with_thresh_ntuples(twinpack *tp,matcher *ma,
//                                       double thresh_ntuples,
//                                       double connolly_thresh,
//                                       double *lobound,double *hibound,
//                                       dyv *wlobound, dyv *whibound,
//                                       dyv *wresult, dyv *wsum, dyv *wsumsq)
{
  knode *kns[MAX_N];
  dym *xs[MAX_N];
  dym *ws[MAX_N];
  double result;
  bool use_symmetry = can_use_symmetry(tp,ma);
  imat *permutation_cache = NULL;

	if (use_symmetry) printf("We are using a symmetric matcher.\n");
	else printf("We are using a compound matcher\n");
	
  twinpack_initialize_search_state(tp,matcher_n(ma),kns,xs,ws);

	if (tp->use_permute || use_symmetry ) {
		printf("We are effectively using a symmetric matcher.\n");
		/*
		tp->use_permute = !tp->use_permute;
		printf("Don't really wanna do this so we are changning the value of	'use_permute' to %d.\n", tp->use_permute);
		*/
	}
	else printf("We are effectively using a compound matcher\n");

  *lobound = 0.0;
  *hibound=total_num_ntuples(matcher_n(ma),use_symmetry||tp->use_permute,kns);
	
	printf("Initial unweighted lo and hi bounds are: %g and %g\n",*lobound,*hibound);
  //*hibound=total_num_ntuples_assymmetric(matcher_n(ma),kns);
  Next_n = 10;

  if (wlobound) zero_dyv(wlobound);
  if (wresult) zero_dyv(wresult);
  if (wsum) zero_dyv(wsum);
  if (wsumsq) zero_dyv(wsumsq);
  if (whibound) {
    dyv *temp_whb = mk_weighted_total_ntuples(matcher_n(ma),
					      use_symmetry||tp->use_permute,
					      kns);
//	  dyv *temp_whb = mk_weighted_total_ntuples(matcher_n(ma),use_symmetry,kns);
		printf("Computing initial weighted hibound\n");
    copy_dyv(temp_whb,whibound);
    free_dyv(temp_whb);
  }
	if (wlobound) fprintf_dyv(stdout,"Initial weighted lower bound",wlobound,"\n");
  if (whibound) fprintf_dyv(stdout,"Initial weighted upper bound",whibound,"\n");

	
  Mr_root = tp->dp_data->mr; /* Only used for drawing lettered datapoints */

  if ( Do_rectangle_animation )
  {
    ag_on("");
    Old_active = FALSE;
  }

  if (!use_symmetry && tp->use_permute)
    permutation_cache = mk_permutation_cache(matcher_n(ma));

  if (Use_Npt2) {
    imat *known_ndpairs = mk_constant_imat(matcher_n(ma),matcher_n(ma),UNKNOWN);
    dym *known_dists = NULL; 
    int i, starts[MAX_N]; ivec *data_map=NULL, *random_map=NULL, *maps[MAX_N];

    if (tp->d_used) {
      int c = tp->dp_data->mr->root->lo_index;
      data_map = mk_ivec(tp->dp_data->mr->root->num_points);
      create_virtual_index_to_dym_row_map(tp->dp_data->mr->root,data_map,&c,c);
    }
    if (tp->r_used) {
      int c = tp->dp_random->mr->root->lo_index;
      random_map = mk_ivec(tp->dp_random->mr->root->num_points);
      create_virtual_index_to_dym_row_map(tp->dp_random->mr->root,random_map,
                                          &c,c);
    }
    for ( i = 0 ; i < matcher_n(ma) ; i++ ) {
      maps[i] = (tp->format[i] == 'd') ? data_map : random_map;
      starts[i] = (tp->format[i] == 'd') ? tp->dp_data->mr->root->lo_index :
                                           tp->dp_random->mr->root->lo_index;
    }
    //dym *known_dists = twinpack_has_random(tp) ? NULL : 
    //  mk_constant_dym(tp->dp_data->x->rows,tp->dp_data->x->rows,UNKNOWN);
    // HERE: put in other cases for formats with random data.
    printf("USING EXPERIMENTAL NPT2 - warning: not supported\n");
    if (Use_MC) printf("USING MONTE CARLO.\n");
    result = fast_npt2(tp->dp_data->ms,xs,ws,ma,use_symmetry,tp->use_permute,
                       kns,thresh_ntuples,connolly_thresh,
                       lobound,hibound,wlobound,whibound,
                       wresult,wsum,wsumsq,permutation_cache,0,
                       known_ndpairs,known_dists,maps,starts);
    //printf("num_not_worth_it_prunes = %g\n",num_not_worth_it_prunes);

    free_imat(known_ndpairs); 
    if (known_dists != NULL) free_dym(known_dists);
    if (tp->d_used) free_ivec(data_map); 
    if (tp->r_used) free_ivec(random_map);
  }
  else if (Use_Npt3) {
    //fheap *fh = fh_makeheap(); fh_setcmp(fh, compare_nodesets);
    sheap *sh = mk_sheap(); 
    int i, starts[MAX_N]; ivec *data_map=NULL, *random_map=NULL, *maps[MAX_N];

    total_ntuples = *hibound;
    enqueue_nodeset(sh,kns,matcher_n(ma),total_ntuples);
    if (tp->d_used) {
      int c = tp->dp_data->mr->root->lo_index;
      data_map = mk_ivec(tp->dp_data->mr->root->num_points);
      create_virtual_index_to_dym_row_map(tp->dp_data->mr->root,data_map,&c,c);
    }
    if (tp->r_used) {
      int c = tp->dp_random->mr->root->lo_index;
      random_map = mk_ivec(tp->dp_random->mr->root->num_points);
      create_virtual_index_to_dym_row_map(tp->dp_random->mr->root,random_map,
                                          &c,c);
    }
    for ( i = 0 ; i < matcher_n(ma) ; i++ ) {
      maps[i] = (tp->format[i] == 'd') ? data_map : random_map;
      starts[i] = (tp->format[i] == 'd') ? tp->dp_data->mr->root->lo_index :
                                           tp->dp_random->mr->root->lo_index;
    }

    //printf("USING EXPERIMENTAL NPT3.\n");
    if (Use_MC) printf("USING MONTE CARLO.\n");
    result = fast_npt3(sh,tp->dp_data->ms,xs,ws,ma,use_symmetry,tp->use_permute,
                       thresh_ntuples,connolly_thresh,
                       lobound,hibound,wlobound,whibound,
                       wresult,wsum,wsumsq,permutation_cache,maps,starts);
    //printf("Heap size at end: %d\n", fheap_size(fh)); fh_deleteheap(fh); 
    printf("Heap size at end: %d\n", sheap_size(sh)); 

    free_sheap(sh); 
    if (tp->d_used) free_ivec(data_map); 
    if (tp->r_used) free_ivec(random_map);
  }
  else if (Projection == BOTH) {
    printf("USING PROJECTED NPT.\n");
    result = fast_npt_proj(tp->dp_data->ms,xs,ws,ma,ma2,
                           use_symmetry,tp->use_permute,
                           kns,thresh_ntuples,connolly_thresh,
                           lobound,hibound,wlobound,whibound,
                           wresult,wsum,wsumsq,permutation_cache,0,
                           Projection,Projmethod);
  }
  else 
    result = fast_npt(tp->dp_data->ms,xs,ws,ma,use_symmetry,tp->use_permute,
                      kns,thresh_ntuples,connolly_thresh,
                      lobound,hibound,wlobound,whibound,
                      wresult,wsum,wsumsq,permutation_cache,0,
                      Projection,Projmethod);

  printf("Final lobound = %g\n",*lobound);
  printf("Final hibound = %g\n",*hibound);

  if ( Do_rectangle_animation ) ag_off();

  if ( *hibound < *lobound ) 
    printf("Warning: hibound ended up less than  lobound\n");
  *hibound = real_max(*lobound,*hibound);

  if (Verbosity >= 0.5)
  {
    printf("Final lobound = %g\n",*lobound);
    printf("Final hibound = %g\n",*hibound);

    if (wlobound) fprintf_dyv(stdout,"Weighted lower bound",wlobound,"\n");
    if (whibound) fprintf_dyv(stdout,"Weighted upper bound",whibound,"\n");
  }

  if (permutation_cache) free_imat(permutation_cache);
}

/* WARNING For this function to work, the lo_index and hi_index
   fields of all the knodes should be set correctly */
double compute_total_ntuples(twinpack *tp,matcher *ma)
{
  knode *kns[MAX_N];
  dym *xs[MAX_N];
  dym *ws[MAX_N];
  twinpack_initialize_search_state(tp,matcher_n(ma),kns,xs,ws);
  return total_num_ntuples(matcher_n(ma),
			   can_use_symmetry(tp,ma)||tp->use_permute,kns);
}

/* WARNING For this function to work, the lo_index and hi_index
   fields of all the knodes should be set correctly */
double total_2pt_tuples(twinpack *tp)
{
  matcher *ma = mk_symmetric_simple_matcher(twinpack_n(tp),
					    twinpack_metric(tp),
					    9e19);
  double result = compute_total_ntuples(tp,ma);
  free_matcher(ma);
  return result;
}

/* Exactly the same as mrkd_fast_npt_with_thresh_ntuples except instead
   of calling with thresh_ntuples, simply call with thresh_errfrac
   set to the largest acceptable fractional error.

   If the true correct npt count is V*, this function promises to
   return Vhat such that

       | V* - Vhat | < thresh_errfrac * V*
*/
void mrkd_fast_npt_autofind(twinpack *tp,matcher *ma,matcher *ma2,
                            double thresh_errfrac,
                            double *lobound,double *hibound,
                            dyv *wlobound, dyv *whibound, dyv *wresult,
                            dyv *wsum, dyv *wsumsq)
//void mrkd_fast_npt_autofind(twinpack *tp,matcher *ma,
//                            double thresh_errfrac,
//                            double *lobound,double *hibound,
//                            dyv *wlobound, dyv *whibound, dyv *wresult,
//                            dyv *wsum, dyv *wsumsq)
{
  double thresh_ntuples = compute_total_ntuples(tp,ma) / 5.0;
  bool finished = FALSE;

  while ( !finished )
  {
    double this_errfrac;
    printf("*** WILL RUN WITH THRESH_NTUPLES = %g\n",thresh_ntuples);
    mrkd_fast_npt_with_thresh_ntuples(tp,ma,ma2,thresh_ntuples,
                                      0.0,lobound,hibound,
                                      wlobound,whibound,wresult,wsum,wsumsq);
    //mrkd_fast_npt_with_thresh_ntuples(tp,ma,thresh_ntuples,
    //                                  0.0,lobound,hibound,
    //                                  wlobound,whibound,wresult,wsum,wsumsq);
    this_errfrac = compute_errfrac(*lobound,*hibound);
    finished = thresh_ntuples <= 1.0 ||
               this_errfrac < thresh_errfrac || this_errfrac < 1e-5;
    printf("*** lo = %g hi = %g\n",*lobound,*hibound);
    printf("*** THRESH_NTUPLES %g GIVES ERRFRAC OF %g (target %g)\n",
	   thresh_ntuples,this_errfrac,thresh_errfrac);
    if ( !finished )
      thresh_ntuples /= 5.0;
  }
}

//nout *mk_run_npt_from_twinpack(twinpack *tp,params *ps,matcher *ma)
nout *mk_run_npt_from_twinpack(twinpack *tp,params *ps,matcher *ma,matcher *ma2)
{
  extern bool Do_rectangle_animation;
  //int start_secs = global_time();  /* AG */
  nout *no = AM_MALLOC(nout);
  double lo,hi;
  FILE *s;
  char *save_name = (ps->autofind) ? "autofind.txt" : "results.txt";
  dyv *wlobound = NULL;
  dyv *whibound = NULL;
  dyv *wresult = NULL;
  dyv *wsum = NULL;
  dyv *wsumsq = NULL;

  Start_secs = global_time(); /* AG */

  Verbosity = ps -> verbosity;
//	Do_rectangle_animation = TRUE;
  Do_rectangle_animation = ps -> rdraw;
	

  if ( ps -> n > MAX_N ) my_error("MAX_N too small");

  Num_pt_dists = 0;
  Num_hr_dists = 0;
  
  if (tp->dp_data && tp->dp_data->w)
  {
    wlobound = mk_dyv(dym_cols(tp->dp_data->w));
    whibound = mk_dyv(dym_cols(tp->dp_data->w));
    wresult = mk_dyv(dym_cols(tp->dp_data->w));
  }
  else if (tp->dp_random && tp->dp_random->w)
  {
    wlobound = mk_dyv(dym_cols(tp->dp_random->w));
    whibound = mk_dyv(dym_cols(tp->dp_random->w));
    wresult = mk_dyv(dym_cols(tp->dp_random->w));
  }

  if (wresult && ps->do_wsums) wsum = mk_dyv(dyv_size(wresult));
  if (wresult && ps->do_wsumsqs) wsumsq = mk_dyv(dyv_size(wresult));

  if (!ps->autofind) 
  {
    mrkd_fast_npt_with_thresh_ntuples(tp,ma,ma2,ps->thresh_ntuples,
	                  ps->connolly_thresh,&lo,&hi,
	                  wlobound,whibound,wresult,wsum,wsumsq);
    //mrkd_fast_npt_with_thresh_ntuples(tp,ma,ps->thresh_ntuples,
	//                  ps->connolly_thresh,&lo,&hi,
	//                  wlobound,whibound,wresult,wsum,wsumsq);
  } 
  else 
  {
    mrkd_fast_npt_autofind(tp,ma,ma2,ps->errfrac,&lo,&hi,
                           wlobound,whibound,wresult,wsum,wsumsq);
    //mrkd_fast_npt_autofind(tp,ma,ps->errfrac,&lo,&hi,
    //                       wlobound,whibound,wresult,wsum,wsumsq);
  }

  no -> count = (lo + hi) / 2.0;
  no -> lo = lo;
  no -> hi = hi;
  if (wresult)
  {
    copy_dyv(wlobound,wresult);
    dyv_plus(wresult,whibound,wresult);
    dyv_scalar_mult(wresult,0.5,wresult);
  }
  no->wlobound = wlobound;
  no->whibound = whibound;
  no->wresult = wresult;
  no->wsum = wsum;
  no->wsumsq = wsumsq;
  no -> ferr = compute_errfrac(lo,hi);
  no -> secs = global_time() - Start_secs;

  if (Verbosity >= 0.0)
  {
    printf("Num_pt_dists = %d\n", Num_pt_dists);
    printf("Num_hr_dists = %d\n", Num_hr_dists);
    printf("Num seconds =  %d\n", global_time() - Start_secs);

    printf("%d-point function using %s is between %g and %g\n",
	   ps->n, matcher_describe_string(ma), lo, hi);

    if (wresult) fprintf_dyv(stdout,"Weighted result",wresult,"\n");
    if (wlobound) fprintf_dyv(stdout,"Weighted lower bound",wlobound,"\n");
    if (whibound) fprintf_dyv(stdout,"Weighted upper bound",whibound,"\n");
    if (wsum) fprintf_dyv(stdout,"Weighted sum result",wsum,"\n");
    if (wsumsq) fprintf_dyv(stdout,"Weighted sum square result",wsumsq,"\n");
  }

  s = fopen(save_name,"a");
  fprintf(s,"dfile=%s rfile=%s format=%s dnum_pts=%d n=%d ma=%s ",
	  tp->dp_data->filename,
	  (tp->dp_random==NULL) ? "none" : tp->dp_random->filename,
	  tp->format,
	  dym_rows(tp->dp_data->x),ps->n,matcher_describe_string(ma));

  if (ps->autofind) 
    fprintf(s,"errfrac=%g ",ps->errfrac);
  else
  {
    fprintf(s,"thresh_ntuples=%g ",ps->thresh_ntuples);
    fprintf(s,"connolly_thresh=%g ",ps->connolly_thresh);
  }

  fprintf(s,"lo=%g hi=%g pt_dists=%d hr_dists=%d secs=%d\n",
	  no->lo,no->hi,Num_pt_dists,Num_hr_dists,no->secs);

  if (wresult) fprintf_oneline_dyv(s,"Weighted result",wresult,"\n");
  if (wlobound) fprintf_oneline_dyv(s,"Weighted lower bound",wlobound,"\n");
  if (whibound) fprintf_oneline_dyv(s,"Weighted upper bound",whibound,"\n");
  if (wsum) fprintf_oneline_dyv(s,"Weighted sum result",wsum,"\n");
  if (wsumsq) fprintf_oneline_dyv(s,"Weighted sum square result",wsumsq,"\n");

  fclose(s);

  return no;
}



nout *mk_2pt_nout(twinpack *tp,double thresh_ntuples,double connolly_thresh,
		  double lo_radius,double hi_radius)
{
  matcher *ma = mk_symmetric_between_matcher(2,
					     twinpack_metric(tp),
					     lo_radius,hi_radius);
  params *ps = mk_default_params();
  nout *no;

  ps -> n = 2;
  ps -> thresh_ntuples = thresh_ntuples;
  ps -> connolly_thresh = connolly_thresh;
  ps -> autofind = FALSE;
  ps -> errfrac = -77.77e77;
  ps -> verbosity = Verbosity;
  ps -> rdraw = FALSE;

  no = mk_run_npt_from_twinpack(tp,ps,ma,NULL);
  //no = mk_run_npt_from_twinpack(tp,ps,ma);

  free_params(ps);
  free_matcher(ma);

  return no;
}

nouts *mk_multi_run_npt(twinpack *tp,params *ps,
			string_array *matcher_strings)
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

bool print_mrkd_level(knode *kn, int target_level, int this_level)
{
  if (target_level == this_level)
  {
    fprintf_oneline_dyv(stdout,"",kn->hr->lo,"");
    fprintf_oneline_dyv(stdout,"",kn->hr->hi,"\n");
    return TRUE;
  }
  else
  {
    bool l=(kn->left&&print_mrkd_level(kn->left,target_level,this_level+1));
    bool r=(kn->right&&print_mrkd_level(kn->right,target_level,this_level+1));
    return (l || r);
  }
}

void print_mrkd_bounds(mrkd *mr)
{
  int target_level = 0;
  bool done = FALSE;

  while (!done)
  {
    printf("Level %d\n",target_level);
    done = !print_mrkd_level(mr->root,target_level++,0);
  }
}

void npt_main(int argc,char *argv[])
{
  params *ps = mk_params_from_args(argc,argv);
  twinpack *tp = mk_twinpack_from_args(ps,argc,argv);
  bool show_bounds = bool_from_args("show_bounds",argc,argv,FALSE);
  bool multi;
  char *projection_str = string_from_args("projection",argc,argv,"none");
  char *projmethod_str = string_from_args("projmethod",argc,argv,"wake");

	printf("Format = %s.\n",tp->format);
  printf("Projection direction = ");
  if ( eq_string(projection_str,"none") ) {
    Projection = NONE; printf("none\n");
  }
  else if ( eq_string(projection_str,"para") ) {
    Projection = PARA; printf("parallel\n");
  }
  else if ( eq_string(projection_str,"perp") ) {
    Projection = PERP; printf("perpendicular\n");
  }
  else if ( eq_string(projection_str,"both") ) {
    Projection = BOTH; printf("both\n");
  }
  else {
    printf("?\n"); my_error("Bad projection direction option.");
  }

  printf("Projection method = ");
  if ( eq_string(projmethod_str,"wake") ) {
    Projmethod = WAKE; printf("Wake's method\n");
  }
  else if ( eq_string(projmethod_str,"dalton") ) {
    Projmethod = DALTON; printf("Dalton's method\n");
  }
  else if ( eq_string(projmethod_str,"fisher") ) {
    Projmethod = FISHER; printf("Fisher's method\n");
  }
  else {
    printf("?\n"); my_error("Bad projection method option.");
  }

  if (show_bounds)
  {
    if (tp->dp_data)
    {
      printf("Bounds for primary data set MRKD tree:\n");
      print_mrkd_bounds(tp->dp_data->mr);
    }
    if (tp->dp_random)
    {
      printf("Bounds for random data set MRKD tree:\n");
      print_mrkd_bounds(tp->dp_random->mr);
    }
  }

  multi = (index_of_arg("binfile",argc,argv) > 0);
  if ( multi )
  {
    char *binfile = string_from_args("binfile",argc,argv,"binfile.txt");
    string_array *matcher_strings = mk_string_array_from_file_tokens(binfile);
    nouts *ns = mk_multi_run_npt(tp,ps,matcher_strings);
    explain_twinpack(tp);
    explain_params(ps);
    explain_nouts(matcher_strings,ns);
    free_nouts(ns);
    free_string_array(matcher_strings);
  }
  else
  {
    matcher *ma = mk_matcher_from_args(ps->n,tp->dp_data->mps->metric,
                                       argc,argv);
    matcher *ma2 = mk_matcher2_from_args(ps->n,tp->dp_data->mps->metric,
                                         argc,argv);
    nout *no;

    Draw_joiners = bool_from_args("draw_joiners",argc,argv,FALSE);

    if ( Draw_joiners )
    {
      ag_on("dataset.ps");
      ms_dym_colored(tp->dp_data->ms,tp->dp_data->x,
		     (dym_rows(tp->dp_data->x)>2000)?
		     PIXEL_MARKTYPE:DOT_MARKTYPE,
		     AG_BLACK);
      ag_off();
      printf("Saved basic data in dataset.ps\n");
      wait_for_key();
      ag_on("joins.ps");
    }

    no = mk_run_npt_from_twinpack(tp,ps,ma,ma2);
    //no = mk_run_npt_from_twinpack(tp,ps,ma);

    if ( Draw_joiners )
    {
      ms_dym_colored(tp->dp_data->ms,tp->dp_data->x,
		     (dym_rows(tp->dp_data->x)>2000)?
		     PIXEL_MARKTYPE:DOT_MARKTYPE,
		     AG_BLACK);
      ag_off();
      wait_for_key();
      printf("Saved joined data in joins.ps\n");
    }

    if (Verbosity >= 0.5)
    {
      explain_twinpack(tp);
      explain_params(ps);
      explain_matcher(ma);
      explain_matcher(ma2);
		}
		
		/* Debugging info */
		theoretical_total_ntuples = tp->dp_data->mr->root->num_points;
		if (tp->dp_random) {
			theoretical_total_ntuples *= tp->dp_random->mr->root->num_points;
			theoretical_total_ntuples *= tp->dp_random->mr->root->num_points;	
		}
		else {
			theoretical_total_ntuples *= tp->dp_data->mr->root->num_points;
			theoretical_total_ntuples *= tp->dp_data->mr->root->num_points;
		}
			
		printf("total inclusions = %f\n", total_num_inclusions);
		printf("total exclusions = %f\n", total_num_exclusions);
		printf("total base cases = %f\n", total_num_base_cases);
		printf("total recursions = %f\n", total_num_recursions);
		printf("total number of 'missing' n-tuples = %f\n",	total_num_missing_ntuples);
		printf("total iterative base cases = %f\n", total_num_iterative_base_cases);
		printf("\ntotal points in the first dataset =	%d\n", 
				tp->dp_data->mr->root->num_points);
		if (tp->dp_random) {
		printf("total points in the second dataset = %d\n", 
				tp->dp_random->mr->root->num_points);
		}
		printf("total number of ntuples seen in the recursion =  %f\n", 
				sum_total_ntuples);
		printf("total theoretical number of ntuples (n1*n2*n2) = %f\n",
				theoretical_total_ntuples);
		printf("the extra number of ntuples seen in the recursion = %f\n",
				sum_total_ntuples - theoretical_total_ntuples);
		
    explain_nout(no);
    free_nout(no);
    free_matcher(ma);
    free_matcher(ma2);
  }
  free_twinpack(tp);
  free_params(ps);
}

#include "npt2.c"
#include "npt3.c"
#include "projnpt.c"
