/*
   File:        npt.h
   Author:      Andrew W. Moore
   Created:     Wed May 17 12:25:12 EDT 2000
   Description: Header for Fast N-point computation

   Copyright 2000, the Auton Lab
*/


#ifndef NPT_H
#define NPT_H

#include "napi.h"

#define MAX_N 50

#ifndef FLT_MIN 
#define FLT_MIN 1.17549e-38
#endif
#ifndef FLT_MAX
#define FLT_MAX 3.40282e+38
#endif

extern double total_num_inclusions;
extern double total_num_exclusions;
extern double total_num_recursions;
extern double total_num_matches;
extern double total_num_mismatches;
extern double max_num_matches;
extern double total_num_base_cases;
extern double total_num_iterative_base_cases;
extern double total_num_missing_ntuples;
extern double sum_total_ntuples;
extern double theoretical_total_ntuples;

extern int iterative;

void rectangle_animate(mapshape *ms,knode **kns,int n,int agcol);
dym *other_x(dym **xs,int n);
void draw_matcher_key(mapshape *ms,matcher *ma);
void draw_lettered_mrkd_points(mapshape *ms,mrkd *mr,dym *x);
void draw_lettered_knode_points(mapshape *ms,knode *kn,dym *x,char *s);
void special_weighted_symmetric_debugging_test(dym **xs,dym **ws,matcher *ma,
											   knode **kns,dyv *wresult);
dyv *mk_weighted_sum_ntuples(int n, bool use_symmetry, bool sq, knode **kns);

/* We associate each datapoint with a "label". Datapoints are labeled
   according to the order that would be visited by a depth-first recursive
   traversal of the tree (the traversal always goes down left branches
   before right branches).

   This function fills up the "lo_index" and "hi_index" fields of all
   knodes in the tree so that "lo_index" = the lowest label of any point
   descended from this node and "hi_index" is one larger than the
   highest label of any point descended from this node. 

   Note this invariant: kn->num_points == kn->hi_index - kn->lo_index */ 
void mrkd_set_search_indexes(mrkd *mr);

double slow_npt(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                bool use_symmetry,int projection,int projmethod,ivec **rowsets,
                dyv *wresult, dyv *wsum,dyv *wsumsq);
double slow_permute_npt(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
                        int projection, int projmethod,ivec **rowsets,
                        imat *permutation_cache,dyv *wresult,
                        dyv *wsum, dyv *wsumsq);

/* Returns 0 (instead of aborting) if m is -ve or > n */
double careful_n_choose_m(int n,int m);

/* Returns TRUE if and only if "b" is a descendant of "a", AND
   "a" owns at least one other datapoint that is not owned by "b".

   Implemented by looking at the lo_index and hi_index labels. */
bool as_indexes_strictly_surround_bs(knode *a,knode *b);

/* Read the documentation of ttn (below) first. 

   This function simply takes as input

    (b,n,kns[0],{kns[1] ... ,kns[i], ... kns[n-1]} , i)

   and returns

    ttn(b,n,kns[0],{kns[1] ... ,kns[i]->left, ... kns[n-1]})
     +
    ttn(b,n,kns[0],{kns[1] ... ,kns[i]->right, ... kns[n-1]})
*/
double two_ttn(int b,int n,knode **kns,int i);

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


*/
double ttn(int b,int n,knode **kns);


/* Returns the number of distinct n-tuples of strictly sorted labels such that
   the first label is from knode kns[0], the second from kns[1] and the
   n'th from kns[n-1].
   
   A n-tuple of labels ( lab1 , lab2 , ... lab[n] ) is strictly sorted
   if and only if forall i in {1,2,..n-1} lab_i < lab_[i+1].

   This number is the maximum possible number of tuples chosen from
   this set of knodes that could possibly match a symmetric (scalar) matcher.
   If the knodes are out of order the answer will come back zero.
*/
double total_num_ntuples_symmetric(int n,knode **kns);

/* If we're using a compound matcher, it's easy. It's always
   the product of the number of points in the knodes because all
   orderings (permutations) can be counted. */
double total_num_ntuples_assymmetric(int n,knode **kns);

/* If I give you an estimate of some number V* as Vhat where
   Vhat = (lo + hi)/2, and assuming that lo <= V* <= hi,
   what is the largest fractional error I could make. I.E.,
   how big might

      | Vhat - V* |
      -------------
           V*

   be? */
double total_num_ntuples(int n,bool use_symmetry,knode **kns);

dyv *mk_weighted_total_ntuples(int n,bool use_symmetry,knode **kns);

double compute_errfrac(double lo,double hi);

double total_2pt_tuples(twinpack *tp);

//nout *mk_run_npt_from_twinpack(twinpack *tp,params *ps,matcher *ma);
nout *mk_run_npt_from_twinpack(twinpack *tp,params *ps,matcher *ma,matcher *ma2);

nout *mk_2pt_nout(twinpack *tp,double thresh_ntuples,double connolly_thresh,
		  double lo_radius,double hi_radius);

#endif /* #ifndef NPT_H */














