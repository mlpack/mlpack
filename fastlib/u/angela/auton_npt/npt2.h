/*
   File:        npt2.h
   Author:      Alexander Gray
   Description: Header for Faster N-point computation
*/


#ifndef NPT2_H
#define NPT2_H

//#include "fib.h"
//#include "fibpriv.h"
#include "sheap.h"

double fast_npt2(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                 bool use_symmetry,bool use_permutes,knode **kns,
                 double thresh_ntuples,double connolly_thresh,
                 double *lobound,double *hibound, 
                 dyv *wlobound, dyv *whibound,dyv *wresult,dyv *wsum,dyv *wsumsq,
                 imat *permutation_cache,int depth,
                 imat *known_ndpairs,dym *known_dists, ivec **maps, int *starts);
imat *prepare_known_ndpairs_matrix(imat *ko,int s);

double slow_npt2_helper(mapshape *ms,dym **xs,dym **ws, matcher *ma,
                        bool use_symmetry,int k,int *row_indexes,ivec **rowsets,
                        dyv *wresult,dyv *wsum, dyv *wsumsq, 
                        imat *known_ndpairs,dym *known_dists);
double slow_npt2(mapshape *ms,dym **xs,dym **ws,matcher *ma,bool use_symmetry,
                 ivec **rowsets,dyv *wresult, dyv *wsum,dyv *wsumsq, 
                 imat *known_ndpairs,dym *known_dists);

double slow_permute_npt2_helper(mapshape *ms,dym **xs,dym **ws,matcher *ma,
                                int k,int *row_indexes,ivec **rowsets,
                                imat *permutation_cache,ivec *permutes_ok,
                                dyv *wresult,dyv *wsum,dyv *wsumsq,
                                imat *known_ndpairs,dym *known_dists);
double slow_permute_npt2(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
						 ivec **rowsets,imat *permutation_cache,dyv *wresult,
						 dyv *wsum, dyv *wsumsq,imat *known_ndpairs,
						 dym *known_dists);

double compute_nsamples(double p, double s, double eps);
double sample_npt(mapshape *ms, dym **xs,dym **ws, matcher *ma, knode **kns,
                  bool use_symmetry,bool use_permutes,imat *permutation_cache,
                  dyv *wresult,dyv *wsum, dyv *wsumsq,
                  imat *known_ndpairs,dym *known_dists,ivec **maps,int *starts,
				  double nsamples);
dyv *mk_compute_proportions(sheap *fsh, double ntuples_total);
double sample_npt_union(mapshape *ms, dym **xs,dym **ws, matcher *ma, 
						sheap *fsh, dyv *proportions,
						bool use_symmetry,bool use_permutes,
						imat *permutation_cache,
						dyv *wresult,dyv *wsum, dyv *wsumsq,
						imat *known_ndpairs,dym *known_dists,ivec **maps,
						int *starts, double nsamples);

nouts *mk_multi_run_npt2(twinpack *tp,params *ps,string_array *matcher_strings);
bool matcher_test_point_pair2(matcher *ma,dym *x1,dym *x2,int row1,int row2,
                              int tuple_index_1,int tuple_index_2,
                              dym *known_dists);
bool matcher_permute_test_point_pair2(matcher *ma,dym *x1,dym *x2,
									  int row1,int row2,
									  int pt_tuple_index_1,int pt_tuple_index_2,
									  imat *permutation_cache,
									  ivec *permutes_ok,dym *known_dists);
void create_virtual_index_to_dym_row_map(knode *kn, ivec *map, int *curr_index,
										 int start_index);
void create_dym_row_to_virtual_index_map(knode *kn, ivec *map, int *curr_index);

#endif /* #ifndef NPT2_H */
