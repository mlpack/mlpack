/*
   File:        projnpt.h
   Author:      Alexander Gray
   Description: 
*/

#ifndef PROJNPT_H
#define PROJNPT_H

void min_and_max_dsqd_proj(dyv *metric, hrect *hr1, hrect *hr2,
                           double *min_dsqd_between_hrs,
                           double *max_dsqd_between_hrs,
                           int projection,int projmethod);
void min_and_max_dsqd_proj_both(dyv *metric, hrect *hr1, hrect *hr2,
                                double *min_dsqd_between_hrs_para,
                                double *max_dsqd_between_hrs_para,
                                double *min_dsqd_between_hrs_perp,
                                double *max_dsqd_between_hrs_perp,
                                int projmethod);
int matcher_test_hrect_pair_proj(matcher *ma,hrect *hr1,hrect *hr2,
                                 int tuple_index_1,int tuple_index_2,
                                 double min_dsqd_between_hrs,
                                 double max_dsqd_between_hrs);
int matcher_permute_test_hrect_pair_proj(matcher *ma,hrect *hr1,hrect *hr2,
                                         int pt_tuple_index_1,
                                         int pt_tuple_index_2,
                                         ivec *permute_status, imat *num_incons,
                                         imat *permutation_cache,
                                         double min_dsqd_between_hrs,
                                         double max_dsqd_between_hrs);
int fast_npt_proj(mapshape *ms,dym **xs,dym **ws,
				  matcher *ma_para, matcher *ma_perp,
				  bool use_symmetry,bool use_permutes,knode **kns,
				  double thresh_ntuples,double connolly_thresh,
				  double *lobound,double *hibound, dyv *wlobound, dyv *whibound,
				  dyv *wresult,dyv *wsum,dyv *wsumsq,
				  imat *permutation_cache,int depth, 
				  int projection,int projmethod);

#endif /* #ifndef PROJNPT_H */
