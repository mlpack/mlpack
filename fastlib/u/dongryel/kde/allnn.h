#ifndef ALLNN_H
#define ALLNN_H

#include "batree.h"
#include "amdmex.h"
#include "amdym.h"
#include "amdyv.h"
#include "mrkd.h"

void kdtree_allknearest_neighbor_init(knode *Q);

void batree_allknearest_neighbor_init(banode *Q);

void batree_allknearest_neighbor(int k, dym *sps, banode *Q, banode *R, 
                                 double mindist, ivec_array *k_best_rows, 
                                 dyv_array* k_best_dists,
                                 ivec *kth_best_indx, dyv *kth_best_dist);

void batree_allnearest_neighbor2(dym *sps, dym *sps2, banode *Q,
                                 banode *R, double mindist, ivec *nn_rows,
                                 dyv* nn_dists);

void batree_allnearest_neighbor(dym *sps, banode *Q, banode *R,
                                double mindist, ivec *nn_rows, dyv* nn_dists);

void kdtree_allnearest_neighbor2(dym *sps, dym *sps2, knode *Q, 
				 knode *R, double mindist, ivec *nn_rows, 
				 dyv* nn_dists, dyv *metric);

void kdtree_allnearest_neighbor(dym *sps, knode *Q, knode *R,
                                double mindist, ivec *nn_rows, dyv* nn_dists,
				dyv *metric);

void verify_allnn(dym *query, dym *reference, dyv *guess_nn_dists,
		  bool selfcase);

#endif /* #ifndef ALLNN_H */
