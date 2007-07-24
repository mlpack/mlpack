#include <time.h>
#include <math.h>
#include <string.h>
#include <values.h>
#include "batree.h"
#include "distutils.h"
#include "ballutils.h"
#include "batree.h"
#include "allnn.h"
#include "kde.h"
#include "mrkd.h"
#include "my_time.h"

/*****************************************************************************/
void kdtree_allnearest_neighbor2(dym *sps, dym *sps2, knode *Q, 
				 knode *R, double mindist, ivec *nn_rows, 
				 dyv* nn_dists, dyv *metric)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if(mindist > Q->mindist_sofar) {
    return;
  }
  else {
    if(knode_is_leaf(Q) && knode_is_leaf(R)) {
      int i; 
      double max_of_node = 0;
      for(i=0; i< num_queries; i++) {
        int row_i = ivec_ref(Q->rows,i), j; double d;

        for(j=0; j<num_data; j++) {
          int row_j = ivec_ref(R->rows,j);

          if ((!LOO || !SELFCASE) || (row_j != row_i)) {
            double dist = sqrt(row_metric_dsqd(sps,sps,NULL,row_i,row_j));
            if(dist < dyv_ref(nn_dists,row_i)) { 
	      dyv_set(nn_dists,row_i,dist); 
	      ivec_set(nn_rows,row_i,row_j); 
	    }
          }
        } 
        d = dyv_ref(nn_dists,row_i); 
	if(d > max_of_node) 
	  max_of_node = d;
      }
      Q->mindist_sofar = max_of_node;
    }
    else if(knode_is_leaf(Q) && !knode_is_leaf(R)) {
      double dist1 = hrect_min_metric_dist(metric, Q->hr, R->left->hr);
      double dist2 = hrect_min_metric_dist(metric, Q->hr, R->right->hr);
      
      knode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      knode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);
      
      kdtree_allnearest_neighbor2(sps,sps2, Q,first,first_dist,
                                 nn_rows,nn_dists, metric);
      kdtree_allnearest_neighbor2(sps,sps2, Q,second,second_dist,
                                 nn_rows,nn_dists, metric);
    }
    else if(!knode_is_leaf(Q) && knode_is_leaf(R))
    {
      double dist1 = hrect_min_metric_dist(metric, Q->left->hr, R->hr);
      double dist2 = hrect_min_metric_dist(metric, Q->right->hr, R->hr);

      
      knode *first = (dist1 < dist2) ? Q->left : Q->right;
      double first_dist = real_min(dist1,dist2);
      knode *second = (dist1 < dist2) ? Q->right : Q->left;
      double second_dist = real_max(dist1,dist2);
      
      kdtree_allnearest_neighbor2(sps,sps2, first,R,first_dist,
				  nn_rows,nn_dists, metric);
      kdtree_allnearest_neighbor2(sps,sps2, second,R,second_dist,
                                 nn_rows,nn_dists, metric);
      
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->left->mindist_sofar, 
					   Q->right->mindist_sofar));
    }
    else if(!knode_is_leaf(Q) && !knode_is_leaf(R)) {
      {
	double dist1 = hrect_min_metric_dist(metric, Q->left->hr, R->left->hr);
	double dist2 = hrect_min_metric_dist(metric, Q->left->hr, R->right->hr);
	
	knode *first = (dist1 < dist2) ? R->left : R->right;
	double first_dist = real_min(dist1,dist2);
	knode *second = (dist1 < dist2) ? R->right : R->left;
	double second_dist = real_max(dist1,dist2);
	
	kdtree_allnearest_neighbor2(sps,sps2, Q->left,first,first_dist,
				    nn_rows,nn_dists, metric);
	kdtree_allnearest_neighbor2(sps,sps2, Q->left,second,second_dist,
				    nn_rows,nn_dists, metric);
      }
      {
	double dist1 = hrect_min_metric_dist(metric, Q->right->hr, R->left->hr);
	double dist2 = hrect_min_metric_dist(metric, Q->right->hr, R->right->hr);
	
	knode *first = (dist1 < dist2) ? R->left : R->right;
	double first_dist = real_min(dist1,dist2);
	knode *second = (dist1 < dist2) ? R->right : R->left;
	double second_dist = real_max(dist1,dist2);
	
	kdtree_allnearest_neighbor2(sps,sps2, Q->right,first,first_dist,
				    nn_rows,nn_dists, metric);
	kdtree_allnearest_neighbor2(sps,sps2, Q->right,second,second_dist,
				    nn_rows,nn_dists, metric);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->left->mindist_sofar, Q->right->mindist_sofar));
    }
  }
}

void kdtree_allnearest_neighbor(dym *sps, knode *Q, knode *R,
                                double mindist, ivec *nn_rows, dyv* nn_dists,
				dyv *metric)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if(mindist > Q->mindist_sofar) {
    return;
  }
  else {
    if(knode_is_leaf(Q) && knode_is_leaf(R)) {
      int i; 
      double max_of_node = 0;
      for(i=0; i< num_queries; i++) {
        int row_i = ivec_ref(Q->rows,i), j; double d;

        for(j=0; j<num_data; j++) {
          int row_j = ivec_ref(R->rows,j);

          if (row_j != row_i) {
            double dist = sqrt(row_metric_dsqd(sps,sps,NULL,row_i,row_j));
            if(dist < dyv_ref(nn_dists,row_i)) { 
	      dyv_set(nn_dists,row_i,dist); 
	      ivec_set(nn_rows,row_i,row_j); 
	    }
          }
        } 
        d = dyv_ref(nn_dists,row_i); 
	if(d > max_of_node) 
	  max_of_node = d;
      }
      Q->mindist_sofar = max_of_node;
    }
    else if(knode_is_leaf(Q) && !knode_is_leaf(R)) {
      double dist1 = hrect_min_metric_dist(metric, Q->hr, R->left->hr);
      double dist2 = hrect_min_metric_dist(metric, Q->hr, R->right->hr);

      knode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      knode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);
      
      kdtree_allnearest_neighbor(sps,Q,first,first_dist,
                                 nn_rows,nn_dists, metric);
      kdtree_allnearest_neighbor(sps,Q,second,second_dist,
                                 nn_rows,nn_dists, metric);
    }
    else if(!knode_is_leaf(Q) && knode_is_leaf(R))
    {
      double dist1 = hrect_min_metric_dist(metric, Q->left->hr, R->hr);
      double dist2 = hrect_min_metric_dist(metric, Q->right->hr, R->hr);

      knode *first = (dist1 < dist2) ? Q->left : Q->right;
      double first_dist = real_min(dist1,dist2);
      knode *second = (dist1 < dist2) ? Q->right : Q->left;
      double second_dist = real_max(dist1,dist2);
      
      kdtree_allnearest_neighbor(sps,first,R,first_dist,
                                 nn_rows,nn_dists, metric);
      kdtree_allnearest_neighbor(sps,second,R,second_dist,
                                 nn_rows,nn_dists, metric);
      
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->left->mindist_sofar, 
					   Q->right->mindist_sofar));
    }
    else if(!knode_is_leaf(Q) && !knode_is_leaf(R)) {
      {
	double dist1 = hrect_min_metric_dist(metric, Q->left->hr, R->left->hr);
	double dist2 = hrect_min_metric_dist(metric,Q->left->hr, R->right->hr);

	knode *first = (dist1 < dist2) ? R->left : R->right;
	double first_dist = real_min(dist1,dist2);
	knode *second = (dist1 < dist2) ? R->right : R->left;
	double second_dist = real_max(dist1,dist2);
	
	kdtree_allnearest_neighbor(sps,Q->left,first,first_dist,
				   nn_rows,nn_dists, metric);
	kdtree_allnearest_neighbor(sps,Q->left,second,second_dist,
				   nn_rows,nn_dists, metric);
      }
      {
	double dist1 = hrect_min_metric_dist(metric, Q->right->hr,R->left->hr);
	double dist2 = hrect_min_metric_dist(metric,Q->right->hr,R->right->hr);
	
	knode *first = (dist1 < dist2) ? R->left : R->right;
	double first_dist = real_min(dist1,dist2);
	knode *second = (dist1 < dist2) ? R->right : R->left;
	double second_dist = real_max(dist1,dist2);
	
	kdtree_allnearest_neighbor(sps,Q->right,first,first_dist,
				   nn_rows,nn_dists, metric);
	kdtree_allnearest_neighbor(sps,Q->right,second,second_dist,
				   nn_rows,nn_dists, metric);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->left->mindist_sofar,
					   Q->right->mindist_sofar));
    }
  }
}

void kdtree_allknearest_neighbor_init(knode *Q)
{
  if(Q != NULL) {
    Q->mindist_sofar = FLT_MAX;
    
    kdtree_allknearest_neighbor_init(Q->left);
    kdtree_allknearest_neighbor_init(Q->right);
  }
}

void batree_allknearest_neighbor_init(banode *Q)
{
  if(Q != NULL) {
    Q->mindist_sofar=FLT_MAX;
    Q->maxdist_sofar=0;

    batree_allknearest_neighbor_init(Q->left);
    batree_allknearest_neighbor_init(Q->right);
  }
}

void batree_allknearest_neighbor(int k, dym *sps, banode *Q, banode *R, 
                                 double mindist, ivec_array *k_best_rows, 
                                 dyv_array* k_best_dists,
                                 ivec *kth_best_indx, dyv *kth_best_dist)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if ( mindist > Q->mindist_sofar ) {
    return;
  }
  else 
  {
    if ( banode_is_leaf(Q) && banode_is_leaf(R) )
    {
      int i; double max_of_node = 0;
      for ( i = 0 ; i < num_queries ; i++ ) {
        int row_i = ivec_ref(Q->rows,i), j; double d;   
	ivec *k_best_rows_i = ivec_array_ref(k_best_rows,row_i);
        dyv *k_best_dists_i = dyv_array_ref(k_best_dists,row_i);

        for ( j = 0 ; j < num_data ; j++ ) {
          int row_j = ivec_ref(R->rows,j);
          
          if ( row_j != row_i ) {
            double dist = sqrt(row_metric_dsqd(sps,sps,NULL,row_i,row_j)); 
	    int w;

            if (ivec_size(k_best_rows_i) < k) {
              /* stick in the new one */
              add_to_ivec(k_best_rows_i,row_j);add_to_dyv(k_best_dists_i,dist);
              if (ivec_size(k_best_rows_i) == k) {
                /* record which is the worst */
                w = dyv_argmax(k_best_dists_i); d = dyv_ref(k_best_dists_i,w);
                ivec_set(kth_best_indx,row_i, w);
		dyv_set(kth_best_dist,row_i,d);
              }
            } else {
              if (dist < dyv_ref(kth_best_dist,row_i)) { 
                /* replace the worst one with the new one */
                ivec_set(k_best_rows_i, ivec_ref(kth_best_indx,row_i), row_j);
                dyv_set(k_best_dists_i, ivec_ref(kth_best_indx,row_i), dist);
                /* record which is the new worst */
                w = dyv_argmax(k_best_dists_i); d = dyv_ref(k_best_dists_i,w);
                ivec_set(kth_best_indx,row_i, w);
		dyv_set(kth_best_dist,row_i,d);
              }
            }
          }           
        }
        d = dyv_ref(kth_best_dist,row_i); 
	if ( d > max_of_node ) max_of_node = d;
      }
      Q->mindist_sofar = max_of_node;
    }
    else if ( banode_is_leaf(Q) && !banode_is_leaf(R) )
    {
      double dist1 = dyv_distance(Q->pivot,R->left->pivot) - Q->radius 
                        - R->left->radius;
      double dist2 = dyv_distance(Q->pivot,R->right->pivot) - Q->radius 
                        - R->right->radius;

      banode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);
      
      batree_allknearest_neighbor(k,sps,Q,first,first_dist,k_best_rows,
                                  k_best_dists,kth_best_indx,kth_best_dist);
      batree_allknearest_neighbor(k,sps,Q,second,second_dist,k_best_rows,
                                  k_best_dists,kth_best_indx,kth_best_dist);
    }
    else if ( !banode_is_leaf(Q) && banode_is_leaf(R) )
    {
      double dist1 = dyv_distance(R->pivot,Q->left->pivot) - R->radius 
                        - Q->left->radius;
      double dist2 = dyv_distance(R->pivot,Q->right->pivot) - R->radius 
                        - Q->right->radius;
      banode *first = (dist1 < dist2) ? Q->left : Q->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? Q->right : Q->left;
      double second_dist = real_max(dist1,dist2);
      
      batree_allknearest_neighbor(k,sps,first,R,first_dist,k_best_rows,
                                  k_best_dists,kth_best_indx,kth_best_dist);
      batree_allknearest_neighbor(k,sps,second,R,second_dist,k_best_rows,
                                  k_best_dists,kth_best_indx,kth_best_dist);

      Q->mindist_sofar = real_min(Q->mindist_sofar,
                  real_max(Q->left->mindist_sofar, Q->right->mindist_sofar));
    }
    else if ( !banode_is_leaf(Q) && !banode_is_leaf(R) )
    {
      {
      double dist1 = dyv_distance(Q->left->pivot,R->left->pivot) 
                        - Q->left->radius - R->left->radius;
      double dist2 = dyv_distance(Q->left->pivot,R->right->pivot) 
                        - Q->left->radius - R->right->radius;
      banode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);
      
      batree_allknearest_neighbor(k,sps,Q->left,first,first_dist,
                                  k_best_rows,k_best_dists,kth_best_indx,
                                  kth_best_dist);
      batree_allknearest_neighbor(k,sps,Q->left,second,second_dist,
                                  k_best_rows,k_best_dists,kth_best_indx,
                                  kth_best_dist);
      }
      {
      double dist1 = dyv_distance(Q->right->pivot,R->left->pivot) 
                        - Q->right->radius - R->left->radius;
      double dist2 = dyv_distance(Q->right->pivot,R->right->pivot) 
                        - Q->right->radius - R->right->radius;
      banode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);
      
      batree_allknearest_neighbor(k,sps,Q->right,first,first_dist,
                                  k_best_rows,k_best_dists,kth_best_indx,
                                  kth_best_dist);
      batree_allknearest_neighbor(k,sps,Q->right,second,second_dist,
                                  k_best_rows,k_best_dists,kth_best_indx,
                                  kth_best_dist);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
                  real_max(Q->left->mindist_sofar, Q->right->mindist_sofar));
    }
  }
}

void batree_allnearest_neighbor2(dym *sps, dym *sps2, banode *Q,
                                 banode *R, double mindist, ivec *nn_rows,
                                 dyv* nn_dists)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if(mindist > Q->mindist_sofar) {
    return;
  }
  else {
    if(banode_is_leaf(Q) && banode_is_leaf(R)) {
      int i;
      double max_of_node = 0;
      for(i=0; i< num_queries; i++) {
        int row_i = ivec_ref(Q->rows,i), j; double d;

        for(j=0; j<num_data; j++) {
          int row_j = ivec_ref(R->rows,j);

          if ((!LOO || !SELFCASE) || (row_j != row_i)) {
            double dist = sqrt(row_metric_dsqd(sps,sps2,NULL,row_i,row_j));
            if(dist < dyv_ref(nn_dists,row_i)) {
              dyv_set(nn_dists,row_i,dist);
              ivec_set(nn_rows,row_i,row_j);
            }
          }
        }
        d = dyv_ref(nn_dists,row_i);
        if(d > max_of_node)
          max_of_node = d;
      }
      Q->mindist_sofar = max_of_node;
    }
    else if(banode_is_leaf(Q) && !banode_is_leaf(R)) {
      double dist1 = (dyv_distance(Q->pivot,R->left->pivot) - Q->radius
                      - R->left->radius);
      double dist2 = (dyv_distance(Q->pivot,R->right->pivot) - Q->radius
                      - R->right->radius);

      banode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);

      batree_allnearest_neighbor2(sps,sps2, Q,first,first_dist,
				  nn_rows,nn_dists);
      batree_allnearest_neighbor2(sps,sps2, Q,second,second_dist,
				  nn_rows,nn_dists);
    }
    else if(!banode_is_leaf(Q) && banode_is_leaf(R))
      {
	double dist1 = (dyv_distance(R->pivot,Q->left->pivot) - R->radius
			- Q->left->radius);
	double dist2 = (dyv_distance(R->pivot,Q->right->pivot) - R->radius
			- Q->right->radius);

	banode *first = (dist1 < dist2) ? Q->left : Q->right;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? Q->right : Q->left;
	double second_dist = real_max(dist1,dist2);

	batree_allnearest_neighbor2(sps,sps2, first,R,first_dist,
				    nn_rows,nn_dists);
	batree_allnearest_neighbor2(sps,sps2, second,R,second_dist,
				    nn_rows,nn_dists);

	Q->mindist_sofar = real_min(Q->mindist_sofar,
				    real_max(Q->left->mindist_sofar,
					     Q->right->mindist_sofar));
      }
    else if(!banode_is_leaf(Q) && !banode_is_leaf(R)) {
      {
        double dist1 = (dyv_distance(Q->left->pivot,R->left->pivot)
                        - Q->left->radius - R->left->radius);
        double dist2 = (dyv_distance(Q->left->pivot,R->right->pivot)
                        - Q->left->radius - R->right->radius);

        banode *first = (dist1 < dist2) ? R->left : R->right;
        double first_dist = real_min(dist1,dist2);
        banode *second = (dist1 < dist2) ? R->right : R->left;
        double second_dist = real_max(dist1,dist2);

        batree_allnearest_neighbor2(sps,sps2, Q->left,first,first_dist,
				    nn_rows,nn_dists);
        batree_allnearest_neighbor2(sps,sps2, Q->left,second,second_dist,
				    nn_rows,nn_dists);
      }
      {
        double dist1 = (dyv_distance(Q->right->pivot,R->left->pivot)
                        - Q->right->radius - R->left->radius);
        double dist2 = (dyv_distance(Q->right->pivot,R->right->pivot)
                        - Q->right->radius - R->right->radius);

        banode *first = (dist1 < dist2) ? R->left : R->right;
        double first_dist = real_min(dist1,dist2);
        banode *second = (dist1 < dist2) ? R->right : R->left;
        double second_dist = real_max(dist1,dist2);

        batree_allnearest_neighbor2(sps,sps2, Q->right,first,first_dist,
				    nn_rows,nn_dists);
        batree_allnearest_neighbor2(sps,sps2, Q->right,second,second_dist,
                                    nn_rows,nn_dists);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
                                  real_max(Q->left->mindist_sofar,
					   Q->right->mindist_sofar));
    }
  }
}

void batree_allnearest_neighbor(dym *sps, banode *Q, banode *R,
                                double mindist, ivec *nn_rows, dyv* nn_dists)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if(mindist > Q->mindist_sofar) {
    return;
  }
  else {
    if(banode_is_leaf(Q) && banode_is_leaf(R)) {
      int i; 
      double max_of_node = 0;
      for(i=0; i< num_queries; i++) {
        int row_i = ivec_ref(Q->rows,i), j; double d;

        for(j=0; j<num_data; j++) {
          int row_j = ivec_ref(R->rows,j);

          if (row_j != row_i) {
            double dist = sqrt(row_metric_dsqd(sps,sps,NULL,row_i,row_j));
            if(dist < dyv_ref(nn_dists,row_i)) { 
	      dyv_set(nn_dists,row_i,dist); 
	      ivec_set(nn_rows,row_i,row_j); 
	    }
          }
        } 
        d = dyv_ref(nn_dists,row_i); 
	if(d > max_of_node) 
	  max_of_node = d;
      }
      Q->mindist_sofar = max_of_node;
    }
    else if(banode_is_leaf(Q) && !banode_is_leaf(R)) {
      double dist1 = (dyv_distance(Q->pivot,R->left->pivot) - Q->radius 
		      - R->left->radius);
      double dist2 = (dyv_distance(Q->pivot,R->right->pivot) - Q->radius 
		      - R->right->radius);
      banode *first = (dist1 < dist2) ? R->left : R->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->right : R->left;
      double second_dist = real_max(dist1,dist2);
      
      batree_allnearest_neighbor(sps,Q,first,first_dist,
                                 nn_rows,nn_dists);
      batree_allnearest_neighbor(sps,Q,second,second_dist,
                                 nn_rows,nn_dists);
    }
    else if(!banode_is_leaf(Q) && banode_is_leaf(R))
    {
      double dist1 = (dyv_distance(R->pivot,Q->left->pivot) - R->radius 
		      - Q->left->radius);
      double dist2 = (dyv_distance(R->pivot,Q->right->pivot) - R->radius 
		      - Q->right->radius);
      banode *first = (dist1 < dist2) ? Q->left : Q->right;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? Q->right : Q->left;
      double second_dist = real_max(dist1,dist2);
      
      batree_allnearest_neighbor(sps,first,R,first_dist,
                                 nn_rows,nn_dists);
      batree_allnearest_neighbor(sps,second,R,second_dist,
                                 nn_rows,nn_dists);
      
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->left->mindist_sofar, 
					   Q->right->mindist_sofar));
    }
    else if(!banode_is_leaf(Q) && !banode_is_leaf(R)) {
      {
	double dist1 = (dyv_distance(Q->left->pivot,R->left->pivot) 
			- Q->left->radius - R->left->radius);
	double dist2 = (dyv_distance(Q->left->pivot,R->right->pivot) 
			- Q->left->radius - R->right->radius);
	banode *first = (dist1 < dist2) ? R->left : R->right;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? R->right : R->left;
	double second_dist = real_max(dist1,dist2);
	
	batree_allnearest_neighbor(sps,Q->left,first,first_dist,
				   nn_rows,nn_dists);
	batree_allnearest_neighbor(sps,Q->left,second,second_dist,
				   nn_rows,nn_dists);
      }
      {
	double dist1 = (dyv_distance(Q->right->pivot,R->left->pivot) 
			- Q->right->radius - R->left->radius);
	double dist2 = (dyv_distance(Q->right->pivot,R->right->pivot) 
			- Q->right->radius - R->right->radius);
	banode *first = (dist1 < dist2) ? R->left : R->right;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? R->right : R->left;
	double second_dist = real_max(dist1,dist2);
	
	batree_allnearest_neighbor(sps,Q->right,first,first_dist,
				   nn_rows,nn_dists);
	batree_allnearest_neighbor(sps,Q->right,second,second_dist,
				   nn_rows,nn_dists);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->left->mindist_sofar, 
					   Q->right->mindist_sofar));
    }
  }
}

void verify_allnn(dym *query, dym *reference, dyv *guess_nn_dists,
		  bool selfcase)
{
  int q, r;
  for(q = 0; q < dym_rows(query); q++) {
    double best_dist = FLT_MAX;
    for(r = 0; r < dym_rows(reference); r++) {
      double dist;
      
      if((!selfcase) || (q != r))
	dist = sqrt(row_metric_dsqd(query,reference,NULL,q,r));
      else
	continue;

      if(dist < best_dist)
	best_dist = dist;
    }

    if(best_dist < dyv_ref(guess_nn_dists,q)) {
      printf("The actual 1-NN distance should be %g, not %g...\n", 
	     best_dist, dyv_ref(guess_nn_dists,q));
    }
  }
}

void verify(dym *sps, ivec_array *k_best_rows,
	    dyv_array *k_best_dists,
	    ivec *kth_best_indx,
	    dyv *kth_best_dist)
{
  int num_data = dym_rows(sps);
  int num_queries = dym_rows(sps);
  int i,k = ivec_size(ivec_array_ref(k_best_rows,0));

  {
    /* LINEAR SCAN: k-NN */
    for (i=0; i<num_queries; i++) {
      dyv *k_best_dists_i = mk_dyv(0);
      int j, a, w; double d = 0;
  
      for (j=0; j<num_data; j++) {
        if (j != i) {
          double dist = sqrt(row_metric_dsqd(sps,sps,NULL,i,j));
          Num_pt_dists -= 1;
          add_to_dyv(k_best_dists_i,dist);
        }
      }
      for (a=0; a<k; a++) {
        w = dyv_argmin(k_best_dists_i); d = dyv_ref(k_best_dists_i,w); 
        //printf("%g ",d);
        dyv_remove(k_best_dists_i,w);
      }
      //printf("\n");
      if (d != dyv_ref(kth_best_dist,i)) 
        printf("%g should be %g\n",dyv_ref(kth_best_dist,i),d);
      free_dyv(k_best_dists_i);
    }
  }
}
