#include <time.h>
#include <math.h>
#include <string.h>
#include "batree.h"
#include "amgr.h"
#include "ongr.h"
#include "dsut.h"
#include "hrect.h"
#include "distutils.h"
#include "sheap.h"
#include "ballutils.h"
#include "batree.h"
#include "allnn.h"
#include "dymutil.h"
#include "timing_dyvs.h"
#include "my_time.h"
#include "bruteforce.h"
#include "stats.h"

/* TIMING is defined in batree.h, because it has to be used in this file
 * and allnn. -crotella
 */

#define NUM_ITERRATIONS 10

extern int Num_ball_prunes;
extern int Num_hrect_prunes;
extern double Verbosity;

#define DEBUG 0
#ifndef MBW
#define MBW 1e-35
#endif

#ifndef FLT_MAX
#define FLT_MAX 1e30
#endif

dyv_array *mk_dyv_array_from_dym(dym *d) {
  int i;
  dyv_array *ret = mk_empty_dyv_array();
  for(i=0; i<dym_rows(d); i++) {
    dyv *row = mk_dyv_from_dym_row(d, i);
    add_to_dyv_array(ret, row);
    free_dyv(row);
  }
  return ret;
}

extern dym *mk_dym_from_filename_improved(char *filename, int argc, 
					  char *argv[]);

void batree_allnearest_neighbor2(dyv_array *sps, dyv_array *sps2, banode *Q, 
				 banode *R, double mindist, ivec *nn_rows, 
				 dyv* nn_dists)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if(mindist > Q->mindist_sofar) {
    Num_ball_prunes++; return;  //Num_hrect_prunes++; return;
  }
  else {
    if(banode_is_leaf(Q) && banode_is_leaf(R)) {
      int i; 
      double max_of_node = 0;
      for(i=0; i< num_queries; i++) {
        int row_i = ivec_ref(Q->rows,i), j; double d;
        dyv *point_i = dyv_array_ref(sps,row_i);

        for(j=0; j<num_data; j++) {
          int row_j = ivec_ref(R->rows,j);
          dyv *point_j = dyv_array_ref(sps2,row_j);

          if (row_j != row_i) {
            double dist = dyv_distance(point_i,point_j);
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
      //double dist1 = hrect_hrect_min_dsqd(Q->hr,R->child1->hr);
      double dist1 = (dyv_distance(Q->pivot,R->child1->pivot) - Q->radius 
		      - R->child1->radius);
      double dist2 = (dyv_distance(Q->pivot,R->child2->pivot) - Q->radius 
		      - R->child2->radius);
      
      banode *first = (dist1 < dist2) ? R->child1 : R->child2;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->child2 : R->child1;
      double second_dist = real_max(dist1,dist2);
      
      batree_allnearest_neighbor2(sps,sps2, Q,first,first_dist,
                                 nn_rows,nn_dists);
      batree_allnearest_neighbor2(sps,sps2, Q,second,second_dist,
                                 nn_rows,nn_dists);
    }
    else if(!banode_is_leaf(Q) && banode_is_leaf(R))
    {
      double dist1 = (dyv_distance(R->pivot,Q->child1->pivot) - R->radius 
		      - Q->child1->radius);
      double dist2 = (dyv_distance(R->pivot,Q->child2->pivot) - R->radius 
		      - Q->child2->radius);
      
      banode *first = (dist1 < dist2) ? Q->child1 : Q->child2;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? Q->child2 : Q->child1;
      double second_dist = real_max(dist1,dist2);
      
      batree_allnearest_neighbor2(sps,sps2, first,R,first_dist,
                                 nn_rows,nn_dists);
      batree_allnearest_neighbor2(sps,sps2, second,R,second_dist,
                                 nn_rows,nn_dists);
      
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->child1->mindist_sofar, 
					   Q->child2->mindist_sofar));
    }
    else if(!banode_is_leaf(Q) && !banode_is_leaf(R)) {
      {
	double dist1 = (dyv_distance(Q->child1->pivot,R->child1->pivot) 
			- Q->child1->radius - R->child1->radius);
	double dist2 = (dyv_distance(Q->child1->pivot,R->child2->pivot) 
			- Q->child1->radius - R->child2->radius);
	
	banode *first = (dist1 < dist2) ? R->child1 : R->child2;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? R->child2 : R->child1;
	double second_dist = real_max(dist1,dist2);
	
	batree_allnearest_neighbor2(sps,sps2, Q->child1,first,first_dist,
				   nn_rows,nn_dists);
	batree_allnearest_neighbor2(sps,sps2, Q->child1,second,second_dist,
				   nn_rows,nn_dists);
      }
      {
	double dist1 = (dyv_distance(Q->child2->pivot,R->child1->pivot) 
			- Q->child2->radius - R->child1->radius);
	double dist2 = (dyv_distance(Q->child2->pivot,R->child2->pivot) 
			- Q->child2->radius - R->child2->radius);
	
	banode *first = (dist1 < dist2) ? R->child1 : R->child2;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? R->child2 : R->child1;
	double second_dist = real_max(dist1,dist2);
	
	batree_allnearest_neighbor2(sps,sps2, Q->child2,first,first_dist,
				   nn_rows,nn_dists);
	batree_allnearest_neighbor2(sps,sps2, Q->child2,second,second_dist,
				    nn_rows,nn_dists);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->child1->mindist_sofar, Q->child2->mindist_sofar));
    }
  }
}

void batree_allnearest_neighbor(dyv_array *sps, banode *Q, banode *R,
                                double mindist, ivec *nn_rows, dyv* nn_dists)
{
  int num_queries = Q->num_points, num_data = R->num_points;

  if(mindist > Q->mindist_sofar) {
    Num_ball_prunes++; return;  //Num_hrect_prunes++; return;
  }
  else {
    if(banode_is_leaf(Q) && banode_is_leaf(R)) {
      int i; 
      double max_of_node = 0;
      for(i=0; i< num_queries; i++) {
        int row_i = ivec_ref(Q->rows,i), j; double d;
        dyv *point_i = dyv_array_ref(sps,row_i);

        for(j=0; j<num_data; j++) {
          int row_j = ivec_ref(R->rows,j);
          dyv *point_j = dyv_array_ref(sps,row_j);

          if (row_j != row_i) {
            double dist = dyv_distance(point_i,point_j);
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
      //double dist1 = hrect_hrect_min_dsqd(Q->hr,R->child1->hr);
      double dist1 = (dyv_distance(Q->pivot,R->child1->pivot) - Q->radius 
		      - R->child1->radius);
      double dist2 = (dyv_distance(Q->pivot,R->child2->pivot) - Q->radius 
		      - R->child2->radius);
      
      banode *first = (dist1 < dist2) ? R->child1 : R->child2;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? R->child2 : R->child1;
      double second_dist = real_max(dist1,dist2);
      
      batree_allnearest_neighbor(sps,Q,first,first_dist,
                                 nn_rows,nn_dists);
      batree_allnearest_neighbor(sps,Q,second,second_dist,
                                 nn_rows,nn_dists);
    }
    else if(!banode_is_leaf(Q) && banode_is_leaf(R))
    {
      double dist1 = (dyv_distance(R->pivot,Q->child1->pivot) - R->radius 
		      - Q->child1->radius);
      double dist2 = (dyv_distance(R->pivot,Q->child2->pivot) - R->radius 
		      - Q->child2->radius);
      
      banode *first = (dist1 < dist2) ? Q->child1 : Q->child2;
      double first_dist = real_min(dist1,dist2);
      banode *second = (dist1 < dist2) ? Q->child2 : Q->child1;
      double second_dist = real_max(dist1,dist2);
      
      batree_allnearest_neighbor(sps,first,R,first_dist,
                                 nn_rows,nn_dists);
      batree_allnearest_neighbor(sps,second,R,second_dist,
                                 nn_rows,nn_dists);
      
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->child1->mindist_sofar, 
					   Q->child2->mindist_sofar));
    }
    else if(!banode_is_leaf(Q) && !banode_is_leaf(R)) {
      {
	double dist1 = (dyv_distance(Q->child1->pivot,R->child1->pivot) 
			- Q->child1->radius - R->child1->radius);
	double dist2 = (dyv_distance(Q->child1->pivot,R->child2->pivot) 
			- Q->child1->radius - R->child2->radius);
	
	banode *first = (dist1 < dist2) ? R->child1 : R->child2;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? R->child2 : R->child1;
	double second_dist = real_max(dist1,dist2);
	
	batree_allnearest_neighbor(sps,Q->child1,first,first_dist,
				   nn_rows,nn_dists);
	batree_allnearest_neighbor(sps,Q->child1,second,second_dist,
				   nn_rows,nn_dists);
      }
      {
	double dist1 = (dyv_distance(Q->child2->pivot,R->child1->pivot) 
			- Q->child2->radius - R->child1->radius);
	double dist2 = (dyv_distance(Q->child2->pivot,R->child2->pivot) 
			- Q->child2->radius - R->child2->radius);
	
	banode *first = (dist1 < dist2) ? R->child1 : R->child2;
	double first_dist = real_min(dist1,dist2);
	banode *second = (dist1 < dist2) ? R->child2 : R->child1;
	double second_dist = real_max(dist1,dist2);
	
	batree_allnearest_neighbor(sps,Q->child2,first,first_dist,
				   nn_rows,nn_dists);
	batree_allnearest_neighbor(sps,Q->child2,second,second_dist,
				   nn_rows,nn_dists);
      }
      Q->mindist_sofar = real_min(Q->mindist_sofar,
				  real_max(Q->child1->mindist_sofar, Q->child2->mindist_sofar));
    }
  }
}

void allnnbatch_main(int argc, char **argv)
{
  char *filename = string_from_args("in",argc,argv,"default.fds");
  dym *data = mk_dym_from_filename_improved(filename, argc, argv);
  dyv_array *sps = mk_dyv_array_from_dym(data);
  int rmin = int_from_args("rmin",argc,argv,5);
  double mbw = double_from_args("mbw",argc,argv,MBW);
  int num_rows = sps->size;
  int iter, iter2;
  alloc_timing_dyvs();

  for(iter = 0; iter < 5; iter++) {
    for(iter2 = 0; iter2 < 2; iter2++) {

      dym *queries;
      dym *sub_data;
      //int local_queries = getnThPartOfnParts(data, 2, iter2, 
      //				     &queries, &sub_data);
      break_dym_into_train_and_test(data, num_rows / 2, &queries, &sub_data);


      dyv_array *queries_dyv_array = mk_dyv_array_from_dym(queries);
      dyv_array *sub_data_dyv_array = mk_dyv_array_from_dym(sub_data);
      
      free_dym(queries);
      free_dym(sub_data);
      
      batree *query_tree = mk_batree_from_dyv_array(queries_dyv_array,rmin,mbw,
						    TRUE);
      batree *sub_data_tree = mk_batree_from_dyv_array(sub_data_dyv_array, 
						       rmin, mbw, FALSE);
      
      int num_queries = dyv_array_size(queries_dyv_array);
      dyv *num_visited = mk_dyv(num_queries);
      
      /* hacky, buy necessary to avoid worse hacks later on... */
      
      ivec *true_nn_rows[1];
      ivec *guess_nn_rows[1];
      dyv *guess_nn_dists[1];
      
      true_nn_rows[0] = mk_ivec(num_queries);
      guess_nn_rows[0] = mk_ivec(num_queries);
      guess_nn_dists[0] = mk_constant_dyv(num_queries, FLT_MAX);
      
      double time1 = get_time();
      
      Num_pt_dists = 0; Num_hr_dists = 0; Num_ball_prunes = 0; 
      Num_hrect_prunes = 0;
      
      batree_allnearest_neighbor2(queries_dyv_array, sub_data_dyv_array,
				  query_tree->root,
				  sub_data_tree->root,FLT_MAX,
				  guess_nn_rows[0], guess_nn_dists[0]);
      
      double time2 = get_time();
      
      double query_time = (time2-time1);
      
      add_to_dyv(query_times, query_time);
      
      free_ivec(true_nn_rows[0]); 
      free_ivec(guess_nn_rows[0]);
      free_dyv(guess_nn_dists[0]); 
      free_dyv(num_visited); 
      
      
      free_dyv_array(sub_data_dyv_array);
      free_dyv_array(queries_dyv_array);
      free_batree(query_tree);
      free_batree(sub_data_tree);
    }
  }
  dyv *total = mk_dyv(dyv_size(build_times));
  for(iter = 0; iter < dyv_size(build_times); iter++)
    dyv_set(total, iter, dyv_ref(build_times, iter) + 
	    dyv_ref(query_times, iter)+ dyv_ref(query_build_times, iter));

  print_timing_report(TRUE);
  printf("Total Elapsed time:\n");
  printf("  STD: %f\n", dyv_sdev(total));
  printf("  MIN: %f\n", dyv_min(total));
  printf("  MAX: %f\n", dyv_max(total));
  printf("  AVE: %f\n", dyv_mean(total));
	 
  free_timing_dyvs();
  free_dym(data);
  free_dyv(total);
  free_dyv_array(sps);
}

void allnnsingle_main(int argc, char **argv)
{
  char *filename = string_from_args("in",argc,argv,"default.fds");
  dym *data = mk_dym_from_filename_improved(filename, argc, argv);
  dyv_array *sps = mk_dyv_array_from_dym(data);
  int rmin = int_from_args("rmin",argc,argv,5);
  double mbw = double_from_args("mbw",argc,argv,MBW);
  int num_rows = sps->size;
  double time1 = get_time();
  alloc_timing_dyvs();
  batree *data_tree = mk_batree_from_dyv_array(sps, rmin, mbw, FALSE);

  int r;
  for(r = 0; r < num_rows; r++) {
    dyv *query = dyv_array_ref(sps, r);
    ivec *visited;
    int row = batree_nearest_neighbor(sps, data_tree, query, r, &visited);
    free_ivec(visited);
  }
  double time2 = get_time();
  printf("Build + Query time: %f\n", time2 - time1);
  free_dyv_array(sps);
  free_dym(data);
  free_batree(data_tree);
  print_timing_report(FALSE);
  free_timing_dyvs();
}

void allnn_main_bruteforce(int argc, char **argv) {
  char *filename = string_from_args("in",argc,argv,"default.fds");
  dym *data = mk_dym_from_filename_improved(filename, argc, argv);
  dyv_array *sps = mk_dyv_array_from_dym(data);
  //free_dym(data);



  int i, j;
  int num_row = dym_rows(data);
  double time1 = get_time();
  for(i = 0; i < num_row; i++) {
    double best_dist = 1e90;
    int best_nn = -1;
    dyv *row = dyv_array_ref(sps, i);
    for(j = 0; j < num_row; j++) {
      double dist;
      if(i == j)
	continue;
      dyv *cmp = dyv_array_ref(sps, j);
      dist = dyv_dsqd(row, cmp);
      if(dist < best_dist) {
	best_dist = dist;
	best_nn = j;
      }
    }
  }
  double time2 = get_time();
  printf("Query time: %f\n", time2 - time1);
  free_dym(data);
  free_dyv_array(sps);
}

void allnn_main_single_tree(int argc, char **argv) {


  char *filename = string_from_args("in",argc,argv,"default.fds");
  dym *data = mk_dym_from_filename_improved(filename, argc, argv);
  dyv_array *sps = mk_dyv_array_from_dym(data);
  //free_dym(data);

  /* Mine: till here data has been read and an array of points created. sps is a pointer to a dyv_array which is a structure which holds a double dimensional array of points */


  alloc_timing_dyvs();

  int rmin = int_from_args("rmin",argc,argv,5);
  double mbw = double_from_args("mbw",argc,argv,MBW);

  double time1 = get_time();
  batree *bat = mk_batree_from_dyv_array(sps,rmin,mbw, FALSE);

  /* Mine: A batree has been created by using the set of points stored in dyv_array */

 
  int num_data = dyv_array_size(sps);
  int num_queries = num_data;
  dyv *num_visited = mk_dyv(num_queries);

  /* num_data and num_queries give the number of query and data points. Both atre set to the same value  */
  /* hacky, buy necessary to avoid worse hacks later on... */

  ivec *true_nn_rows[1];
  ivec *guess_nn_rows[1];
  dyv *guess_nn_dists[1];

  true_nn_rows[0] = mk_ivec(num_queries);
  guess_nn_rows[0] = mk_ivec(num_queries);
  guess_nn_dists[0] = mk_constant_dyv(num_queries, FLT_MAX);

  Num_pt_dists = 0; Num_hr_dists = 0; Num_ball_prunes = 0; 
  Num_hrect_prunes = 0;

 
    
  batree_allnearest_neighbor(sps,bat->root,bat->root,FLT_MAX,
			     guess_nn_rows[0], guess_nn_dists[0]);
  

  double time2 = get_time();

  double tot_time = (time2-time1);

  printf("Build + Query time: %f\n",tot_time); 

  
  print_timing_report(FALSE);
  
  free_ivec(true_nn_rows[0]);
  free_ivec(guess_nn_rows[0]);
  free_dyv(guess_nn_dists[0]);
  free_dyv(num_visited); 
  free_batree(bat); free_dyv_array(sps);
  free_timing_dyvs();

#ifdef TIMING
  printf("Timing enabled on this run.\n");
#endif

  free_dym(data);
}

void allnn_main(int argc,char *argv[]) {
  Verbosity = 0;
  allnn_main_single_tree(argc, argv);
}
