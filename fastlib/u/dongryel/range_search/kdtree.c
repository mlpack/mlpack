#include <math.h>
#include <string.h>
#include "hrect.h"
#include "distutils.h"
#include "kdtree.h"
#include "dymutil.h"
#include "timing_dyvs.h"
#include "my_time.h"


#define EXCLUDE 0
#define SUBSUME 1
#define INCONCLUSIVE 2


//#define TIMING

#define NUM_ITTERATIONS 10
#define NUM_ITTERATIONS_DOUBLE ((double)NUM_ITTERATIONS)

int Num_hrect_prunes;
int Num_ball_prunes;

ivec *mk_random_subset_of_ints(int n,int subsize);

dyv *my_metric;

void mrkd_slow_nearest(dyv *metric,ivec *rows,dym *x,dyv *q,
		       int *r_row,double *r_dsqd,int not_me_row,ivec *visited)
{
  int i;
  for ( i = 0 ; i < ivec_size(rows) ; i++ )
  {
    int row = ivec_ref(rows,i);
    double dsqd = row_dyv_metric_dsqd(x,metric,row,q);
    if ( dsqd < *r_dsqd )
    {
      if ( row != not_me_row )
      {
        *r_row = row;
        *r_dsqd = dsqd;
      }
    }
    if ( visited != NULL )
      add_to_ivec(visited,row);
  }
}

void mrkd_knode_nearest(dyv *metric,knode *kn,dym *x,dyv *q,
			int *r_row,double *r_dsqd,double kn_q_dsqd,
			int not_me_row,ivec *visited)
{
  if ( kn_q_dsqd >= *r_dsqd )
  {
    Num_hrect_prunes++;
  }
  else if ( knode_is_leaf(kn) )
    mrkd_slow_nearest(metric,kn->rows,x,q,r_row,r_dsqd,not_me_row,visited);
  else
  {
    double kn_left_dsqd = hrect_dyv_min_metric_dsqd(metric,kn->left->hr,q);
    double kn_right_dsqd = hrect_dyv_min_metric_dsqd(metric,kn->right->hr,q);
    bool left_closer = kn_left_dsqd < kn_right_dsqd;
    double kn_closer_dsqd = (left_closer) ? kn_left_dsqd : kn_right_dsqd;
    knode *kn_closer = (left_closer) ? kn->left : kn->right;
    double kn_further_dsqd = (!left_closer) ? kn_left_dsqd : kn_right_dsqd;
    knode *kn_further = (!left_closer) ? kn->left : kn->right;
    mrkd_knode_nearest(metric,kn_closer,x,q,r_row,r_dsqd,kn_closer_dsqd,
                       not_me_row,visited);
    mrkd_knode_nearest(metric,kn_further,x,q,r_row,r_dsqd,kn_further_dsqd,
                       not_me_row,visited);
  }
}

int mrkd_nearest_neighbor(mrkd *mr,dym *x,dyv *q,int not_me_row,
			  ivec **r_visited_rows)
{
  //dyv *metric = mrkd_metric(mr);
  double max_possible_dsqd = 
    2 * hrect_dyv_max_metric_dsqd(my_metric,mr->root->hr,q);
  double dsqd = max_possible_dsqd;
  double root_q_dsqd = hrect_dyv_min_metric_dsqd(my_metric,mr->root->hr,q);
  int row = -1;
  if (r_visited_rows != NULL) 
    *r_visited_rows = mk_ivec(0); 

  mrkd_knode_nearest(my_metric,mr->root,x,q,&row,&dsqd,root_q_dsqd,not_me_row,
                     *r_visited_rows);
  return row;
}

void mrkd_slow_ortho_range_search(ivec *rows,dym *x,
				  dym *bounds,int *count)
{
  int i,d, prune_flag;

  for ( i = 0 ; i < ivec_size(rows) ; i++ )   {
    int row = ivec_ref(rows,i);
    prune_flag = SUBSUME;
    
    for(d = 0; d < dym_cols(x); d++) {
      // determine which one of the two cases we have: EXCLUDE, SUBSUME
      
      // first the EXCLUDE case: when dist is above the upper bound distance
      // of this dimension, or dist is below the lower bound distance of
      // this dimension
      if(dym_ref(x,row,d) > dym_ref(bounds, d, 1) || 
	 dym_ref(x,row,d) < dym_ref(bounds, d, 0)) {
	prune_flag = EXCLUDE;
	break;
      }
    }
    
    if(prune_flag == SUBSUME) {
      *count = (*count) + 1;
    }
  }
}

void mrkd_knode_ortho_range_search(knode *kn, dym *x, dym *bounds,
				   int *counts, int start_dim)
{
  int d;
  int prune_flag = SUBSUME;

  // loop over each dimension to determine inclusion/exclusion by determining
  // the lower and the upper bound distance per each dimension for the
  // given reference node, kn
  for(d = start_dim; d < dym_cols(x); d++) {

    // determine which one of the three cases we have: EXCLUDE, SUBSUME, or
    // INCONCLUSIVE.

    // first the EXCLUDE case: when mindist is above the upper bound distance
    // of this dimension, or maxdist is below the lower bound distance of
    // this dimension
    if(dyv_ref(kn->hr->lo,d) > dym_ref(bounds, d, 1) || 
       dyv_ref(kn->hr->hi,d) < dym_ref(bounds, d, 0)) {
      Num_hrect_prunes++;
      return;
    }
    // otherwise, check for SUBSUME case
    else if(dym_ref(bounds, d, 0) <= dyv_ref(kn->hr->lo,d) && 
	    dyv_ref(kn->hr->hi,d) <= dym_ref(bounds, d, 1)) {
    }
    // if any dimension turns out to be inconclusive, then break.
    else {
      start_dim = d;
      prune_flag = INCONCLUSIVE;
      break;
    }
  }
  
  // in case of subsume, then add all points owned by this node to
  // candidates
  if ( prune_flag == SUBSUME )  {
    int r;
    /*
    for(r = 0; r < kn->num_points; r++) {
      add_to_ivec(candidates, ivec_ref(kn->rows, r));
    }
    */
    (*count) = (*count) + kn->num_points;
    Num_hrect_prunes++;
    return;
  }
  else if ( knode_is_leaf(kn) ) {
    mrkd_slow_ortho_range_search(kn->rows,x,bounds,&counts);
  }
  else   {
    mrkd_knode_ortho_range_search(kn->left,x,bounds,counts, start_dim);
    mrkd_knode_ortho_range_search(kn->right,x,bounds,counts, start_dim);
  }
}

/** @brief Performs the orthogonal range search for a single query point
 *
 * @param mr
 * @param x
 * @param q
 * @param not_me_row
 * @param bounds
 * @param candidates
 *
 * @return Void.
 */
void mrkd_ortho_range_search(mrkd *mr,dym *x,dym *bounds, int *counts)
{
  mrkd_knode_ortho_range_search(mr->root,x,bounds,candidates, counts);
}

dym *mk_dym_from_filename_improved(char *filename,int argc,char *argv[])
{
  datset *ds = ds_load_with_options_simple(filename,argc,argv);
  dym *m = mk_dym_from_datset(ds);
  free_datset(ds);
  return m;
}

mrkd *mk_mrkd_from_args(int argc,char *argv[],dym **r_data)
{
  char *filename = string_from_args("in",argc,argv,"default.fds");
  mrkd *dmrkd; 
  double time1, time2, tot_time = 0;  
  dym *data = mk_dym_from_filename_improved(filename,argc,argv);
  mrpars *dmrpars = mk_default_mrpars_for_data(data); 
  int rmin = int_from_args("rmin",argc,argv,2);

  dmrpars->has_sums = 0; dmrpars->has_xxts = 0; dmrpars->rmin = rmin;
  dmrpars->has_sum_sqd_mags = 0; dmrpars->has_sq_sum_lengths = 0;
  dmrpars->has_sum_quad_mags = 0; dmrpars->has_scaled_sums = 0;
  dmrpars->has_points_throughout = 0;
  dmrpars->has_points = hpAllNodes;
  constant_dyv(dmrpars->metric,1.0);  Verbosity = -1;

  time1 = get_time();
  dmrkd = mk_mrkd(dmrpars,data); 
  time2 = get_time(); tot_time += (time2 - time1);
  printf("%f sec. elapsed for building kd-tree\n",tot_time); 

  free_mrpars(dmrpars); 
  *r_data = data;

  //ag_window_shape(winsize,winsize);
  //
  //if ( dyv_array_num_dims(sps) == 2  )
  //{
  //  ag_on("");
  //  dyv_array_set_ag_frame(sps);
  //  if ( dyv_array_size(sps) < 5000 )
  //    draw_batree(bat,sps);
  //  ag_off();
  //  wait_for_key();
  //}

  return dmrkd;
}

mrkd *mk_mrkd_from_dym(dym *data, int rmin)
{

  mrkd *dmrkd;
  double time1, time2;

  mrpars *dmrpars = mk_default_mrpars_for_data(data); 

  dmrpars->has_sums = 0; dmrpars->has_xxts = 0; dmrpars->rmin = rmin;
  dmrpars->has_sum_sqd_mags = 0; dmrpars->has_sq_sum_lengths = 0;
  dmrpars->has_sum_quad_mags = 0; dmrpars->has_scaled_sums = 0;
  dmrpars->has_points_throughout = 1;
  constant_dyv(dmrpars->metric,1.0);  Verbosity = -1;

  time1 = get_time();
#ifdef TIMING
  mrkd *dmrkd_reps[9];
  int reps;
    for(reps=0; reps<9; reps++) {
      dmrkd_reps[reps] = mk_mrkd(dmrpars,data); 
    }
#endif

  dmrkd = mk_mrkd(dmrpars,data);
  time2 = get_time();

#ifdef TIMING
  double build_time = (time2-time1)/10.0;
#else
  double build_time = (time2-time1)/10.0;
#endif

  add_to_dyv(build_times, build_time);

  printf("%f sec. elapsed for building kd-tree\n",build_time);

  free_mrpars(dmrpars); 

#ifdef TIMING
  for(reps=0; reps<9; reps++)
    free_mrkd(dmrkd_reps[reps]);
#endif

  return dmrkd;
}

bool validate_range(dym *range)
{
  int d;
  for(d = 0; d < dym_rows(range); d++) {
    if(dym_ref(range, d, 0) > dym_ref(range, d, 1))
      return FALSE;
  }
  return TRUE;
}

int main(int argc,char *argv[])
{
  char *data_filename = string_from_args("-data",argc,argv,"data.ds");
  char *range_filename = string_from_args("-range",argc,argv,"range.ds");
  dym *data = mk_dym_from_filename_improved(data_filename,argc,argv);
  dym *range = mk_dym_from_filename_improved(range_filename,argc,argv);

  alloc_timing_dyvs();
  Num_pt_dists = 0; Num_hr_dists = 0;Num_ball_prunes = 0; Num_hrect_prunes = 0;

  mrkd *dmrkd = mk_mrkd_from_dym(data, int_from_args("rmin",argc,argv,30));

  // validate integrity of the range dataset
  if(!validate_range(range)) {
    printf("Invalid range values. Needs to have lo <= hi!\n");
    exit(0);
  }

  double time1 = get_time();

  // stores the row numbers of the data points that are within the range
  int candidates = 0;

  // stores the rwo numbers of the data points found by the exhaustive method
  int slow_candidates = 0;
  
  mrkd_ortho_range_search(dmrkd,data,range,&candidates);

  printf("%g seconds elapsed in range searching for the tree-based...\n", 
	 get_time() - time1);
  ivec_sort(candidates, candidates);
  printf("Pruned %d times...\n", Num_hrect_prunes);
  fprintf_dym(stdout, "Searching in range [lo,hi] for each dimension: ", 
	      range, "\n");
  //  fprintf_ivec(stdout, "Candidate row numbers: ", candidates, "\n");

  printf("Verifying with the exhaustive method...\n");
  time1 = get_time();

  printf("Searching %d rows...\n", ivec_size(dmrkd->root->rows));
  mrkd_slow_ortho_range_search(dmrkd->root->rows,data,range,slow_candidates);

  printf("Exhaustive method took %g seconds...\n", get_time() - time1);
  ivec_sort(slow_candidates, &slow_candidates);
  //fprintf_ivec(stdout, "Exhaustive method found: ", slow_candidates, "\n");

  if(slow_candidates == candidates) {
    printf("Both methods found equal candidates...\n");
  }
  else {
    printf("Both methods outputed different candidates...\n");
  }

  free_ivec(slow_candidates);
  free_dym(data); free_dym(range);
  free_ivec(candidates);
  free_timing_dyvs();
  return 0;
}
