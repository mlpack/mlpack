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

bool validate_range(dym *range)
{
  int d;
  for(d = 0; d < dym_rows(range); d++) {
    if(dym_ref(range, d, 0) > dym_ref(range, d, 1))
      return FALSE;
  }
  return TRUE;
}
