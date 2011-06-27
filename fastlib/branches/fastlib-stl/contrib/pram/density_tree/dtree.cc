#include <vector>

#include "fastlib/fastlib.h"
#include "dtree.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>



void DTree::SampleSetWithReplacement_(std::vector<double> &dim_vals,
				      std::vector<double> *bs_dim_vals) {
  index_t size = dim_vals.size();
  srand(time(NULL));
  for (index_t i = 0; i < size; i++) {
    // pick a random number between [0, size)
    index_t ind = rand() % size;
    bs_dim_vals->push_back(dim_vals[ind]);
  }
  // sorting the bootstrap vector
  std::sort(bs_dim_vals->begin(), bs_dim_vals->end());
  return;
}

long double DTree::ComputeNodeError_(index_t total_points) {
  long double range = 1.0;
  index_t node_size = stop_ - start_;

  DEBUG_ASSERT(max_vals_.size() == min_vals_.size());
  for (index_t i = 0; i < max_vals_.size(); i++) {
    // if no variation in a dimension, we do not care
    // about that dimension
    if (max_vals_[i] - min_vals_[i] > 0.0) {
      range *= max_vals_[i] - min_vals_[i];
      // std::cout << "r:" << range << "\n";fflush(NULL);
    }
  }

//   std::cout << "First range:" << range << "\n";fflush(NULL);

  long double error = -1.0 * node_size * node_size
    / (range * total_points * total_points);

  return error;
}

bool DTree::FindSplit_(Matrix& data, index_t total_n,
		       index_t *split_dim, index_t *split_ind,
		       long double *left_error, long double *right_error) {

  DEBUG_ASSERT(data.n_cols() == stop_ - start_);
  DEBUG_ASSERT(data.n_rows() == max_vals_.size());
  DEBUG_ASSERT(data.n_rows() == min_vals_.size());
  index_t n_t = data.n_cols();
  long double min_error = error_;
  bool some_split_found = false;
  index_t point_mass_in_dim = 0;
  index_t min_leaf_size = fx_param_int(module_, "min_leaf_size", 15),
    max_leaf_size = fx_param_int(module_, "max_leaf_size", 30);

  // printf("In FindSplit %Lg\n", error_);fflush(NULL);

  // loop through each dimension
  for (index_t dim = 0; dim < max_vals_.size(); dim++) {
    // have to deal with REAL, INTEGER, NOMINAL data
    // differently so have to think of how to do that.
    // Till them experiment with comparisons with kde
    // and also scalability experiments and visualization.

    //       if (dim_type_[dim] == NOMINAL) {

    //       } else {
    double min = min_vals_[dim], max = max_vals_[dim];

    // checking if there is any scope of splitting in this dim
    if (max - min > 0.0) {
      // initializing all the stuff for this dimension
      bool dim_split_found = false;
      long double min_dim_error = min_error,
	temp_lval = 0.0, temp_rval = 0.0;
      index_t dim_split_ind = -1;

      long double range = 1.0;
      for (index_t i = 0; i < max_vals_.size(); i++) {
	// 	if (dim_type_[i] == REAL) {
	if (max_vals_[i] -min_vals_[i] > 0.0 && i != dim) {
	  range *= max_vals_[i] - min_vals_[i];
	}
	// 	}
      }
      // printf("range: %Lg\n", range); fflush(NULL);


      // get the values for the dimension
      std::vector<double> dim_val_vec;
      for (index_t i = 0; i < n_t; i++) {
	dim_val_vec.push_back (data.get(dim, i));
      }
      // sort the values in ascending order
      std::sort(dim_val_vec.begin(), dim_val_vec.end());

      // get ready to go through the sorted list and compute error

      DEBUG_ASSERT(dim_val_vec.size() > max_leaf_size);
      // enforcing the leaves to have a minimum of MIN_LEAF_SIZE 
      // number of points to avoid spikes

      // one way of doing it is only considering splits resulting
      // in sizes > MIN_LEAF_SIZE
      index_t left_child_size = min_leaf_size -1, right_child_size;

      // finding the best split for this dimension
      // need to figure out why there are spikes if 
      // this min_leaf_size is enforced here
      for (std::vector<double>::iterator it
	     = dim_val_vec.begin() + min_leaf_size -1;
	   it < dim_val_vec.end() - min_leaf_size -1;
	   it++, left_child_size++) {
	double split, lsplit = *it, rsplit = *(it+1);

	if (lsplit < rsplit) {
	  // 	if (dim_type_[dim] == REAL) {
	  split = (*it + *(it+1))/2;
	  // 	} else {
	  // 	  split = *it;
	  // 	}
	  if (split - min > 0.0 && max - split > 0.0) {
	    // 	  printf("potential split %lg on dim %"LI"d\n",
	    // 		 split, dim);fflush(NULL);
	    long double temp_l
	      = -1.0 * ((double)(left_child_size+1)/(double)total_n)
	      * ((double)(left_child_size+1)/(double)total_n)
	      / (range * (split - min));
	    DEBUG_ASSERT(-1.0*temp_l < DBL_MAX);
	    right_child_size = n_t - left_child_size - 1;
	    DEBUG_ASSERT(right_child_size >= min_leaf_size);
	    
	    long double temp_r
	      = -1.0 * ((double)right_child_size/(double)total_n)
	      * ((double)right_child_size/(double)total_n)
	      / (range * (max - split));
	    DEBUG_ASSERT(-1.0*temp_r < DBL_MAX);

	    //if (temp_l + temp_r <= min_dim_error) {
	    // why not just less than
	    if (temp_l + temp_r < min_dim_error) {
	      //	      printf(".");
	      min_dim_error = temp_l + temp_r;
	      temp_lval = temp_l;
	      temp_rval = temp_r;
	      dim_split_ind = left_child_size;
	      dim_split_found = true;
	    } // end if
	  } // end if
	} // end if lsplit < rsplit
      } // end for
      
      dim_val_vec.clear();

      if ((min_dim_error < min_error) && dim_split_found) {
	min_error = min_dim_error;
	*split_dim = dim;
	*split_ind = dim_split_ind;
	*left_error = temp_lval;
	*right_error = temp_rval;
	some_split_found = true;
      } // end if better split found in this dim
    } else {
      point_mass_in_dim++;
    } // end if
  } // end for each dimension

  // This might occur when you have many instances of the
  // same point in the dataset (point mass). Have to figure out a way to
  // deal with it
  // DEBUG_ASSERT(point_mass_in_dim != max_vals_.size());

  // printf("%"LI"d point mass dims.\n", point_mass_in_dim);fflush(NULL);

  return some_split_found;
//   DEBUG_ASSERT_MSG(some_split_found,
// 		   "Weird - no split found"
// 		   " %"LI"d points, %"LI"d %lg %"LI"d\n",
// 		   data.n_cols(), total_n, min_error, point_mass_in_dim);
} // end FindSplit_


void DTree::SplitData_(Matrix& data, index_t split_dim, index_t split_ind,
		       Matrix *data_l, Matrix *data_r, 
		       ArrayList<index_t> *old_from_new, 
		       double *split_val,
		       double *lsplit_val, double *rsplit_val) {

  // get the values for the split dim
  std::vector<double> dim_val_vec;
  for (index_t i = 0; i < data.n_cols(); i++) {
    dim_val_vec.push_back(data.get(split_dim, i));
  } // end for

    // sort the values
  std::sort(dim_val_vec.begin(), dim_val_vec.end());

  *lsplit_val =  *(dim_val_vec.begin()+split_ind);
  *rsplit_val =  *(dim_val_vec.begin() + split_ind + 1);
  *split_val = (*lsplit_val + *rsplit_val) / 2 ;

  index_t i = split_ind, j = split_ind + 1;
  while ( i > -1 && j < data.n_cols()) {
    while (i > -1 && data.get(split_dim, i) < *split_val)
      i--;

    while (j < data.n_cols() && data.get(split_dim, j) > *split_val)
      j++;

    // swapping values
    if (i > -1 && j < data.n_cols()) {
      Vector vec1, vec2;
      data.MakeColumnVector(i, &vec1);
      data.MakeColumnVector(j, &vec2);
      vec1.SwapValues(&vec2);

      index_t temp = (*old_from_new)[start_ + i];
      (*old_from_new)[start_ +i] = (*old_from_new)[start_ +j];
      (*old_from_new)[start_ +j] = temp;

      i--;
      j++;
    }
  }

  DEBUG_ASSERT_MSG((i==-1)||(j==data.n_cols()),
		   "i = %"LI"d, j = %"LI"d N = %"LI"d",
		   i, j, data.n_cols());

  data.MakeColumnSlice(0, split_ind+1, data_l);
  data.MakeColumnSlice(split_ind+1, data.n_cols()-split_ind-1, data_r);

} // end SplitData_

void DTree::GetMaxMinVals_(Matrix& data, ArrayList<double> *max_vals,
			   ArrayList<double> *min_vals) {
  max_vals->Init(data.n_rows());
  min_vals->Init(data.n_rows());
  Matrix temp_d;
  la::TransposeInit(data, &temp_d);
  for (index_t i = 0; i < temp_d.n_cols(); i++) {
    // if (dim_type[i] != NOMINAL) {
    Vector dim_vals;
    temp_d.MakeColumnVector(i, &dim_vals);
    std::vector<double> dim_vals_vec(dim_vals.ptr(),
				     dim_vals.ptr() + temp_d.n_rows());

    sort(dim_vals_vec.begin(), dim_vals_vec.end());
    (*min_vals)[i] = *(dim_vals_vec.begin());
    (*max_vals)[i] = *(dim_vals_vec.end() -1);
    // }
  }

} // end GetMaxMinVals_



//       // on finding the best split here bootstrap to see
//       // if the reduction in error : [error - (error_l+error_r)]
//       // is actually significant.
//       if (fx_param_bool(module_, "do_bootstrap", false)
//                      && dim_split_found) {
// 	// NOTIFY("Bootstrapping");
// 	// the original value is the min_dim_error
// 	double mde_b_sq = 0.0, mde_b = 0.0;
// 	index_t successful_b_steps = 0;

// 	std::vector<double> rie_vec;

// 	for (index_t b = 0; b < fx_param_int(module_, "b", 20); b++) {
// 	  // do each of the bootstrap step

// 	  // picking up a bootstrap sample
// 	  std::vector<double> *bs_dim_val_vec
// 	    = new std::vector<double>(); 
// 	  SampleSetWithReplacement_(dim_val_vec, bs_dim_val_vec);
// 	  left_child_size = min_leaf_size -1;
// 	  right_child_size = 0;
 
// 	  // need to decide here what the min and max values
// 	  // should be - 
// 	  // Should it be the original one for this dimension
// 	  // or the min and max of this bootstrap sample
// 	  // For now I am going to use original min max since
// 	  // min maxs are not actual points but rather midway
// 	  // between.
// 	  // double min_b = (*bs_dim_val_vec)[0], 
// 	  //  max_b = (*bs_dim_val_vec)[bs_dim_val_vec->size() -1];

// 	  // finding the best split for this new set of points
// 	  // in this dimension
// 	  double min_dim_error_b = min_error;
// 	  bool dim_split_found_b = false;
// 	  for (std::vector<double>::iterator it
// 		 = bs_dim_val_vec->begin() + min_leaf_size -1;
// 	       it < bs_dim_val_vec->end() - min_leaf_size -1;
// 	       it++, left_child_size++) {
// 	    double split;
// 	    // 	if (dim_type_[dim] == REAL) {
// 	    split = (*it + *(it+1))/2;
// 	    // 	} else {
// 	    // 	  split = *it;
// 	    // 	}
// 	    if (split - min > 0.0 && max - split > 0.0) {
// 	      // if (split - min_b > 0.0 && max_b - split > 0.0) {
// 	      double temp_l
// 		= -1.0 * ((double)(left_child_size+1)/(double)total_n)
// 		* ((double)(left_child_size+1)/(double)total_n)
// 		/ (range * (split - min));
// 	      // / (range * (split - min_b));
// 	      DEBUG_ASSERT(-1.0*temp_l < DBL_MAX);
// 	      right_child_size = n_t - left_child_size - 1;
// 	      DEBUG_ASSERT(right_child_size >= min_leaf_size);
	    
// 	      double temp_r
// 		= -1.0 * ((double)right_child_size/(double)total_n)
// 		* ((double)right_child_size/(double)total_n)
// 		/ (range * (max - split));
// 	      // / (range * (max_b - split));
// 	      DEBUG_ASSERT(-1.0*temp_r < DBL_MAX);

// 	      //if (temp_l + temp_r <= min_dim_error) {
// 	      // why not just less than
// 	      if (temp_l + temp_r < min_dim_error_b) {
// 		//	      printf(".");
// 		min_dim_error_b = temp_l + temp_r;
// 		dim_split_found_b = true;
// 	      } // end if
// 	    } // end if
// 	  } // end for

// 	    // if a split is found, update the bootstrap stuff
// 	  if (dim_split_found_b) {
// 	    double red_in_error = min_dim_error_b;
// 	    // DEBUG_ASSERT(red_in_error > 0);
// 	    mde_b += red_in_error;
// 	    mde_b_sq += (red_in_error * red_in_error);
// 	    successful_b_steps++;
// 	    rie_vec.push_back(red_in_error);
// 	  }
// 	} // end of the bootstrap steps

// 	  // computing the standard error
// 	// 	  double se_sq =
// 	// 	    (mde_b_sq - (mde_b * mde_b / (double) successful_b_steps)) 
// 	// 	    / (double) successful_b_steps;


// 	// alternate computation of se
// 	double b_mean = mde_b / (double) successful_b_steps;
// 	double alt_se = 0.0;
// 	for (index_t i = 0; i < rie_vec.size(); i++) {
// 	  double diff = rie_vec[i] - b_mean;
// 	  alt_se += (diff * diff);
// 	}
// 	alt_se /= (double) successful_b_steps;
// 	double z_stat = (b_mean - min_dim_error) / sqrt(alt_se);


// 	if (z_stat < 1.96 && z_stat > -1.96) {
// 	  // the split is significant
// 	  // NOTIFY("Significant split");
// 	  dim_split_found = true;
// 	} else {
// 	  // the split is insignificant, hence return from here
// 	  // printf("High variance split...%lg\n", z_stat);
// 	  // printf("zstat = %lg, steps = %"LI"d,"
// // 		 " alt_se = %lg, (b_mean - mde) = %lg\n",
// // 		 z_stat, successful_b_steps, sqrt(alt_se),
// // 		 (b_mean - min_dim_error));
// 	  dim_split_found = false;
// 	}
//       } // if do_bootstrap
