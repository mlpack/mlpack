#include <vector>

#include "fastlib/fastlib.h"
#include "ttree.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>



// void TTree::SampleSetWithReplacement_(std::vector<double> &dim_vals,
// 				      std::vector<double> *bs_dim_vals) {
//   size_t size = dim_vals.size();
//   srand(time(NULL));
//   for (size_t i = 0; i < size; i++) {
//     // pick a random number between [0, size)
//     size_t ind = rand() % size;
//     bs_dim_vals->push_back(dim_vals[ind]);
//   }
//   // sorting the bootstrap vector
//   std::sort(bs_dim_vals->begin(), bs_dim_vals->end());
//   return;
// }

double TTree::ComputeNodeError_(Vector &y, double *gamma) {

  // possibly change it from max to a more 
  // probabilistic version

  double max = 0, sum = 0;
  for (size_t i = 0; i < y.length(); i++) {
    if (y[i] > max)
      max = y[i];

    sum += y[i];
  }

  double error = y.length() * max - sum;

  *gamma = max;

  return error;
}

bool TTree::FindSplit_(Matrix& x, Vector& y, size_t total_n,
		       size_t *split_dim, double *split_val,
		       double *left_error, double *right_error,
		       double *gam_l, double *gam_r) {

  DEBUG_ASSERT(x.n_cols() == stop_ - start_);
  DEBUG_ASSERT(x.n_rows() == max_vals_.size());
  DEBUG_ASSERT(x.n_rows() == min_vals_.size());

  DEBUG_ASSERT(x.n_cols() == y.length());

  size_t n_t = x.n_cols();
  double min_error = error_;
  bool some_split_found = false;
  size_t point_mass_in_dim = 0;
  size_t min_leaf_size = fx_param_int(module_, "min_leaf_size", 1),
    max_leaf_size = fx_param_int(module_, "max_leaf_size", 10);

  // printf("In FindSplit %lg\n", error_);fflush(NULL);

  // loop through each dimension
  for (size_t dim = 0; dim < max_vals_.size(); dim++) {
    // have to deal with REAL, INTEGER, NOMINAL x
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
      double min_dim_error = min_error,
	temp_lval = 0.0, temp_rval = 0.0;
      double dim_split_val = DBL_MAX;

      double gam_ld = -1, gam_rd = -1;

//       double range = 1.0;
//       for (size_t i = 0; i < max_vals_.size(); i++) {
// 	// 	if (dim_type_[i] == REAL) {
// 	if (max_vals_[i] -min_vals_[i] > 0.0 && i != dim) {
// 	  range *= max_vals_[i] - min_vals_[i];
// 	}
// 	// 	}
//       }
//       // printf("range: %Lg\n", range); fflush(NULL);


      // get the values for the dimension
      std::vector<std::pair<double, double> > dim_val_vec;
      for (size_t i = 0; i < n_t; i++) {
	std::pair<double, double> x_y (x.get(dim, i), y[i]);
	dim_val_vec.push_back(x_y);
      }
      // sort the values in ascending order
      // HAVE TO CHECK: if the corresponding y-vals
      // move around with the x-vals
      std::sort(dim_val_vec.begin(), dim_val_vec.end());

      Vector y_sorted_x;
      y_sorted_x.Init(n_t);

      for (size_t i = 0; i < n_t; i++) 
	y_sorted_x[i] = dim_val_vec[i].second;

      // get ready to go through the sorted list and compute error

      DEBUG_ASSERT((size_t) dim_val_vec.size() > max_leaf_size);
      // enforcing the leaves to have a minimum of MIN_LEAF_SIZE 
      // number of points to avoid spikes

      // one way of doing it is only considering splits resulting
      // in sizes > MIN_LEAF_SIZE
      size_t left_child_size = min_leaf_size; // , right_child_size;

      // finding the best split for this dimension
      // need to figure out why there are spikes if 
      // this min_leaf_size is enforced here
      std::vector<std::pair<double, double> >::iterator it;
      for (it = dim_val_vec.begin() + min_leaf_size -1;
	   it < dim_val_vec.end() - min_leaf_size;
	   it++, left_child_size++) {
	
	double lsplit = it->first, rsplit = (it+1)->first;

	// make sure that these are distinct values
	if (lsplit < rsplit) {
	// 	if (dim_type_[dim] == REAL) {
	//  split = lsplit;
	// 	} else {
	// 	  split = *it;
	// 	}
// 	  if (split - min > 0.0 && max - split > 0.0) {
// 	  printf("potential split %lg on dim %zud\n",
// 		 split, dim);fflush(NULL);
	  double gl, gr, el, er;

	  Vector y_l, y_r;
	  y_sorted_x.MakeSubvector(0, left_child_size, &y_l);
	  y_sorted_x.MakeSubvector(left_child_size,
				   n_t - left_child_size, 
				   &y_r);

	  el = ComputeNodeError_(y_l, &gl);
	  er = ComputeNodeError_(y_r, &gr);


	  //if (temp_l + temp_r <= min_dim_error) {
	  // why not just less than
	  if (el + er < min_dim_error) {
	    min_dim_error = el + er;
	    temp_lval = el;
	    temp_rval = er;
	    dim_split_val = lsplit;
	    dim_split_found = true;

	    gam_ld = gl;
	    gam_rd = gr;
	  } // end if
	} // end if lsplit < rsplit 
      } // end for
      
      dim_val_vec.clear();

      if ((min_dim_error < min_error) && dim_split_found) {
	min_error = min_dim_error;
	*split_dim = dim;
	*split_val = dim_split_val;
	*left_error = temp_lval;
	*right_error = temp_rval;
	some_split_found = true;

	*gam_l = gam_ld;
	*gam_r = gam_rd;
      } // end if better split found in this dim
    } else {
      point_mass_in_dim++;
    } // end if
  } // end for each dimension

  // This might occur when you have many instances of the
  // same point in the dataset (point mass). Have to figure out a way to
  // deal with it
  // DEBUG_ASSERT(point_mass_in_dim != max_vals_.size());

  // printf("%zud point mass dims.\n", point_mass_in_dim);fflush(NULL);

  return some_split_found;
//   DEBUG_ASSERT_MSG(some_split_found,
// 		   "Weird - no split found"
// 		   " %zud points, %zud %lg %zud\n",
// 		   x.n_cols(), total_n, min_error, point_mass_in_dim);
} // end FindSplit_



// FIX THIS FUNCTCLIN
void TTree::SplitData_(Matrix& x, Vector& y, 
		       size_t split_dim, double split_val,
		       Matrix *x_l, Vector *y_l,
		       Matrix *x_r, Vector *y_r,
		       ArrayList<size_t> *old_from_new) {
// 		       double *split_val,
// 		       double *lsplit_val, double *rsplit_val) {

  // get the values for the split dim
//   std::vector<double> dim_val_vec;
//   for (size_t i = 0; i < x.n_cols(); i++) {
//     dim_val_vec.push_back(x.get(split_dim, i));
//   } // end for

//     // sort the values
//   std::sort(dim_val_vec.begin(), dim_val_vec.end());

//   *lsplit_val =  *(dim_val_vec.begin()+split_ind);
//   *rsplit_val =  *(dim_val_vec.begin() + split_ind + 1);
//   *split_val = (*lsplit_val + *rsplit_val) / 2 ;

  DEBUG_ASSERT(x.n_cols() == stop_ - start_);

  size_t left_size = 0;
  for(size_t i = 0; i < x.n_cols(); i++) {
    if (x.get(split_dim, i) <= split_val)
      left_size++;
  }

  x_l->Init(x.n_rows(), left_size);
  x_r->Init(x.n_rows(), x.n_cols() - left_size);

  y_l->Init(left_size);
  y_r->Init(x.n_cols() - left_size);

  size_t j = 0, k = 0;

  // make sure to swap indices in old_to_new thing (firstly 
  // do I need it anymore)

  ArrayList<size_t> temp_swap_list;
  temp_swap_list.Init(stop_ - start_);

  for (size_t i = 0; i < x.n_cols(); i++) {
    if (x.get(split_dim, i) <= split_val) {

      // add this to x_l
      Vector l_vec;
      x_l->MakeColumnVector(j, &l_vec);
      l_vec.CopyValues(x.GetColumnPtr(i));

      (*y_l)[j] = y[i];

      temp_swap_list[j] = (*old_from_new)[start_ + i];
      j++;
    } else {

      // add this is x_r
      Vector r_vec;
      x_r->MakeColumnVector(k, &r_vec);
      r_vec.CopyValues(x.GetColumnPtr(i));

      (*y_r)[k] = y[i];

      temp_swap_list[left_size + k]
	= (*old_from_new)[start_ + i];
      k++;
    }
  } // end for

  DEBUG_ASSERT(j == left_size);
  DEBUG_ASSERT(k == x.n_cols() - left_size);


  for (size_t i = 0; i < x.n_cols(); i++) 
    (*old_from_new)[start_ + i] 
      = temp_swap_list[i];

//   size_t i = split_ind, j = split_ind + 1;
//   while ( i > -1 && j < x.n_cols()) {
//     while (i > -1 && x.get(split_dim, i) < *split_val)
//       i--;

//     while (j < x.n_cols() && x.get(split_dim, j) > *split_val)
//       j++;

//     // swapping values
//     if (i > -1 && j < x.n_cols()) {
//       Vector vec1, vec2;
//       x.MakeColumnVector(i, &vec1);
//       x.MakeColumnVector(j, &vec2);
//       vec1.SwapValues(&vec2);

//       size_t temp = (*old_from_new)[start_ + i];
//       (*old_from_new)[start_ +i] = (*old_from_new)[start_ +j];
//       (*old_from_new)[start_ +j] = temp;

//       i--;
//       j++;
//     }
//   }

//   DEBUG_ASSERT_MSG((i==-1)||(j==x.n_cols()),
// 		   "i = %zud, j = %zud N = %zud",
// 		   i, j, x.n_cols());

//   x.MakeColumnSlice(0, split_ind+1, x_l);
//   x.MakeColumnSlice(split_ind+1, x.n_cols()-split_ind-1, x_r);

} // end SplitData_

void TTree::GetMaxMinVals_(Matrix& x, ArrayList<double> *max_vals,
			   ArrayList<double> *min_vals) {
  max_vals->Init(x.n_rows());
  min_vals->Init(x.n_rows());
  Matrix temp_d;
  la::TransposeInit(x, &temp_d);
  for (size_t i = 0; i < temp_d.n_cols(); i++) {
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
