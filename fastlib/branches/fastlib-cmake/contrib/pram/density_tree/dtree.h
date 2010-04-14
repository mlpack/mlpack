 /**
 * @file dtree.h
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Density Tree class
 *
 */

#ifndef DTREE_H
#define DTREE_H

#include <vector>

#include "fastlib/fastlib.h"
#define LEAF_SIZE 5
#define MIN_LEAF_SIZE 1

class DTree{
  //k;
 
  ////////////////////// Member Variables /////////////////////////////////////
  
private:
  // The indices in the complete set of points
  // (after all forms of swapping in the 
  // old_from_new array)
  int start_, stop_;
  
  // The split dim
  index_t split_dim_;

//   // split dim types
//   ArrayList<enum> dim_types_;

//   // The split value for non-numeric data
//   string split_val_;

  // The split val on that dim
  double split_value_;

  // error of the node
  double error_;

  // error of the leaves of the subtree
  double subtree_leaves_error_;

  // number of leaves of the subtree
  index_t subtree_leaves_;

  // ratio of number of points in the node
  double ratio_;

  // since we are using uniform density, we need
  // the max and min of every dimension for every node
  ArrayList<double> max_vals_;
  ArrayList<double> min_vals_;

  // The children
  DTree *left_;
  DTree *right_;

  ////////////////////// Constructors /////////////////////////////////////////
  
  FORBID_ACCIDENTAL_COPIES(DTree);

public: 

  DTree() {
    left_ = NULL;
    right_ = NULL;
  }

  ~DTree() {
    if (left_ != NULL){
      delete left_;
    }
    if (right_ != NULL){
      delete right_;
    }
  }

  ////////////////////// Getters and Setters //////////////////////////////////
  index_t start() { return start_; }

  index_t stop() { return stop_; }

  index_t split_dim() { return split_dim_; }

  double split_value() { return split_value_; }

  double error() { return error_; }

  double subtree_leaves_error() { return subtree_leaves_error_; }

  index_t subtree_leaves() { return subtree_leaves_; }

  DTree* left_child() { return left_; }
  DTree* right_child() { return right_; }

  ////////////////////// Private Functions ////////////////////////////////////
 private:

  double ComputeNodeError_(index_t total_points) {
    double range = 1.0;
    index_t node_size = stop_ - start_;

    DEBUG_ASSERT(max_vals_.size() == min_vals_.size());
    for (index_t i = 0; i < max_vals_.size(); i++) {
      // if no variation in a dimension, we do not care
      // about that dimension
      if (max_vals_[i] - min_vals_[i] > 0.0) {
	range *= max_vals_[i] - min_vals_[i];
      }
    }

    double error = -1.0 * node_size * node_size
      / (range * total_points * total_points);

    return error;
  }

  bool FindSplit_(Matrix& data, index_t total_n,
		  index_t *split_dim, index_t *split_ind,
		  double *left_error, double *right_error) {

    DEBUG_ASSERT(data.n_cols() == stop_ - start_);
    DEBUG_ASSERT(data.n_rows() == max_vals_.size());
    DEBUG_ASSERT(data.n_rows() == min_vals_.size());
    index_t n_t = data.n_cols();
    double min_error = error_;
    bool some_split_found = false;
    index_t nsd = 0;


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
	double min_dim_error = min_error,
	  temp_lval = 0.0, temp_rval = 0.0;
	index_t dim_split_ind = -1, ind = 0;

	double range = 1.0;
	for (index_t i = 0; i < max_vals_.size(); i++) {
	  // 	if (dim_type_[i] == REAL) {
	  if (max_vals_[i] -min_vals_[i] > 0.0 && i != dim) {
	    range *= max_vals_[i] - min_vals_[i];
	  }
	  // 	}
	}

	// get the values for the dimension
	std::vector<double> dim_val_vec;
	for (index_t i = 0; i < n_t; i++) {
	  dim_val_vec.push_back (data.get(dim, i));
	}
	// sort the values in ascending order
	std::sort(dim_val_vec.begin(), dim_val_vec.end());

	// get ready to go through the sorted list and compute error

	// enforcing the leaves to have a minimum of MIN_LEAF_SIZE 
	// number of points to avoid spikes
	for (std::vector<double>::iterator it = dim_val_vec.begin();
	     it < dim_val_vec.end() -1; it++, ind++) {
	  double split;
	  // 	if (dim_type_[dim] == REAL) {
	  split = (*it + *(it+1))/2;
	  // 	} else {
	  // 	  split = *it;
	  // 	}
	  if (split - min > 0.0 && max - split > 0.0) {
	    double temp_l = -1.0 * ((double)(ind+1)/(double)total_n)
	      * ((double)(ind+1)/(double)total_n)
	      / (range * (split - min));
	    DEBUG_ASSERT(-1.0*temp_l < DBL_MAX);
	    double temp_r = -1.0 * ((double)(n_t - ind-1)/(double)total_n)
	      * ((double)(n_t - ind-1)/(double)total_n)
	      / (range * (max - split));
	    DEBUG_ASSERT(-1.0*temp_r < DBL_MAX);

	    if (temp_l + temp_r <= min_dim_error) {
	      //	      printf(".");
	      min_dim_error = temp_l + temp_r;
	      temp_lval = temp_l;
	      temp_rval = temp_r;
	      dim_split_ind = ind;
	      dim_split_found = true;
	    } // end if
	  } // end if
	} // end for

	dim_val_vec.clear();


	if ((min_dim_error <= min_error) && dim_split_found) {
	  min_error = min_dim_error;
	  *split_dim = dim;
	  *split_ind = dim_split_ind;
	  *left_error = temp_lval;
	  *right_error = temp_rval;
	  some_split_found = true;
	} // end if
      } else {
	nsd++;
      } // end if
    } // end for

    // This might occur when you have many instances of the
    // same point in the dataset. Have to figure out a way to
    // deal with it
    DEBUG_ASSERT(nsd != max_vals_.size());

    return some_split_found;
    DEBUG_ASSERT_MSG(some_split_found,
		     "Weird - no split found"
		     " %"LI"d points, %"LI"d %lg %"LI"d\n",
		     data.n_cols(), total_n, min_error, nsd);
  } // end FindSplit_

  void SplitData_(Matrix& data, index_t split_dim, index_t split_ind,
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

  void GetMaxMinVals_(Matrix& data, ArrayList<double> *max_vals,
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

  ///////////////////// Public Functions //////////////////////////////////////
 public:
  
  // Root node initializer
  void Init(ArrayList<double>& max_vals,
	    ArrayList<double>& min_vals,
	    index_t total_points) {
    start_ = 0;
    stop_ = total_points;
    max_vals_.InitCopy(max_vals);
    min_vals_.InitCopy(min_vals);
    left_ = NULL;
    right_ = NULL;
    error_ = ComputeNodeError_(total_points);
    DEBUG_ASSERT_MSG(-1.0*error_ < DBL_MAX, "E:%lg", error_);
  }

  // Non-root node initializer
  void Init(ArrayList<double>& max_vals,
	    ArrayList<double>& min_vals,
	    index_t start, index_t stop,
	    double error){
    start_ = start;
    stop_ = stop;
    max_vals_.InitCopy(max_vals);
    min_vals_.InitCopy(min_vals);
    left_ = NULL;
    right_ = NULL;
    DEBUG_ASSERT_MSG(-1.0*error < DBL_MAX,"E:%lg", error);
    error_ = error;
  }

  /**
   * Expand tree
   */
  double Grow(Matrix& data, ArrayList<index_t> *old_from_new) {    

    DEBUG_ASSERT(data.n_cols() == stop_ - start_);
    DEBUG_ASSERT(data.n_rows() == max_vals_.size());
    DEBUG_ASSERT(data.n_rows() == min_vals_.size());
    double left_g, right_g;

    // computing points ratio
    ratio_ = (double) (stop_ - start_)
      / (double) old_from_new->size();

    // Checking if node is large enough
    if ((stop_ - start_) > LEAF_SIZE) {

      // find the split
      index_t dim, split_ind;
      double left_error, right_error;
      if (FindSplit_(data, old_from_new->size(),
		     &dim, &split_ind,
		     &left_error, &right_error)) {

      // Split the data for the children
      Matrix data_l, data_r;
      double split_val, lsplit_val, rsplit_val;
      SplitData_(data, dim, split_ind,
		 &data_l, &data_r, old_from_new, &split_val,
		 &lsplit_val, &rsplit_val);

      // make max and min vals for the children
      ArrayList<double> max_vals_l, max_vals_r;
      ArrayList<double> min_vals_l, min_vals_r;

      max_vals_l.InitCopy(max_vals_);
      max_vals_r.InitCopy(max_vals_);
      min_vals_l.InitCopy(min_vals_);
      min_vals_r.InitCopy(min_vals_);
      max_vals_l[dim] = split_val;
      min_vals_r[dim] = split_val;

      DEBUG_ASSERT(max_vals_l.size() == max_vals_.size());
      DEBUG_ASSERT(min_vals_l.size() == min_vals_.size());
      DEBUG_ASSERT(max_vals_r.size() == max_vals_.size());
      DEBUG_ASSERT(min_vals_r.size() == min_vals_.size());
      // store split dim and split val in the node
      split_value_ = split_val;
      split_dim_ = dim;

      // Recursively growing the children
      left_ = new DTree();
      right_ = new DTree();
      left_->Init(max_vals_l, min_vals_l, start_,
		  start_ + split_ind + 1, left_error);
      right_->Init(max_vals_r, min_vals_r, start_
		   + split_ind + 1, stop_, right_error);
      left_g = left_->Grow(data_l, old_from_new);
      right_g = right_->Grow(data_r, old_from_new);

      // storing values of R(T~) and |T~|
      subtree_leaves_ = left_->subtree_leaves() + right_->subtree_leaves();
      subtree_leaves_error_ = left_->subtree_leaves_error()
	+ right_->subtree_leaves_error();

      // Forming T1 by removing leaves for which
      // R(t) = R(t_L) + R(t_R)
      if ((left_->subtree_leaves() == 1)
	  && (right_->subtree_leaves() == 1)) {
	if (left_->error() + right_->error() == error_) {
	  delete left_;
	  left_ = NULL;
	  delete right_;
	  right_ = NULL;
	  subtree_leaves_ = 1;
	  subtree_leaves_error_ = error_;
	} // end if
      } // end if
      } else {
	// no split found so make a leaf out of it
	subtree_leaves_ = 1;
	subtree_leaves_error_ = error_;
      } // end if-else
    } else {
      // This is a leaf node, do something here, probably compute
      // density here or something
      DEBUG_ASSERT_MSG(stop_ - start_ >= MIN_LEAF_SIZE,
		       "%"LI"d points", stop_ - start_);
      subtree_leaves_ = 1;
      subtree_leaves_error_ = error_;
    } // end if-else

    // if leaf do not compute g_k(t), else compute, store,
    // and propagate min(g_k(t_L),g_k(t_R),g_k(t)), 
    // unless t_L and/or t_R are leaves
    if (subtree_leaves_ == 1) {
      return DBL_MAX;
    } else {
      double g_t = (error_ - subtree_leaves_error_) 
	/ (subtree_leaves_ - 1);

      return min(g_t, min(left_g, right_g));
    } // end if-else

    // need to compute (c_t^2)*r_t for all subtree leaves
    // this is equal to n_t^2/r_t*n^2 = -error_ !!
    // therefore the value we need is actually
    // -1.0*subtree_leaves_error_
  } // Grow

  double PruneAndUpdate(double old_alpha) {

    // compute g_t
    if (subtree_leaves_ == 1) {
      return DBL_MAX;
    } else {
      double g_t = (error_ - subtree_leaves_error_)
	/ (subtree_leaves_ - 1);

      if (g_t > old_alpha) { // go down the tree and update accordingly
	// traverse the children
	double left_g = left_->PruneAndUpdate(old_alpha);
	double right_g = right_->PruneAndUpdate(old_alpha);

	// update values
	subtree_leaves_ = left_->subtree_leaves()
	  + right_->subtree_leaves();
	subtree_leaves_error_ = left_->subtree_leaves_error()
	  + right_->subtree_leaves_error();

	// update g_t value
	g_t = (error_ - subtree_leaves_error_)
	  / (subtree_leaves_ - 1);

	DEBUG_ASSERT_MSG(g_t < DBL_MAX,
			 "g:%lg, rt:%lg, rtt:%lg, l:%"LI"d",
			 g_t, error_, subtree_leaves_error_,
			 subtree_leaves_);
	if (left_->subtree_leaves() == 1
	    && right_->subtree_leaves() == 1) {
	  return g_t;
	} else if (left_->subtree_leaves() == 1) {
	  return min(g_t, right_g);
	} else if (right_->subtree_leaves() == 1) {
	  return min(g_t, left_g);
	} else {
	  return min(g_t, min(left_g, right_g));
	}
      } else { // prune this subtree
	// making this node a leaf node
	subtree_leaves_ = 1;
	subtree_leaves_error_ = error_;
	delete left_;
	left_ = NULL;
	delete right_;
	right_ = NULL;
	// passing information upward
	return DBL_MAX;
      } // end if-else
    }
  } // PruneAndUpdate

  double ComputeValue(Vector& query) {

    DEBUG_ASSERT_MSG(query.length() == max_vals_.size(),
		     "dim = %"LI"d, maxval size= %"LI"d"
		     ", sl=%"LI"d",
		     query.length(), max_vals_.size(),
		     subtree_leaves_);
    DEBUG_ASSERT(query.length() == min_vals_.size());

    if (subtree_leaves_ == 1) { // if leaf
      // compute value
      // compute r_t
      double range = 1.0;
      for (index_t i = 0; i < max_vals_.size(); i++) {
	if (max_vals_[i] - min_vals_[i] > 0.0) {
	  range *= max_vals_[i] - min_vals_[i];
	}
      } // end for
      return ratio_ / range;
    } else if (query[split_dim_] <= split_value_) { // if left subtree
      // go to left child
      return left_->ComputeValue(query);
    } else { // if right subtree
      // go to right child
      return right_->ComputeValue(query);
    } // end if-else
  } // ComputeValue  

  void WriteTree(index_t level){
    if (likely(left_ != NULL)){
      printf("\n");
      for (index_t i = 0; i < level; i++){
	printf("|\t");
      }
      printf("Var. %"LI"d > %lg ", split_dim_, split_value_);
      right_->WriteTree(level+1);
      printf("\n");
      for (index_t i = 0; i < level; i++){
	printf("|\t");
      }      
      printf("Var. %"LI"d <= %lg ", split_dim_, split_value_);
      left_->WriteTree(level+1);
    } else {
      double range = 1.0;
      for (index_t i = 0; i < max_vals_.size(); i++) {
	if (max_vals_[i] - min_vals_[i] > 0.0) {
	  range *= max_vals_[i] - min_vals_[i];
	}
      } // end for
      printf("-> Predict:%lg", ratio_ / range);
    }  
  }
  
}; // Class DTree

#endif
