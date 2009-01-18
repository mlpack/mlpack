/**
 * @file dtree.h
 *
 * Density Tree class
 *
 */

#ifndef DTREE_H
#define DTREE_H

#include <vector>

#include "fastlib/fastlib.h"
#define LEAF_SIZE 5

class DTree{
 

  /////////////////// Nested Classes /////////////////////////////////////////


  ////////////////////// Member Variables /////////////////////////////////////
  
 private:
  // The indices in the complete set of points
  // (after all forms of swapping in the 
  // old_from_new array)
  int start_, stop_;

  // The split dim
  index_t split_dim_;

  // The split val on that dim
  double split_value_;

  // error of the node
  double error_;

  // error of the leaves of the subtree
  double subtree_leaves_error_;

  // number of leaves of the subtree
  index_t subtree_leaves_;

  // since we are using uniform density, we need
  // the max and min of every dimension for every node
  ArrayList<double> *max_vals_;
  ArrayList<double> *min_vals_;

  // The children
  DTree *left_;
  DTree *right_;

  ////////////////////// Constructors /////////////////////////////////////////
  
  FORBID_ACCIDENTAL_COPIES(DTree);

 public: 

  DTree() {
    max_vals_ = NULL;
    min_vals_ = NULL;
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
//     if (max_vals_ != NULL) {
//       max_vals_->Clear();
//     }
//     if (min_vals_ != NULL) {
//       min_vals_->Clear();
//     }
  }

  ////////////////////// Getters and Setters //////////////////////////////////
  index_t start() { return start_; }

  index_t stop() { return stop_; }

  index_t split_dim() { return split_dim_; }

  double error() { return error_; }

  double subtree_leaves_error() { return subtree_leaves_error_; }

  index_t subtree_leaves() { return subtree_leaves_; }

  DTree* left_child() { return left_; }
  DTree* right_child() { return right_; }

  ////////////////////// Private Functions ////////////////////////////////////
  
  double ComputeNodeError_(index_t total_points) {
    double range = 1.0;
    index_t node_size = stop_ - start_;

    DEBUG_ASSERT(max_vals_->size() == min_vals_->size());
    for (index_t i = 0; i < max_vals_->size(); i++) {
      // if no variation in a dimension, we do not care
      // about that dimension
      if (((*max_vals_)[i] - (*min_vals_)[i]) > 0.0) {
	range *= ((*max_vals_)[i] - (*min_vals_)[i]);
      }
    }

    double error = -1.0 * node_size * node_size
      / (range * total_points * total_points);

    return error;
  }

  void FindSplit_(Matrix& data, index_t total_n,
		  index_t *split_dim, index_t *split_ind,
		  double *left_error, double *right_error) {

//     NOTIFY("FindSplit")
    DEBUG_ASSERT(data.n_cols() == stop_ - start_);
    index_t n_t = data.n_cols();
    double min_error = error_;
    bool some_split_found = false;

    // loop through each dimension
    for (index_t dim = 0; dim < max_vals_->size(); dim++) {
      bool dim_split_found = false;

      double min = (*min_vals_)[dim], max = (*max_vals_)[dim];
      double range = 1.0;
      for (index_t i = 0; i < max_vals_->size(); i++) {
	if (((*max_vals_)[i] -(*min_vals_)[i] > 0.0) && (i != dim)) {
	  range *= ((*max_vals_)[i] -(*min_vals_)[i]);
	}
      }
      double k = -1.0 / (total_n * total_n * range);

      // get the values for the dimension
      std::vector<double> dim_val_vec;
      for (index_t i = 0; i < n_t; i++) {
	dim_val_vec.push_back (data.get(dim, i));
      }
      // sort the values in ascending order
      std::sort(dim_val_vec.begin(), dim_val_vec.end());

      // get ready to go through the sorted list and compute error
      double min_dim_error = min_error,
	temp_lval = 0.0, temp_rval = 0.0;
      index_t dim_split_ind = -1, ind = 0;
      for (std::vector<double>::iterator it = dim_val_vec.begin();
	   it < dim_val_vec.end()-1; it++, ind++) {
	double split = (*it + *(it+1))/2;
	double temp_l = k * (ind+1) * (ind+1) / (split - min);
	double temp_r = k * (n_t - ind-1) * (n_t - ind-1) / (max - split);
	if (temp_l + temp_r <= min_dim_error) {
	  min_dim_error = temp_l + temp_r;
	  temp_lval = temp_l;
	  temp_rval = temp_r;
	  dim_split_ind = ind;
	  dim_split_found = true;
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
    } // end for

    DEBUG_ASSERT_MSG(some_split_found, "Weird - no split found\n");
  } // end FindSplit_()

  void SplitData_(Matrix& data, index_t split_dim, index_t split_ind,
		  Matrix *data_l, Matrix *data_r, 
		  ArrayList<index_t> *old_from_new, 
		  double *split_val) {

//     NOTIFY("SplitData %"LI"d %"LI"d", split_ind, data.n_cols());
    // get the values for the split dim
    std::vector<double> dim_val_vec;
    for (index_t i = 0; i < data.n_cols(); i++) {
      dim_val_vec.push_back(data.get(split_dim, i));
    } // end for

    // sort the values
    std::sort(dim_val_vec.begin(), dim_val_vec.end());

    *split_val = (*(dim_val_vec.begin()+split_ind)
      +  *(dim_val_vec.begin()+split_ind+1)) / 2;

//     NOTIFY("Split val %lg", *split_val);
    index_t i = split_ind, j = split_ind + 1;
    while ( i > -1 && j < data.n_cols()) {
      while (i > -1 && data.get(split_dim, i) < *split_val)
	i--;

      while (j < data.n_cols() && data.get(split_dim, j) > *split_val)
	j++;

      // swapping values
      if (i > -1 && j < data.n_cols()) {
	//	NOTIFY("swapping %"LI"d", i);
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

  } // end SplitData_()

  ///////////////////// Helper Functions //////////////////////////////////////

  
  // Root node initializer
  void Init(ArrayList<double> *max_vals,
	    ArrayList<double> *min_vals,
	    index_t total_points) {
    start_ = 0;
    stop_ = total_points;
    max_vals_ = max_vals;
    min_vals_ = min_vals;
    left_ = NULL;
    right_ = NULL;
    error_ = ComputeNodeError_(total_points);
  }

  // Non-root node initializer
  void Init(ArrayList<double> *max_vals,
	    ArrayList<double> *min_vals,
	    index_t start, index_t stop,
	    double error){
    start_ = start;
    stop_ = stop;
    max_vals_ = max_vals;
    min_vals_ = min_vals;
    left_ = NULL;
    right_ = NULL;
    error_ = error;
  }

  /*
   * Expand tree
   */
  double Grow(Matrix& data, ArrayList<index_t> *old_from_new) {    

    //     NOTIFY("grow");
    DEBUG_ASSERT(data.n_cols() == stop_ - start_);
    double left_g, right_g;
    if ((stop_ - start_) > LEAF_SIZE) {

      // find the split
      index_t dim, split_ind;
      double left_error, right_error;
      //    printf("here %"LI"d\n", data.n_cols());
      FindSplit_(data, old_from_new->size(),
		 &dim, &split_ind,
		 &left_error, &right_error);
//       printf(" there %"LI"d", split_ind);
      // Split the data for the children
      Matrix data_l, data_r;
      double split_val;
      SplitData_(data, dim, split_ind,
		 &data_l, &data_r, old_from_new, &split_val);

      // make max and min vals for the children
      ArrayList<double> max_vals_l, max_vals_r;
      ArrayList<double> min_vals_l, min_vals_r;
      max_vals_l.InitCopy(*max_vals_);
      max_vals_r.InitCopy(*max_vals_);
      min_vals_l.InitCopy(*min_vals_);
      min_vals_r.InitCopy(*min_vals_);
      max_vals_l[dim] = split_val;
      min_vals_r[dim] = split_val;


      // Recursively growing the children
      left_ = new DTree();
      right_ = new DTree();
      left_->Init(&max_vals_l, &min_vals_l, start_,
		  start_ + split_ind + 1, left_error);
      right_->Init(&max_vals_r, &min_vals_r, start_
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
	  delete right_;
	  subtree_leaves_ = 1;
	  subtree_leaves_error_ = error_;
	} // end if
      } // end if
    } else {
      // This is a leaf node, do something here, probably compute
      // density here or something
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

	return min(g_t, min(left_g, right_g));

      } else { // prune this subtree
	// making this node a leaf node
	//	printf(".");fflush(NULL);
	subtree_leaves_ = 1;
	subtree_leaves_error_ = error_;
	delete left_;
	left_ = NULL;
	delete right_;
	right_ = NULL;
	//	printf("+");fflush(NULL);
	// passing information upward
	return DBL_MAX;
      } // end if-else
    }
  } // PruneAndUpdate
  
//  /*
//    * Prune tree- remove subtrees if small decrease in
//    * Gini index does not justify number of leaves.
//    */
 
//   double Prune(double lambda){
//     if (likely(left_ != NULL)){
//       double result;
//       result = left_->Prune(lambda);
//       result = min(result, right_->Prune(lambda));     
//       double child_error;
//       int leafs;
//       leafs = left_->GetNumNodes() + right_->GetNumNodes();
//       child_error = left_->GetChildError()*left_->Count()
// 	+ right_->GetChildError()*right_->Count();      
//       double criterion = (stop_ - start_)*error_ - child_error;   
//       if (criterion <= (lambda * (leafs-1))) {
// 	left_ = NULL;
// 	right_ = NULL;
// 	return BIG_BAD_NUMBER;
//       } else {	
// 	return min(result, criterion / (leafs-1));
//       }	
//     } else {
//       return BIG_BAD_NUMBER;
//     }
//   }
  

//   void SetTestError(TrainingSet* test_data, int index){
//     test_count_++;
//     if (likely(left_ != NULL)){
//       ArrayList<ArrayList<double> > split_data;
//       split_data = split_criterion_.GetSplitParams();
//       int split_dim = (int)split_data[0][0];
//       // Split variable is not ordered...
//       if (test_data->GetVariableType(split_dim) > 0 ){
// 	bool go_left = 0;
// 	for (int i = 1; i < split_data[0].size(); i++){
// 	  go_left = go_left | 
// 	  (split_data[0][i] == (int)(test_data->Get(split_dim, index)));
// 	}
// 	if (go_left){
// 	  left_->SetTestError(test_data, index);
// 	} else {
// 	  right_->SetTestError(test_data, index);
// 	}
//       } else {
//       // Split Variable is ordered.
//  	double split_point = split_data[0][1];
// 	if (test_data->Get(split_dim, index) >= split_point){
// 	  right_->SetTestError(test_data, index);
// 	} else {
// 	  left_->SetTestError(test_data, index);
// 	}
//       }
//     } 
//     if (test_data->GetTargetType(target_dim_) > 0) {
//       test_error_ = test_error_ + 
// 	(value_ != test_data->Get(target_dim_, index));
//     } else {
//       test_error_ = test_error_ + 
// 	(value_ - test_data->Get(target_dim_, index))*
// 	(value_ - test_data->Get(target_dim_, index));
//     }
//   }

//   double GetTestError(){
//     if (likely(left_ != NULL)){
//       return left_->GetTestError() + right_->GetTestError();
//     } else {
//       if (points_->GetTargetType(target_dim_) > 0) {
// 	return test_error_;
//       } else {
// 	return sqrt(test_error_ / test_count_);
//       }
//     }
//   }

//   void WriteTree(int level, FILE* fp){
//     if (likely(left_ != NULL)){
//       fprintf(fp, "\n");
//       for (int i = 0; i < level; i++){
// 	fprintf(fp, "|\t");
//       }
//       ArrayList<ArrayList<double> > split_data;
//       split_data = split_criterion_.GetSplitParams();      
//       int split_dim;
//       double split_val;
//       split_dim =  (int)split_data[0][0];
//       split_val = split_data[0][1];      
//       fprintf(fp, "Var. %d >=%5.2f ", split_dim, split_val);
//       right_->WriteTree(level+1, fp);
//       fprintf(fp, "\n");
//       for (int i = 0; i < level; i++){
// 	fprintf(fp, "|\t");
//       }      
//       fprintf(fp, "Var. %d < %5.2f ", split_dim, split_val);
//       left_->WriteTree(level+1, fp);
//     } else {            
//       fprintf(fp, ": Predict =%4.0f", value_);
//     }  
//   }
  
//   /*
//    * Find predicted value for a given data point
//    */
//   double Test(TrainingSet* test_data, int index){   
//     if (likely(left_ != NULL)){
//       ArrayList<ArrayList<double> > split_data;
//       split_data = split_criterion_.GetSplitParams();
//       int split_dim = (int)split_data[0][0];
//       // Split variable is not ordered...
//       if (test_data->GetVariableType(split_dim) > 0 ){
// 	bool go_left = 0;
// 	for (int i = 1; i < split_data[0].size(); i++){
// 	  go_left = go_left | 
// 	  (split_data[0][i] == (int)(test_data->Get(split_dim, index)));
// 	}
// 	if (go_left){
// 	  return left_->Test(test_data, index);
// 	} else {
// 	  return right_->Test(test_data, index);
// 	}
//       } else {
//       // Split Variable is ordered.
//  	double split_point = split_data[0][1];
// 	if (test_data->Get(split_dim, index) >= split_point){
// 	  return right_->Test(test_data, index);
// 	} else {
// 	  return left_->Test(test_data, index);
// 	}
//       }
//     } else{
//       return value_;
//     }
//   } // Test

//   double GetChildError(){
//     if (left_ != NULL){
//       return (left_->GetChildError()*left_->Count() + 
// 	      right_->GetChildError()*right_->Count()) / (stop_ - start_);
//     } else {
//       return error_;
//     }
//   }

//   int GetNumNodes(){
//     if (left_  != NULL){
//       return left_->GetNumNodes() + right_->GetNumNodes();
//     } else {
//       return 1; 
//     }
//   }

//   int Count(){
//     return stop_ - start_;
//   }

//   void SetValue(double val_in){
//     value_ = val_in;
//   }
  
}; // Class DTree

#endif
