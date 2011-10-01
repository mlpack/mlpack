 /**
 * @file ttree.h
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Tent Tree class
 *
 */

#ifndef TTREE_H
#define TTREE_H

#include <vector>

#include "fastlib/fastlib.h"
#include <stdlib.h>
#include <time.h>

const fx_entry_doc ttree_entries[] = {
  {"min_leaf_size", FX_PARAM, FX_INT, NULL,
   " Minimum leaf size in the unpruned tree "
   "(defaults to 15).\n"},
  {"max_leaf_size", FX_PARAM, FX_INT, NULL,
   " Maximum leaf size in the unpruned tree "
   "(defaults to 30).\n"},
  {"b", FX_PARAM, FX_INT, NULL,
   " Number of bootstrap steps.\n"},
  {"do_bootstrap", FX_PARAM, FX_BOOL, NULL,
   " Whether to do the bootstrap or not.\n"},
  {"use_vol_reg", FX_PARAM, FX_BOOL, NULL,
   " Whether to use usual decision tree "
   "regularization (default/false) or use the volume of "
   "the node for regularization (if true)"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc ttree_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc ttree_doc = {
  ttree_entries, ttree_submodules,
  "Parameters of the decision tree data structure.\n"
};


class TTree{
 
  ////////////////////// Member Variables /////////////////////////////////////
  
private:
  // The indices in the complete set of points
  // (after all forms of swapping in the 
  // old_from_new array)
  int start_, stop_;
  
  // The split dim
  size_t split_dim_;

//   // split dim types
//   ArrayList<enum> dim_types_;

//   // The split value for non-numeric data
//   string split_val_;

  // The split val on that dim
  double split_value_;

  // error of the node
  double error_;

  double gamma_;

  // error of the leaves of the subtree
  double subtree_leaves_error_;

  // number of leaves of the subtree
  size_t subtree_leaves_;

  // flag to indicate if this is the root node
  // used to check whether the query point is 
  // within the range
  size_t root_;

  // ratio of number of points in the node
  double ratio_;


  // the volume of the node
  // long double v_t_inv_;

  // sum of the reciprocal of the inverse v_ts
  // the leaves of this subtree
  // long double subtree_leaves_v_t_inv_;

  // sum of the estimates of the points in the subtree divided by N
  // double st_estimate_;

  // del f / del r(split_dim)
  // double del_f_del_r_;

  // since we are using uniform density, we need
  // the max and min of every dimension for every node
  ArrayList<double> max_vals_;
  ArrayList<double> min_vals_;

  // the tag for the leaf used for hashing points
  size_t bucket_tag_;

  // The children
  TTree *left_;
  TTree *right_;

  // The module containing the parameters for the data structure
  struct datanode* module_;

  ////////////////////// Constructors /////////////////////////////////////////
  
  FORBID_ACCIDENTAL_COPIES(TTree);

public: 

  TTree() {
    left_ = NULL;
    right_ = NULL;
  }

  ~TTree() {
    if (left_ != NULL){
      delete left_;
    }
    if (right_ != NULL){
      delete right_;
    }
  }

  ////////////////////// Getters and Setters //////////////////////////////////
  size_t start() { return start_; }

  size_t stop() { return stop_; }

  size_t split_dim() { return split_dim_; }

  double split_value() { return split_value_; }

  double error() { return error_; }

  double subtree_leaves_error() { return subtree_leaves_error_; }

  size_t subtree_leaves() { return subtree_leaves_; }

  double gamma() { return gamma_; }

  double ratio() { return ratio_; }

  // long double v_t_inv() { return v_t_inv_; }

  // long double subtree_leaves_v_t_inv() { return subtree_leaves_v_t_inv_; }

  // double st_estimate() {return st_estimate_; }

  TTree* left_child() { return left_; }
  TTree* right_child() { return right_; }

  size_t root() { return root_; }

  ////////////////////// Private Functions ////////////////////////////////////
 private:

//   void SampleSetWithReplacement_(std::vector<double> &dim_vals,
// 				 std::vector<double> *bs_dim_vals);

  double ComputeNodeError_(Vector& y, double *gamma);
  
  bool FindSplit_(Matrix& x, Vector& y, size_t total_n,
		  size_t *split_dim, double *split_val,
		  double *left_error, double *right_error,
		  double *gam_l, double *gam_r);

  void SplitData_(Matrix& x, Vector& y,
		  size_t split_dim, double split_val,
		  Matrix *x_l, Vector *y_l, 
		  Matrix *x_r, Vector *y_r,
		  ArrayList<size_t> *old_from_new);
// 		  double *split_val,
// 		  double *lsplit_val, double *rsplit_val);

  void GetMaxMinVals_(Matrix& x, ArrayList<double> *max_vals,
		      ArrayList<double> *min_vals);

  ///////////////////// Public Functions //////////////////////////////////////
 public:
  
  // Root node initializer
  void Init(ArrayList<double>& max_vals,
	    ArrayList<double>& min_vals,
	    size_t total_points, 
	    Vector& y, datanode* module) {
    start_ = 0;
    stop_ = total_points;
    max_vals_.InitCopy(max_vals);
    min_vals_.InitCopy(min_vals);
    left_ = NULL;
    right_ = NULL;

    // change this to compute node error using the y-vals
    error_ = ComputeNodeError_(y, &gamma_);

    // printf("root: %lg\n", gamma_);
    DEBUG_ASSERT_MSG(error_ < DBL_MAX, "E:%lg", error_);
    bucket_tag_ = -1;
    module_ = module;

    root_ = 1;
  }

  // Non-root node initializer
  void Init(ArrayList<double>& max_vals,
	    ArrayList<double>& min_vals,
	    size_t start, size_t stop,
	    double error, double gam, 
	    datanode* module){
    start_ = start;
    stop_ = stop;
    max_vals_.InitCopy(max_vals);
    min_vals_.InitCopy(min_vals);
    left_ = NULL;
    right_ = NULL;
    DEBUG_ASSERT_MSG(error < DBL_MAX,"E:%lg", error);
    error_ = error;
    gamma_ = gam;
    bucket_tag_ = -1;
    module_ = module;

    root_ = 0;

    // printf(": %lg\n", gamma_);
  }

  /**
   * Expand tree
   */
  double Grow(Matrix& x, Vector& y,
	      ArrayList<size_t> *old_from_new) {    

    DEBUG_ASSERT(x.n_cols() == stop_ - start_);
    DEBUG_ASSERT(x.n_rows() == max_vals_.size());
    DEBUG_ASSERT(x.n_rows() == min_vals_.size());

    DEBUG_ASSERT(x.n_cols() == y.length());

    double left_g, right_g;

//     // computing points ratio
//     ratio_ = (double) (stop_ - start_)
//       / (double) old_from_new->size();

//     // computing the v_t_inv:
//     // the inverse of the volume of the node
//     v_t_inv_ = 1.0;
//     for (size_t i = 0; i < max_vals_.size(); i++)
//       if (max_vals_[i] - min_vals_[i] > 0.0) 
// 	v_t_inv_ /= (long double) (max_vals_[i] - min_vals_[i]);
//       else {
// 	// in case of point mass use token binwidth
// 	// v_t_inv_ /= 1e-4;
//       }

    // Checking if node is large enough
    if ((stop_ - start_) > fx_param_int(module_, "max_leaf_size", 10)) {

      // find the split
      size_t dim;
      double split_val;
      double gam_l, gam_r;
      double left_error, right_error;
      if (FindSplit_(x, y, old_from_new->size(),
		     &dim, &split_val,
		     &left_error, &right_error,
		     &gam_l, &gam_r)) {

	// printf("Split found: %lg, %lg\n", gam_l, gam_r);fflush(NULL);
	// Split the x for the children
	Matrix x_l, x_r;
	Vector y_l, y_r;

	// double lsplit_val, rsplit_val;
	SplitData_(x, y, dim, split_val,
		   &x_l, &y_l, &x_r, &y_r,  
		   old_from_new); 
	// &split_val,
// 		   &lsplit_val, &rsplit_val);

	// make max and min vals for the children
	ArrayList<double> max_vals_l, max_vals_r;
	ArrayList<double> min_vals_l, min_vals_r;

	max_vals_l.InitCopy(max_vals_);
	max_vals_r.InitCopy(max_vals_);
	min_vals_l.InitCopy(min_vals_);
	min_vals_r.InitCopy(min_vals_);
	max_vals_l[dim] = split_val; // changed from just lsplit_val
	min_vals_r[dim] = split_val; // changed from just rsplit_val

	DEBUG_ASSERT(max_vals_l.size() == max_vals_.size());
	DEBUG_ASSERT(min_vals_l.size() == min_vals_.size());
	DEBUG_ASSERT(max_vals_r.size() == max_vals_.size());
	DEBUG_ASSERT(min_vals_r.size() == min_vals_.size());
	// store split dim and split val in the node
	split_value_ = split_val;
	split_dim_ = dim;



	// Recursively growing the children
	left_ = new TTree();
	right_ = new TTree();
	left_->Init(max_vals_l, min_vals_l, start_,
		    start_ + x_l.n_cols(), left_error, 
		    gam_l, module_);
	right_->Init(max_vals_r, min_vals_r, start_
		     + x_l.n_cols(), stop_, right_error,
		     gam_r,  module_);
	left_g = left_->Grow(x_l, y_l, old_from_new);
	right_g = right_->Grow(x_r, y_r, old_from_new);

	// storing values of R(T~) and |T~|
	subtree_leaves_ = left_->subtree_leaves() + right_->subtree_leaves();
	subtree_leaves_error_ = left_->subtree_leaves_error()
	  + right_->subtree_leaves_error();

	// JT : TO REMOVE
// 	DEBUG_WARN_MSG_IF(error_ == subtree_leaves_error_,
// 		     "G: error = subtree leaves error\n");

// 	if (error_ == subtree_leaves_error_) 
// 	  printf("G: IF: error = subtree error\n");
	//////////////////////////

// 	// storing the subtree_leaves_v_t_inv
// 	subtree_leaves_v_t_inv_ = left_->subtree_leaves_v_t_inv()
// 	  + right_->subtree_leaves_v_t_inv();

	// storing the sum of the estimates
// 	st_estimate_ = left_->st_estimate() + right_->st_estimate();

// 	// storing del_f / del r(split_dim)
// 	double del_f = (ratio_ * v_t_inv_)
// 	  - (left_->ratio() * left_->v_t_inv());
// 	double del_r = max_vals_[split_dim_] - split_value_;
// 	del_f_del_r_ = fabs(del_f / del_r);

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
	    // subtree_leaves_v_t_inv_ = v_t_inv_;
	  } // end if
	} // end if
      } else {
	// no split found so make a leaf out of it
	subtree_leaves_ = 1;
	subtree_leaves_error_ = error_;
	// subtree_leaves_v_t_inv_ = v_t_inv_;
	// st_estimate_ = ratio_ * ratio_ * v_t_inv_;
	// del_f_del_r_ = 0.0;
      } // end if-else
    } else {
      // This is a leaf node, do something here, probably compute
      // density here or something
      DEBUG_ASSERT_MSG(stop_ - start_ >= 
		       fx_param_int(module_, "min_leaf_size", 5),
		       "%zu"d points", stop_ - start_);
      subtree_leaves_ = 1;
      subtree_leaves_error_ = error_;
//       subtree_leaves_v_t_inv_ = v_t_inv_;
//       st_estimate_ = ratio_ * ratio_ * v_t_inv_;
//       del_f_del_r_ = 0.0;
    } // end if-else
    
    // if leaf do not compute g_k(t), else compute, store,
    // and propagate min(g_k(t_L),g_k(t_R),g_k(t)), 
    // unless t_L and/or t_R are leaves
    if (subtree_leaves_ == 1) {
      return DBL_MAX;
    } else {
      double g_t;
//       if (fx_param_bool(module_, "use_vol_reg", false)) {
// 	g_t = (error_ - subtree_leaves_error_) 
// 	  / (subtree_leaves_v_t_inv_ - v_t_inv_);
//       } else {
      g_t = (error_ - subtree_leaves_error_) 
	/ (subtree_leaves_ - 1);
//       }

      DEBUG_ASSERT(g_t > 0.0);
      return min(g_t, min(left_g, right_g));
    } // end if-else

  } // Grow


  double PruneAndUpdate(double old_alpha) {

    // compute g_t
    if (subtree_leaves_ == 1) {
      // printf("%lg:Leaf\n", old_alpha);
      return DBL_MAX;
    } else {
      double g_t;
//       if (fx_param_bool(module_, "use_vol_reg", false)) {
// 	g_t = (error_ - subtree_leaves_error_) 
// 	  / (subtree_leaves_v_t_inv_ - v_t_inv_);
//       } else {
	g_t = (error_ - subtree_leaves_error_) 
	  / (subtree_leaves_ - 1);
//       }

      if (g_t > old_alpha) { // go down the tree and update accordingly
	// traverse the children
	//printf("%lg:%lg LEFT\n", old_alpha, del_f_del_r_);
	double left_g = left_->PruneAndUpdate(old_alpha);
	//printf("%lg:%lg RIGHT\n", old_alpha, del_f_del_r_);
	double right_g = right_->PruneAndUpdate(old_alpha);

	// update values
	subtree_leaves_ = left_->subtree_leaves()
	  + right_->subtree_leaves();
	subtree_leaves_error_ = left_->subtree_leaves_error()
	  + right_->subtree_leaves_error();
// 	subtree_leaves_v_t_inv_ = left_->subtree_leaves_v_t_inv()
// 	  + right_->subtree_leaves_v_t_inv();

// 	// updating values for the sum of density estimates 
// 	st_estimate_
// 	  = left_->st_estimate() + right_->st_estimate();

	// update g_t value
// 	if (fx_param_bool(module_, "use_vol_reg", false)) {
// 	  g_t = (error_ - subtree_leaves_error_) 
// 	    / (subtree_leaves_v_t_inv_ - v_t_inv_);
// 	} else {
	  g_t = (error_ - subtree_leaves_error_) 
	    / (subtree_leaves_ - 1);
// 	}

	DEBUG_ASSERT(g_t < DBL_MAX);
// 			 "g:%lg, rt:%lg, rtt:%lg, l:%zu"d",
// 			 g_t, error_, subtree_leaves_error_,
// 			 subtree_leaves_);
	//printf("%lg:Return:%lg\n", old_alpha, min(min(left_g, right_g), g_t));

	return min(g_t, min(left_g, right_g));
// 	if (left_->subtree_leaves() == 1
// 	    && right_->subtree_leaves() == 1) {
// 	  return g_t;
// 	} else if (left_->subtree_leaves() == 1) {
// 	  return min(g_t, right_g);
// 	} else if (right_->subtree_leaves() == 1) {
// 	  return min(g_t, left_g);
// 	} else {
// 	  return min(g_t, min(left_g, right_g));
// 	}
      } else { // prune this subtree

	// otherwise this should be equal to the alpha
	// for this node. So we check that:
	// IMP: The thing is that it should be equal in the 
	// full original tree, but once you start CV, the 
	// CV trees have different values for alpha and are 
	// just pruning on the provided alpha values
	// DEBUG_ASSERT_MSG(g_t == old_alpha, "Alpha != g(t) but less than!!");

	// compute \del f_hat(x) / \del r(split_dim)
// 	double st_change_in_estimate 
// 	  = st_estimate_ - (ratio_ * ratio_ * v_t_inv_);

 	// printf("%lg:%lg Pruned %lg\n",
 	//       old_alpha, del_f_del_r_, st_change_in_estimate);

	// JT : TO REMOVE
// 	DEBUG_WARN_MSG_IF(error_ == subtree_leaves_error_,
// 		     "PU: error = subtree leaves error\n");

// 	if (error_ == subtree_leaves_error_) 
// 	  printf("PU: IF: error = subtree error\n");

// 	printf("PU: %Lg, %Lg, %Lg\n", error_, subtree_leaves_error_, 
// 	       error_ - subtree_leaves_error_);
	//////////////////////////


	// making this node a leaf node
	subtree_leaves_ = 1;
	subtree_leaves_error_ = error_;
// 	subtree_leaves_v_t_inv_ = v_t_inv_;
// 	st_estimate_ = ratio_ * ratio_ * v_t_inv_;
// 	del_f_del_r_ = 0.0;
	delete left_;
	left_ = NULL;
	delete right_;
	right_ = NULL;
	// passing information upward
	return DBL_MAX;
      } // end if-else
    }
  } // PruneAndUpdate


  // Checking whether a given point is within the
  // bounding box of this node (check generally done
  // at the root, so its the bounding box of the x)
  //
  // Option to open up the range with epsilons on 
  // both sides
  bool WithinRange_(Vector& query) {

    for (size_t i = 0; i < query.length(); i++)
      if ((query[i] < min_vals_[i]) || (query[i] > max_vals_[i]))
	return false;

    return true;
  }

  double ComputeValue(Vector& query, bool printer) {

    DEBUG_ASSERT_MSG(query.length() == max_vals_.size(),
		     "dim = %zu"d, maxval size= %zu"d"
		     ", sl=%zu"d",
		     query.length(), max_vals_.size(),
		     subtree_leaves_);
    DEBUG_ASSERT(query.length() == min_vals_.size());


    if (root_ == 1) // if root
      // check if query is within range
      if (!WithinRange_(query))
	return gamma_;

    if (subtree_leaves_ == 1)  // if leaf
      return gamma_;
    else
      if (query[split_dim_] <= split_value_)  // if left subtree
	// go to left child
	return left_->ComputeValue(query, printer);
      else  // if right subtree
	    // go to right child
	return right_->ComputeValue(query, printer);
    // end if-else
    // end WithinRange_ if-else
    // end root if
  } // ComputeValue  

  void WriteTree(size_t level, FILE *fp){
    if (likely(left_ != NULL)){
      fprintf(fp, "\n");
      for (size_t i = 0; i < level; i++){
	fprintf(fp, "|\t");
      }
//       long double g_t = (error_ - subtree_leaves_error_)
//  	/ (subtree_leaves_ - 1);
//       long double g_t = (error_ - subtree_leaves_error_)
// 	/ (subtree_leaves_v_t_inv_ - v_t_inv_);

      fprintf(fp, "Var. %zu"d > %lg",
	     split_dim_, split_value_);
      right_->WriteTree(level+1, fp);
      fprintf(fp, "\n");
      for (size_t i = 0; i < level; i++){
	fprintf(fp, "|\t");
      }      
      fprintf(fp, "Var. %zu"d <= %lg ", split_dim_, split_value_);
      left_->WriteTree(level+1, fp);
    } else {
      fprintf(fp, ": f(x)=%lg", gamma_);
      if (bucket_tag_ != -1) 
	fprintf(fp, " BT:%zu"d", bucket_tag_);
    }  
  }

  size_t TagTree(size_t tag) {
    if (subtree_leaves_ == 1) {
      bucket_tag_ = tag;
      return (tag+1);
    } else {
      return right_->TagTree(left_->TagTree(tag));
    }
  }

  size_t FindBucket(Vector& query) {
    DEBUG_ASSERT_MSG(query.length() == max_vals_.size(),
		     "dim = %zu"d, maxval size= %zu"d"
		     ", sl=%zu"d",
		     query.length(), max_vals_.size(),
		     subtree_leaves_);
    DEBUG_ASSERT(query.length() == min_vals_.size());

    if (subtree_leaves_ == 1) { // if leaf
      // if (printer)
      // printf ("%lg,%lg,%lg \n", ratio_, range, ratio_ / range);
      return bucket_tag_;
    } else if (query[split_dim_] <= split_value_) { // if left subtree
      // go to left child
      return left_->FindBucket(query);
    } else { // if right subtree
      // go to right child
      return right_->FindBucket(query);
    } // end if-else
  }

  void ComputeVariableImportance(ArrayList<long double> *imps) {
    if (subtree_leaves_ == 1) {
      // if leaf, do nothing
      return;
    } else {
      // compute the improvement in error because of the 
      // split
      long double error_improv = error_ - (left_->error() + right_->error());
      (*imps)[split_dim_] += error_improv;
      left_->ComputeVariableImportance(imps);
      right_->ComputeVariableImportance(imps);
      return;
    }
  }
  
}; // Class TTree

#endif
