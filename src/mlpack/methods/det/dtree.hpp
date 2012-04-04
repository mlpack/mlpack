/**
 * @file dtree.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Density Tree class
 *
 */

#ifndef DTREE_HPP
#define DTREE_HPP

#include <assert.h>
#include <vector>

#include <mlpack/core.hpp>

using namespace mlpack;
using namespace std;


// This two types in the template are used 
// for two purposes:
// eT - the type to store the data in (for most practical 
// purposes, storing the data as a float suffices).
// cT - the type to perform computations in (computations 
// like computing the error, the volume of the node etc.). 
// For high dimensional data, it might be possible that the 
// computation might overflow, so you should use either 
// normalize your data in the (-1, 1) hypercube or use 
// long double or modify this code to perform computations
// using logarithms.
template<typename eT = float,
	 typename cT = long double>
class DTree{
 
  ////////////////////// Member Variables /////////////////////////////////////
  
 private:

  typedef arma::Mat<eT> MatType;
  typedef arma::Col<eT> VecType;
  typedef arma::Row<eT> RowVecType;


  // The indices in the complete set of points
  // (after all forms of swapping in the original data
  // matrix to align all the points in a node 
  // consecutively in the matrix. The 'old_from_new' array 
  // maps the points back to their original indices.
  size_t start_, end_;
  
  // The split dim for this node
  size_t split_dim_;

  // The split val on that dim
  eT split_value_;

  // L2-error of the node
  cT error_;

  // sum of the error of the leaves of the subtree
  cT subtree_leaves_error_;

  // number of leaves of the subtree
  size_t subtree_leaves_;

  // flag to indicate if this is the root node
  // used to check whether the query point is 
  // within the range
  bool root_;

  // ratio of number of points in the node to the 
  // total number of points (|t| / N)
  cT ratio_;

  // the inverse of  volume of the node
  cT v_t_inv_;

  // sum of the reciprocal of the inverse v_ts
  // the leaves of this subtree
  cT subtree_leaves_v_t_inv_;

  // since we are using uniform density, we need
  // the max and min of every dimension for every node
  VecType* max_vals_;
  VecType* min_vals_;

  // the tag for the leaf used for hashing points
  int bucket_tag_;

  // The children
  DTree<eT, cT> *left_;
  DTree<eT, cT> *right_;

  ////////////////////// Constructors /////////////////////////////////////////

public: 

  ////////////////////// Getters and Setters //////////////////////////////////
  size_t start() { return start_; }

  size_t end() { return end_; }

  size_t split_dim() { return split_dim_; }

  eT split_value() { return split_value_; }

  cT error() { return error_; }

  cT subtree_leaves_error() { return subtree_leaves_error_; }

  size_t subtree_leaves() { return subtree_leaves_; }

  cT ratio() { return ratio_; }

  cT v_t_inv() { return v_t_inv_; }

  cT subtree_leaves_v_t_inv() { return subtree_leaves_v_t_inv_; }

  DTree<eT, cT>* left() { return left_; }
  DTree<eT, cT>* right() { return right_; }

  bool root() { return root_; }

  ////////////////////// Private Functions ////////////////////////////////////
 private:

  cT ComputeNodeError_(size_t total_points);
  
  bool FindSplit_(MatType* data,
		  size_t* split_dim,
		  size_t* split_ind,
		  cT* left_error, 
		  cT* right_error,
		  size_t maxLeafSize = 10,
		  size_t minLeafSize = 5);

  void SplitData_(MatType* data,
		  size_t split_dim,
		  size_t split_ind,
		  arma::Col<size_t>* old_from_new, 
		  eT* split_val,
		  eT* lsplit_val,
		  eT* rsplit_val);

  void GetMaxMinVals_(MatType* data,
		      VecType* max_vals,
		      VecType* min_vals);

  bool WithinRange_(VecType* query);

  ///////////////////// Public Functions //////////////////////////////////////
 public:
  
  DTree();

  // Root node initializer
  // with the bounding box of the data
  // it contains instead of just the data.
  DTree(VecType* max_vals, 
	VecType* min_vals,
	size_t total_points);

  // Root node initializer
  // with the data, no bounding box.
  DTree(MatType* data);

  // Non-root node initializers
  DTree(VecType* max_vals, 
	VecType* min_vals,
	size_t start,
	size_t end,
	cT error);

  DTree(VecType* max_vals, 
	VecType* min_vals,
	size_t total_points,
	size_t start,
	size_t end);

  ~DTree();

  // Greedily expand the tree
  cT Grow(MatType* data, 
	  arma::Col<size_t> *old_from_new,
	  bool useVolReg = false,
	  size_t maxLeafSize = 10,
	  size_t minLeafSize = 5);

  // perform alpha pruning on the tree
  cT PruneAndUpdate(cT old_alpha,
		    bool useVolReg = false);

  // compute the density at a given point
  cT ComputeValue(VecType* query);

  // print the tree (in a DFS manner)
  void WriteTree(size_t level, FILE *fp);

  // indexing the buckets for possible usage later
  int TagTree(int tag);

  // This is used to generate the class membership
  // of a learned tree.
  int FindBucket(VecType* query);

  // This computes the variable importance list 
  // for the learned tree.
  void ComputeVariableImportance(arma::Col<double> *imps);

  // A public function to test the private functions
  bool TestPrivateFunctions() {


    bool return_flag = true;

    // Create data
    MatType test_data(3,5);

    test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	      << 5 << 0 << 1 << 7 << 1 << arma::endr
	      << 5 << 6 << 7 << 1 << 8 << arma::endr;

    // save current data
    size_t true_start = start_, true_end = end_;
    VecType* true_max_vals = max_vals_;
    VecType* true_min_vals = min_vals_;
    cT true_error = error_;


    // Test GetMaxMinVals_
    min_vals_ = NULL;
    max_vals_ = NULL;
    max_vals_ = new VecType();
    min_vals_ = new VecType();

    GetMaxMinVals_(&test_data, max_vals_, min_vals_);

    if ((*max_vals_)[0] != 7 || (*min_vals_)[0] != 3) {
      Log::Warn << "Test: GetMaxMinVals_ failed." << endl;
      return_flag =  false;
    }

    if ((*max_vals_)[1] != 7 || (*min_vals_)[1] != 0) {
      Log::Warn << "Test: GetMaxMinVals_ failed." << endl;
      return_flag =  false;
    }

    if ((*max_vals_)[2] != 8 || (*min_vals_)[2] != 1) {
      Log::Warn << "Test: GetMaxMinVals_ failed." << endl;
      return_flag =  false;
    }

    // Test ComputeNodeError_
    start_ = 0; 
    end_ = 5;
    cT node_error = ComputeNodeError_(5);
    cT log_vol = (cT) std::log(4) + (cT) std::log(7) + (cT) std::log(7);
    cT true_node_error = -1.0 * std::exp(-log_vol);

    if (std::abs(node_error - true_node_error) > 1e-7) {
      Log::Warn << "Test: True error : " << true_node_error
		<< ", Computed error: " << node_error
		<< ", diff: " << std::abs(node_error - true_node_error)
		<< endl;
      return_flag =  false;
    }

    start_ = 3; 
    end_ = 5;
    node_error = ComputeNodeError_(5);
    true_node_error = -1.0 * std::exp(2 * std::log((cT) 2 / (cT) 5) - log_vol);

    if (std::abs(node_error - true_node_error) > 1e-7) {
      Log::Warn << "Test: True error : " << true_node_error
		<< ", Computed error: " << node_error
		<< ", diff: " << std::abs(node_error - true_node_error)
		<< endl;
      return_flag =  false;
    }

    // Test WithinRange_

    VecType test_query(3);
    test_query << 4.5 << 2.5 << 2;

    if (!WithinRange_(&test_query)) {
      Log::Warn << "Test: WithinRange_ failed" << endl;
      return_flag =  false;
    }

    test_query << 8.5 << 2.5 << 2;

    if (WithinRange_(&test_query)) {
      Log::Warn << "Test: WithinRange_ failed" << endl;
      return_flag =  false;
    }

    // Test FindSplit_
    start_ = 0;
    end_ = 5;
    error_ = ComputeNodeError_(5);

    size_t ob_dim, true_dim, ob_ind, true_ind;
    cT true_left_error, ob_left_error, true_right_error, ob_right_error;

    true_dim = 2;
    true_ind = 1;
    true_left_error = -1.0 * std::exp(2 * std::log((cT) 2 / (cT) 5) 
				      - (std::log((cT) 7) + std::log((cT) 4)
					 + std::log((cT) 4.5)));
    true_right_error =  -1.0 * std::exp(2 * std::log((cT) 3 / (cT) 5) 
				      - (std::log((cT) 7) + std::log((cT) 4)
					 + std::log((cT) 2.5)));

    if(!FindSplit_(&test_data, &ob_dim, &ob_ind, 
		   &ob_left_error, &ob_right_error, 2, 1)) {
      Log::Warn << "Test: FindSplit_ returns false." << endl;
      return_flag =  false;
    }

    if (true_dim != ob_dim) {
      Log::Warn << "Test: FindSplit_ - True dim: " << true_dim
		<< ", Obtained dim: " << ob_dim << endl;
      return_flag =  false;
    }

    if (true_ind != ob_ind) {
      Log::Warn << "Test: FindSplit_ - True ind: " << true_ind
		<< ", Obtained ind: " << ob_ind << endl;
      return_flag =  false;
    }

    if (std::abs(true_left_error - ob_left_error) > 1e-7) {
      Log::Warn << "Test: FindSplit_ - True left_error: " << true_left_error
		<< ", Obtained left_error: " << ob_left_error 
		<< ", diff: " << std::abs(true_left_error - ob_left_error)
		<< endl;
      return_flag =  false;
    }

    if (std::abs(true_right_error - ob_right_error) > 1e-7) {
      Log::Warn << "Test: FindSplit_ - True right_error: " << true_right_error
		<< ", Obtained right_error: " << ob_right_error 
		<< ", diff: " << std::abs(true_right_error - ob_right_error)
		<< endl;
      return_flag =  false;
    }

    // Test SplitData_
    MatType split_test_data(test_data);
    arma::Col<size_t> o_test(5);
    o_test << 1 << 2 << 3 << 4 << 5;

    start_ = 0;
    end_ = 5;
    size_t split_dim = 2, split_ind = 1;
    eT true_split_val, ob_split_val, true_lsplit_val, ob_lsplit_val,
      true_rsplit_val, ob_rsplit_val;

    true_lsplit_val = 5;
    true_rsplit_val = 6;
    true_split_val = (true_lsplit_val + true_rsplit_val) / 2;

    SplitData_(&split_test_data, split_dim, split_ind, 
	       &o_test, &ob_split_val, 
	       &ob_lsplit_val, &ob_rsplit_val);

    if (o_test[0] != 1 || o_test[1] != 4 || o_test[2] != 3 
	|| o_test[3] != 2 || o_test[4] != 5) {
      Log::Warn << "Test: SplitData_ - OFW should be 1,4,3,2,5"
		<< ", is " << o_test.t();
      return_flag =  false;
    }

    if (true_split_val != ob_split_val) {
      Log::Warn << "Test: SplitData_ - True split val: " << true_split_val
		<< ", Ob split val: " << ob_split_val << endl;
      return_flag =  false;
    }

    if (true_lsplit_val != ob_lsplit_val) {
      Log::Warn << "Test: SplitData_ - True lsplit val: " << true_lsplit_val
		<< ", Ob lsplit val: " << ob_lsplit_val << endl;
      return_flag =  false;
    }

    if (true_rsplit_val != ob_rsplit_val) {
      Log::Warn << "Test: SplitData_ - True rsplit val: " << true_rsplit_val
		<< ", Ob rsplit val: " << ob_rsplit_val << endl;
      return_flag =  false;
    }


    // restore original values
    delete max_vals_;
    delete min_vals_;
    max_vals_ = true_max_vals;
    min_vals_ = true_min_vals;
    start_ = true_start;
    end_ = true_end;
    error_ = true_error;

    return return_flag;

  } // TestPrivateFunctions
  
}; // Class DTree

#include "dtree_impl.hpp"

#endif
