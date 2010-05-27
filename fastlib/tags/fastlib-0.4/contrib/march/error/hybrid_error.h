/**
 * @file hybrid_error.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Contains a class for running timing tests for hybrid error schemes.
 */

#ifndef HYBRID_ERROR_H
#define HYBRID_ERROR_H

#include "fastlib/fastlib.h"


/*
 * I need a templated error class.  It needs to:
 * - return the correct error bound - relative, absolute, hybrid
 * - Be able to inform the tree building so I get the right stats for each kind
 *   of error
 *
 * It might be possible to just templatize with the Stat class.  It can have a 
 * function to compute the allowed error
 *
 * TODO: write a naive version too, to evaluate the approximations against
 */

/**
 * Dual-tree implementation of a Gaussian kernel summation with monopole 
 * error bounds.
 *
 * TODO: I think this should be templatized to allow for different error styles
 * I could templatize with the stat class, which has a function to determine if 
 * pruning is possible.  
 */ 
template <typename TErrorStat>
class GaussianKernelErrorTester {

  //FORBID_ACCIDENTAL_COPY(GaussianKernelErrorTester);
public:
   
  GaussianKernelErrorTester() {}
  
  ~GaussianKernelErrorTester() {}
  
typedef BinarySpaceTree<DHrectBound<2>, Matrix, TErrorStat> ErrorTree;

 private:

  // The common, global bandwidth of the Gaussians
  double bandwidth_;
  
  // The centers of the Gaussians
  Matrix centers_;
  
  // The result of the sum
  Vector results_;
  
  // The fx module for timing
  struct datanode* module_;
  
  // The total number of points
  index_t num_points_;
  
  // The dimension
  index_t dimension_;
  
  ErrorTree* tree_;
  
  ArrayList<index_t> old_from_new_;
  
  double max_error_;
  double min_error_;
  double steepness_;
  
  index_t num_prunes_;
  
  
  /**
   * Computes the value of the gaussian centered at r at the point referred to 
   * by q.
   *
   * I don't think I'll worry about normalization for now.  
   */
  double ComputeGaussian_(index_t q, index_t r) {
    
    Vector q_vec;
    centers_.MakeColumnVector(q, &q_vec);
    
    Vector r_vec;
    centers_.MakeColumnVector(r, &r_vec);
    
    double dist = la::DistanceSqEuclidean(q_vec, r_vec);
    
    return(exp(-bandwidth_ * dist));
    
  } // ComputeGaussian_()
  
  double ComputeGaussian_(double dist_sq) {
  
    return (exp(-bandwidth_ * dist_sq));
  
  } // ComputeGaussian_
  
  // This function needs to decrease the stat's query_count, since I don't need
  // to allocate any more error to those query points 
  void ComputeSumBaseCase_(ErrorTree* query, ErrorTree* reference) {
    
    double min_query_val = DBL_MAX;
    
    for (index_t query_index = query->begin(); query_index < query->end(); 
         query_index++) {
      
      double query_value = results_[query_index];
      
      for (index_t ref_index = reference->begin(); ref_index < reference->end(); 
           ref_index++) {
        
        query_value = query_value + ComputeGaussian_(query_index, ref_index);
        
      }
      
      DEBUG_ASSERT(query_value >= 0.0);
      
      results_[query_index] = query_value;
      if (query_value < min_query_val) {
        min_query_val = query_value;
      }
      
    }
    
  query->stat().set_query_lower_bound(min_query_val);
    
    /*index_t remainder = query->stat().remaining_references();
    index_t reference_count = reference->count();
    DEBUG_ASSERT(remainder - reference_count >= 0);
    query->stat().set_remaining_references(remainder - reference_count);
    */
  } // ComputeSumBaseCase_()
  
  void ComputeSumRecursion_(ErrorTree* query, ErrorTree* reference) {
    
    double q_min_dist = query->bound().MinDistanceSq(reference->bound());
    double q_max_dist = query->bound().MaxDistanceSq(reference->bound());
    
    double q_upper_bound = ComputeGaussian_(q_min_dist) * reference->count();
    double q_lower_bound = ComputeGaussian_(q_max_dist) * reference->count();
    
   
    if(query->stat().CanPrune(
                q_upper_bound, q_lower_bound, reference->count())) {
      
      num_prunes_++;
      
      double approximate_result = 0.5 * (q_upper_bound + q_lower_bound);
      DEBUG_ASSERT(approximate_result >= 0.0);
      
      Vector subvec;
      results_.MakeSubvector(query->begin(), query->count(), &subvec);
      
      Vector approx;
      approx.Init(query->count());
      approx.SetAll(approximate_result);

      la::AddTo(approx, &subvec);

      query->stat().set_query_lower_bound(
          query->stat().query_lower_bound() + approximate_result);

    } // Pruning case
    else if (query->is_leaf() && reference->is_leaf()) {
      
      ComputeSumBaseCase_(query, reference);
      
    } // Base case
    else if(query->is_leaf()) {
      
      ComputeSumRecursion_(query, reference->left());
      ComputeSumRecursion_(query, reference->right());
      
    } // only split references
    else if(reference->is_leaf()) {
      
      // How to propagate lower bound down the tree?
      // Will try parent's bound
      // it's loose on one (both?) children, but I don't have a better idea
      
      query->left()->stat().set_query_lower_bound(
          query->stat().query_lower_bound());
      query->right()->stat().set_query_lower_bound(
          query->stat().query_lower_bound());
     
      ComputeSumRecursion_(query->left(), reference);
      ComputeSumRecursion_(query->right(), reference);
      
      query->stat().set_query_lower_bound(
          min(query->left()->stat().query_lower_bound(), 
              query->right()->stat().query_lower_bound()));
              
      DEBUG_ASSERT(query->stat().query_lower_bound() >= 0.0);
      
    } // only split queries
    else {
      
      
      query->left()->stat().set_query_lower_bound(
          query->stat().query_lower_bound());
      query->right()->stat().set_query_lower_bound(
          query->stat().query_lower_bound());      
      
      ComputeSumRecursion_(query->left(), reference->left());
      ComputeSumRecursion_(query->left(), reference->right());
      
      ComputeSumRecursion_(query->right(), reference->left());
      ComputeSumRecursion_(query->right(), reference->right());
      
      query->stat().set_query_lower_bound(
          min(query->left()->stat().query_lower_bound(), 
              query->right()->stat().query_lower_bound()));
      
      DEBUG_ASSERT(query->stat().query_lower_bound() >= 0.0);

    } // four-way
    
  } // ComputeSumRecursion_
  
public:

  void InitStats(ErrorTree* node) {
    
    if (!(node->is_leaf())) {
      
      InitStats(node->left());
      InitStats(node->right());
      
    }
    
    node->stat().SetParams(max_error_, min_error_, steepness_);
    node->stat().set_remaining_references(num_points_);
    // This is only true for these unit coefficient Gaussians
    //node->stat().set_query_upper_bound(num_points_);
    node->stat().set_query_lower_bound(0.0);
    
  } // InitStats()

  
  void Init(struct datanode* mod, const Matrix& cent, double band, 
            double max_err, double min_err, double steep) {
    
    module_ = mod;
    
    centers_.Copy(cent);
    
    bandwidth_ = band;
    DEBUG_ASSERT(bandwidth_ > 0.0);
    
    max_error_ = max_err;
    DEBUG_ASSERT(max_error_ > 0.0);
    
    min_error_ = min_err;
    DEBUG_ASSERT(min_error_ > 0.0);
    
    steepness_ = steep;
    DEBUG_ASSERT(steepness_ > 0.0);
    
    num_points_ = centers_.n_cols();
    
    dimension_ = centers_.n_rows();
    
    num_prunes_ = 0;
    
    results_.Init(num_points_);
    results_.SetZero();
    
    tree_ = tree::MakeKdTreeMidpoint<ErrorTree>(centers_, 
                fx_param_int(mod, "leaf_size", 20), &old_from_new_, NULL);
                
    //tree_->stat().set_query_count(num_points_);
    
    InitStats(tree_);
            
  } // Init()
  
  void ComputeTotalSum(Vector* results_vec) {
    
    fx_timer_start(module_, "summation");
    ComputeSumRecursion_(tree_, tree_);
    fx_timer_stop(module_, "summation");
    
    results_vec->Init(num_points_);
    
    // unpermute the results for analysis
    for (index_t i = 0; i < num_points_; i++) {
    
      (*results_vec)[old_from_new_[i]] = results_[i];
    
    } // i
    
   // ot::Print(*results_vec);
    
    fx_format_result(module_, "num_prunes", "%d", num_prunes_);
    fx_format_result(module_, "num_points", "%d", num_points_);
    
    
  } // ComputeTotalSum()

  
}; // class GaussianKernelErrorTester



#endif