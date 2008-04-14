/*
 * =====================================================================================
 * 
 *       Filename:  dual_manifold_engine.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/18/2008 11:08:05 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef DUAL_MANIFOLD_ENGINE_
#define DUAL_MANIFOLD_ENGINE_
#include "../l_bfgs/l_bfgs.h"
#include "../l_bfgs/optimization_utils.h"
/**
 * DualManifoldEngine treats the problem of non-linear, non negative matrix factorization
 * The classic problem of NMF is (\f& D \simeq WH\f&)
 * For large scale problems D is sparse and it has a lot of zeros that represent
 * missing or don't care data. This algorithm solves the following set of optimization
 * problems:
 * (\f$ \max trHH^T subject to A_i \bullet D = A_i \bullet (WH))
 * (\f$ \max trWW^T subject to A_i \bullet D = A_i \bullet (WH))
 * where (\f$\bullet \f$) is the matrix dot product
 * (\f$ A_i \f$) is a selection matrix that selects the non-zero elements of (\f$D\f$)
 * The matrix (\f& D \f&) can be real or non-negative.
 * We can also restrict (\f$ W, H \f$) to be non-negative or sparse
 * This depends on the definition of the OptimizedFunction
 */
template<typename OptimizedFunction>
class DualManifoldEngine {
 public:
  /**
   * pairs_to_consider are (row,column) indices from a given sparse D
   * matrix, These are the elements that we care about in our factorization and 
   * dot_prod_values are the values that we are trying to match
   * 
   */
  void Init(datanode *module, 
    // index pairs to consider from the matrix (row,column) pairs
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider, 
    // The values of the (row, column) values, also known as the dot products
    ArrayList<double> &dot_prod_values);
  void Destruct();
  void ComputeLocalOptimum(); 
 
 private:
  double feasibility_tolerance_;
  double desired_feasibility_;
  double norm_grad_tolerance_;
  index_t iterations_;
  index_t max_iterations_;
  double desired_error_;
  datanode *module_;
  index_t num_of_components_;
    
  LBfgs<OptimizedFunction> l_bfgs1_;
  LBfgs<OptimizedFunction> l_bfgs2_;
  OptimizedFunction optimized_function1_;
  OptimizedFunction optimized_function2_;
};

#include "dual_manifold_engine_impl.h"
#endif // DUAL_MANIFOLD_ENGINE_
