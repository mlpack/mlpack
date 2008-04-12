/*
 * =====================================================================================
 * 
 *       Filename:  dual_manifold_engine_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  03/18/2008 11:34:02 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

template<typename OptimizedFunction>
void DualManifoldEngine<OptimizedFunction>::Init(datanode *module, 
    // index pairs to consider from the matrix (row,column) pairs
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider, 
    // The values of the (row, column) values, also known as the dot products
    ArrayList<double> &dot_prod_values) {

  module_=module;
  l_bfgs1_.Init(&optimized_function1_, fx_param_node(module_, "l_bfgs"));
  l_bfgs2_.Init(&optimized_function2_, fx_param_node(module_, "l_bfgs"));

  optimized_function1_.Init(fx_param_node(module_, "opt1"), 
      l_bfgs1_.coordinates(), 
      pairs_to_consider,
      dot_prod_values);
  for(index_t i=0; i<pairs_to_consider.size(); i++) {
    std::swap(pairs_to_consider[i].first, pairs_to_consider[i].second);
  }
  optimized_function2_.Init(fx_param_node(module_, "opt2"), 
      l_bfgs1_.coordinates(), 
      pairs_to_consider,
      dot_prod_values);
  
}

template<typename OptimizedFunction>
void DualManifoldEngine<OptimizedFunction>::Destruct() {

}

template<typename OptimizedFunction>
void DualManifoldEngine<OptimizedFunction>::ComputeLocalOptimum() {
  for(index_t i=0; i<max_iterations_; i++) {
    l_bfgs1_.ComputeLocalOptimumBFGS();
    l_bfgs1_.Reset();
    l_bfgs2_.ComputeLocalOptimumBFGS();
    l_bfgs2_.Reset();
    double error;
    optimized_function1_.ComputeFeasibilityError(l_bfgs1_.coordinates(), &error);
    if (error<desired_error_) {
      break;
    }
  } 
}

