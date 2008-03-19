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
DualManifoldEngine<OptimizedFunction>::Init(datanode *module, 
    ArrayList<std::pair<index_t, index_t> > &pairs_to_consider,
    ArrayList<double> &dot_prod_values) {

  module_=module;
  lbfg1_.Init(&optimized_function1_, fx_param_node(module_, "lbfgs"));
  lbfg2_.Init(&optimized_function2_, fx_param_node(module_, "lbfgs"));

  optimized_function1_.Init(fx_param_node(module_, "opt1"), 
      lbfgs1_.coordinates(), 
      pairs_to_consider,
      dot_prod_values);
  for(index_t i=0; i<pairs_to_consider.size(); i++) {
    std::swap(pairs_to_consider[i].first, pairs_to_consider[i].second);
  }
  optimized_function2_.Init(fx_param_node(module_, "opt2"), 
      lbfgs1_.coordinates(), 
      pairs_to_consider,
      dot_prod_values);
  
}

