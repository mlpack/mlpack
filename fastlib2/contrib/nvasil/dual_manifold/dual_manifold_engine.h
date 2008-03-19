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
#include "dual_manifold_objective.h"
#include "../l_bfgs/l_bfgs.h"

template<typename OptimizedFunction>
class DualManifoldEngine {
 public:
  void Init(datanode *module);
  void Destruct();
    
  
 private:
  datanode *module;
  LBfgs<OptimizedFunction> lbfgs1_;
  LBfgs<OptimizedFunction> lbfgs2_;
  OptimizedFunction optimized_function1_;
  OptimizedFunction optimized_function2_;

};

#include "dual_manifold_engine_impl.h"
#endif // DUAL_MANIFOLD_ENGINE_
