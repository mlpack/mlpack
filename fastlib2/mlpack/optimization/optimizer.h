/*
 * =====================================================================================
 * 
 *       Filename:  optimizer.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  11/10/2008 03:10:29 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_
#include "fastlib/fastlib.h"
#include "NFL.h"
#include "OptQNewton.h"
namespace optim {
template <typename METHOD, typename Objective, typename Constraints>
class Optimizer {
 public:
   void Init(Method *method, Objective *objective) {
     method_    = method;
     objective_ = objective;
   }
   void Optimize() {
   
   }
 
 private:
  Objective *objective_;
  METHOD *method_; 
  index_t dimension_;
  
  void Initialize(int ndim, NEWMAT::ColumnVector &x) {
    DEBUG_ASSERT(ndim==dimension_);
    Vector vec;  
    vec->Alias(x.data(), dimension_);
    objective_->GiveInit(&vec);
    
  }
  void ComputeObjective(int ndim, NEWMAT::ColumnVector &x, 
      double &fx, int &result) {
    DEBUG_ASSERT(ndim==dimension_);
    Vector vec;  
    vec->Alias(x.data(), dimension_);
    objective_->ComputeObjective(vec, &fx);
  };
  
  void ComputeObjective(int mode, int ndim, NEWMAT::ColumnVector &x, 
      double &fx, NEWMAT::ColumnVector &gx, int &result) {
    DEBUG_ASSERT(ndim==dimension_);
    Vector vecx;  
    vecx->Alias(x.data(), dimension_);
    objective_->ComputeObjective(vecx, &fx);
    Vector vecg;
    vecg->Alias(gx.data(), dimension_);
    objective_->ComputeGradient(vecx, &vec)g;   
  }
    
  void ComputeObjective(int mode, int ndim, NEWMAT::ColumnVector &x, 
      double &fx, NEWMAT::ColumnVector &gx, 
    NEWMAT::SymmetricMatrix &hx, int &result) {
    DEBUG_ASSERT(ndim==dimension_);
    Vector vecx;  
    vecx->Alias(x.data(), dimension_);
    objective_->ComputeObjective(vec, &fx);
    Vector vecg;
    vecg->Alias(gx.data(), dimension_);
    objective_->ComputeGradient(vecx, &vecg); 
    Matrix hessian;
    hessian.Alias(hx.data(), dimension_, dimension_);  
    objective_->ComputeHessian(vecx, hessian);  
  }
 
};
  
};
#endif
