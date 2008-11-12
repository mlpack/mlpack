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
#ifndef HAVE_STD
#define HAVE_STD
#endif

#ifndef HAVE_NAMESPACES
#define HAVE_NAMESPACES
#endif

#include "fastlib/fastlib.h"
#include "opt++/include/newmat.h"
#include "opt++/include/NLF.h"
#include "opt++/include/OptCG.h"
#include "opt++/include/OptLBFGS.h"
#include "opt++/include/OptFDNewton.h"
#include "opt++/include/OptNewton.h"
#include "opt++/include/OptQNewton.h"


namespace optim {

typedef OPTPP::OptCG       CG;
typedef OPTPP::OptQNewton  QNewton;
typedef OPTPP::OptQNewton  BFGS;
typedef OPTPP::OptFDNewton FDNewton;
typedef OPTPP::OptNewton   Newton;
typedef OPTPP::OptLBFGS       LBFGS;


// The default is good for CG and L-BFGS
template<typename Method>
class OptimizationTrait {
 public:
  typedef OPTPP::NLF1 NlpType;
  static void InitializeMethod(fx_module *module, Method *method) {
    std::string search_strategy = fx_param_str(module, 
                                      "search_strategy",
                                      "line_search");
    if (search_strategy=="line_search") {
      method->setSearchStrategy(OPTPP::LineSearch);
    } else {
      if (search_strategy=="trust_region") {
        method->setSearchStrategy(OPTPP::TrustRegion);
      } else {
        if (search_strategy=="trust_pds") {
          method->setSearchStrategy(OPTPP::TrustPDS);
        } else {
          FATAL("The specified search strategy %s is not supported", 
              search_strategy.c_str());
        }
      }
    } 
    if (method->checkDeriv()==false) {
      NONFATAL("Warning finite difference derivative/hessian doesn't much "
          "analytic");
    }    
  }
};

// Generic Newton 
template<>
class OptimizationTrait<OPTPP::OptNewtonLike> {
 public:
  typedef OPTPP::NLF1 NlpType;
  static void InitializeMethod(fx_module *module, 
      OPTPP::OptNewtonLike *method) {
     std::string search_strategy = fx_param_str(module, 
                                      "search_strategy",
                                      "line_search");
    if (search_strategy=="line_search") {
      method->setSearchStrategy(OPTPP::LineSearch);
    } else {
      if (search_strategy=="trust_region") {
        method->setSearchStrategy(OPTPP::TrustRegion);
        method->setTRSize(fx_param_double(module, "trust_region_size", 
              method->getTRSize()));
      } else {
        if (search_strategy=="trust_pds") {
          method->setSearchStrategy(OPTPP::TrustPDS);
        } else {
          FATAL("The specified search strategy %s is not supported", 
              search_strategy.c_str());
        }
      }
    } 
    if (method->checkDeriv()==false) {
      NONFATAL("Warning finite difference derivative/hessian doesn't much "
         " analytic");
    }    
  }
};

// Quasi-Newton with Finite Difference approximation on the Hessian
template<>
class OptimizationTrait<OPTPP::OptFDNewton> {
 public:
  typedef OPTPP::NLF1 NlpType;
  static void InitializeMethod(fx_module *module, OPTPP::OptFDNewton *method) {
    OptimizationTrait<OPTPP::OptNewtonLike>::InitializeMethod(
        module, method);  
  }
};

// Quasi-Newton BFGS 
template<>
class OptimizationTrait<OPTPP::OptQNewton> {
 public:
  typedef OPTPP::NLF1 NlpType;
  static void InitializeMethod(fx_module *module, OPTPP::OptFDNewton *method) {
    OptimizationTrait<OPTPP::OptNewtonLike>::InitializeMethod(
        module, method);  
  }
};


// Newton  with analytic expression for the Hessian
template<>
class OptimizationTrait<OPTPP::OptNewton> {
 public:
  typedef OPTPP::NLF2 NlpType;
  static void InitializeMethod(fx_module *module, OPTPP::OptNewton *method) {
   OptimizationTrait<OPTPP::OptNewtonLike>::InitializeMethod(
        module, method);  
 
  }
};



template <typename Method, typename Objective>
class StaticUnconstrainedOptimizer {
 public:
  StaticUnconstrainedOptimizer() {
    objective_=NULL;
    method_=NULL;
  }
  ~StaticUnconstrainedOptimizer() {
    if (method_==NULL) {
      delete method_;
    }
  }
  void Init(fx_module *module, Objective *objective) {
    module_ = module;
    objective_ = objective;
    dimension_=objective_->dimension();
    typename OptimizationTrait<Method>::NlpType nlp(dimension_, ComputeObjective, 
        Initialize, (OPTPP::CompoundConstraint *)NULL);
    // setting some generic options for the optimization method
    const char *status_file=fx_param_str(module_, "status_file", "status.txt" );
    if (method_->setOutputFile(status_file, 0)==false) {
      FATAL("Failed to open the status file %s for writing", status_file);
    }
    // Setup the tolerances 
    OPTPP::TOLS tols;
    tols.setDefaultTol();
    if (fx_param_exists(module_, "function_tol")) {
      tols.setFTol(fx_param_double_req(module_, "function_tol"));
    } else {
      if (fx_param_exists(module_, "step_tol")) {
        tols.setStepTol(fx_param_double_req(module_, "step_tol"));
      } else {
        if (fx_param_exists(module_, "grad_tol")) {
          tols.setGTol(fx_param_double_req(module_, "grad_tol"));
        } 
      }
    }
    if (fx_param_exists(module_, "max_step")) {
      tols.setMaxStep(fx_param_double_req(module_, "max_step"));
    }
    if (fx_param_exists(module_, "min_step")) {
      tols.setMinStep(fx_param_double_req(module_, "min_step"));
    }
    if (fx_param_exists(module_, "line_search_tol")) {
      tols.setLSTol(fx_param_double_req(module_, "min_step"));
    }
  
    tols.setMaxIter(fx_param_int(module_, "max_iter", 1000));
    method_ = new Method(&nlp, tols);

    OptimizationTrait<Method>::InitializeMethod(module_, method_);
    
  }
  void Destruct() {
    if (method_==NULL) {
      delete method_;
    }
  }
  void Optimize(Vector *result) {
    method_->optimize();
    NEWMAT::ColumnVector x=method_->getXPrev();
    result->Copy(x.data(), dimension_); 
    fx_result_bool(module_, "converged", method_->checkConvg());
    fx_result_int(module_, "iterations", method_->getIter());
    method_->cleanup();
  }
 
 private:
  fx_module *module_;
  static Objective *objective_;
  Method *method_; 
  index_t dimension_;
  
  static void Initialize(int ndim, NEWMAT::ColumnVector &x) {
    Vector vec;  
    vec.Alias(x.data(), ndim);
    objective_->GiveInit(&vec);
    
  }
  static void ComputeObjective(int ndim, const NEWMAT::ColumnVector &x, 
      double &fx, int &result) {
    Vector vec;  
    vec.Alias(x.data(), ndim);
    objective_->ComputeObjective(vec, &fx);
    result = OPTPP::NLPFunction;
  };
  
  static void ComputeObjective(int mode, int ndim, const NEWMAT::ColumnVector &x, 
      double &fx, NEWMAT::ColumnVector &gx, int &result) {
    Vector vecx;  
    vecx.Alias((double *)x.data(), ndim);
    if (mode & OPTPP::NLPFunction) {
      objective_->ComputeObjective(vecx, &fx);
      result = OPTPP::NLPFunction;
    } else {
      if (mode & OPTPP::NLPGradient) {
        Vector vecg;
        vecg.Alias(gx.data(), ndim);
        objective_->ComputeGradient(vecx, &vecg);
        result = OPTPP::NLPGradient;
      }
    }   
  }
    
  static void ComputeObjective(int mode, int ndim, const NEWMAT::ColumnVector &x, 
      double &fx, NEWMAT::ColumnVector &gx, 
    NEWMAT::SymmetricMatrix &hx, int &result) {
    Vector vecx;  
    vecx.Alias(x.data(), ndim);
    if (mode & OPTPP::NLPFunction) {
      objective_->ComputeObjective(vecx, &fx);
      result = OPTPP::NLPFunction;
    } else {
      if (mode & OPTPP::NLPGradient) {
        Vector vecg;
        vecg.Alias(gx.data(), ndim);
        objective_->ComputeGradient(vecx, &vecg); 
        result = OPTPP::NLPGradient;
      } else {
        if (mode & OPTPP::NLPHessian) {
          Matrix hessian;
          hessian.Alias(hx.data(), ndim, ndim);  
          objective_->ComputeHessian(vecx, hessian);  
          result = OPTPP::NLPHessian;
        }
      }
    }
  } 
};
  
}; // Namespace optim
#endif
