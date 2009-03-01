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
#include "include/newmat.h"
#include "include/NLF.h"
#include "include/OptCG.h"
#include "include/OptLBFGS.h"
#include "include/OptFDNewton.h"
#include "include/OptNewton.h"
#include "include/OptQNewton.h"
#include "include/BoundConstraint.h"
#include "include/CompoundConstraint.h"
#include "include/LinearEquation.h"
#include "include/LinearInequality.h"
#include "include/NonLinearEquation.h"
#include "include/NonLinearInequality.h"


namespace optim {

typedef OPTPP::OptCG       CG;
typedef OPTPP::OptQNewton  QNewton;
typedef OPTPP::OptQNewton  BFGS;
typedef OPTPP::OptFDNewton FDNewton;
typedef OPTPP::OptNewton   Newton;
typedef OPTPP::OptLBFGS    LBFGS;

enum ConstraintType {
  NoConstraint       = 0,
  BoundConstraint    = 2,
  LinearEquality     = 4,
  LinearInequality   = 8,
  NonLinearEquality  = 16,
  NonLinearInequality= 32
};

template <typename Method, 
         typename Objective, 
         ConstraintType Constraint=NoConstraint>
class StaticOptppOptimizer;

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



template<typename Method, typename Objective, bool Applicable>
class BoundConstraintTrait {
 public:
  static void UpdateConstraints(Objective *object, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {  
  }
};

template<typename Method, typename Objective>
class BoundConstraintTrait<Method, Objective, true> {
 public:
  static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
    index_t dimension = objective->dimension();
    OPTPP::Constraint bc;
    Vector *lower_bound;
    Vector *upper_bound;
    objective->GetBoundConstraint(lower_bound, upper_bound);
    if (lower_bound!=NULL and upper_bound==NULL) {
      NEWMAT::ColumnVector lower(lower_bound->ptr(), lower_bound->length());
      bc = new OPTPP::BoundConstraint(dimension, lower); 
    } else {
      if (lower_bound==NULL and upper_bound!=NULL){
        NEWMAT::ColumnVector upper(upper_bound->ptr(), upper_bound->length());
        bc = new OPTPP::BoundConstraint(dimension, upper, false);
      } else {
        NEWMAT::ColumnVector lower(lower_bound->ptr(), lower_bound->length());
        NEWMAT::ColumnVector upper(upper_bound->ptr(), upper_bound->length());
        bc = new OPTPP::BoundConstraint(dimension, lower, upper);
      }
    }
    constraint_array->append(bc);
  }

};

template<typename Method, typename Objective, bool Applicable>
class LinearEqualityTrait {
 public:
  static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
  
  }

};

template<typename Method, typename Objective>
class LinearEqualityTrait<Method, Objective, true> {
 public:
   static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
     index_t dimension = objective->dimension();
     Matrix *a_mat;
     Vector *b_vec;
     objective->GetLinearEquality(a_mat, b_vec);
     NEWMAT::Matrix alpha(a_mat->ptr(), a_mat->n_rows(), a_mat->n_cols());
     NEWMAT::ColumnVector beta(b_vec->ptr(), b_vec->length());  
     OPTPP::Constraint  leq = new OPTPP::LinearEquation(alpha, beta);
     constraint_array->append(leq);
  }


};

template<typename Method, typename Objective, bool Applicable>
class LinearInequalityTrait {
 public:
  static void UpdateConstraints(Objective *object, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
  }

};

template<typename Method, typename Objective>
class LinearInequalityTrait<Method, Objective, true> {
 public:
  static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {  
    index_t dimension = objective->dimension();
    OPTPP::Constraint lineq;
    Matrix *a_mat;
    Vector *left_b;
    Vector *right_b;
    objective->GetLinearInequalityConstraint(a_mat, left_b, right_b);
    NEWMAT::Matrix alpha(a_mat->ptr(), a_mat->n_rows(), a_mat->n_cols());
    DEBUG_ASSERT(a_mat!=NULL);
    DEBUG_ASSERT(left_b!=NULL or right_b!=NULL); 
    if (left_b!=NULL and right_b==NULL) {
      NEWMAT::ColumnVector left(left_b->ptr(), left_b->length());
      lineq = new OPTPP::LinearInequality(alpha, left); 
    } else {
      if (left_b==NULL and right_b!=NULL){
        NEWMAT::ColumnVector right(right_b->ptr(), right_b->length());
        lineq = new OPTPP::LinearInequality(alpha, right, false);
      } else {
        NEWMAT::ColumnVector left(left_b->ptr(), left_b->length());
        NEWMAT::ColumnVector right(right_b->ptr(), right_b->length());
        lineq = new OPTPP::LinearInequality(alpha, left, right);
      }
    }
    constraint_array->append(lineq);
  }
};

template<typename Method, typename Objective, bool Applicable>
class NonLinearEqualityTrait {
 public:
  static void UpdateConstraints(Objective *object, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
  }
};

template<typename Method, typename Objective>
class NonLinearEqualityTrait<Method, Objective, true> {
 public:
  static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
    index_t dimension = objective->dimension();
    OPTPP::NLP *nlp = new OPTPP::NLP(
        new typename OptimizationTrait<Method>::NlpType(
          StaticOptppOptimizer<Method, Objective>::
          ComputeNonLinearEqualityConstraints));
    OPTPP::Constraint eq = new OPTPP::NonLinearEquation(nlp,
        objective->num_of_non_linear_equations());
    constraint_array->append(eq);
  }
};

template<typename Method, typename Objective, bool Applicable>
class NonLinearInequalityTrait {
 public: 
  static void UpdateConstraints(Objective *objective, 
      OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
  }
  
};

template<typename Method, typename Objective>
class NonLinearInequalityTrait<Method, Objective, true> {
 public: 
    static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
    index_t dimension = objective->dimension();
    OPTPP::NLP *nlp = new OPTPP::NLP(
        new typename OptimizationTrait<Method>::NlpType(
          StaticOptppOptimizer<Method, Objective>::
          ComputeNonLinearInequalityConstraints));
    Vector *left_b;
    Vector *right_b;
    objective->GetNonLinearInequalityConstraintBounds(left_b, right_b);
    OPTPP::Constraint ineq; 
    if (left_b==NULL and right_b==NULL) {
      ineq = new OPTPP::NonLinearEquation(nlp, 
          objective->num_of_non_linear_inequalities());
    } else {
      if (left_b!=NULL and right_b==NULL) {
        NEWMAT::ColumnVector left(left_b->ptr(), left_b->length());    
        ineq = new OPTPP::NonLinearInequality(nlp, left,
            objective->num_of_non_linear_inequalities());  
      } else {   
        if (left_b==NULL and right_b!=NULL) {
          NEWMAT::ColumnVector right(right_b->ptr(), right_b->length());    
          ineq = new OPTPP::NonLinearInequality(nlp, right, false,
              objective->num_of_non_linear_inequalities());  
   
        } else {
          if (left_b!=NULL and right_b!=NULL) {
            NEWMAT::ColumnVector right(right_b->ptr(), right_b->length());    
            NEWMAT::ColumnVector left(left_b->ptr(), left_b->length());    
            ineq = new OPTPP::NonLinearInequality(nlp, left, right,
                objective->num_of_non_linear_inequalities());
          
          }
        }
      }
    } 
  } 
};

/*  Wrapper to Opt++ so that your life is easier 
 *  when it comes to interfacing Fastlib
 */ 
template <typename Method, typename Objective, ConstraintType Constraint>
class StaticOptppOptimizer {
 public:
  StaticOptppOptimizer() {
    objective_=NULL;
    method_=NULL;
    nlp_=NULL;
    compound_constraint_=NULL;
  }
  ~StaticOptppOptimizer() {
    Destruct();
  }
  void Init(fx_module *module, Objective *objective) {
    module_ = module;
    objective_ = objective;
    dimension_=objective_->dimension();
    OPTPP::OptppArray<OPTPP::Constraint> constraint_array;
    BoundConstraintTrait<Method, Objective, 
     (bool)(Constraint & BoundConstraint)>::UpdateConstraints(objective_, 
          &constraint_array);
    LinearEqualityTrait<Method, Objective, 
     (bool)(Constraint & LinearEquality)>::UpdateConstraints(objective_, 
          &constraint_array);
    LinearInequalityTrait<Method, Objective, 
     (bool)(Constraint & LinearInequality)>::UpdateConstraints(objective_, 
          &constraint_array);
    NonLinearEqualityTrait<Method, Objective, 
     (bool)(Constraint & NonLinearEquality)>::UpdateConstraints(objective_, 
          &constraint_array);
    NonLinearInequalityTrait<Method, Objective, 
     (bool)(Constraint & NonLinearInequality)>::UpdateConstraints(objective_, 
          &constraint_array);
 
    nlp_ = new typename OptimizationTrait<Method>::NlpType(dimension_, ComputeObjective, 
        Initialize, (OPTPP::CompoundConstraint *)NULL);
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
    method_ = new Method(nlp_, tols);
    // setting some generic options for the optimization method
    const char *status_file=fx_param_str(module_, "status_file", "status.txt" );
    if (method_->setOutputFile(status_file, 0)==false) {
      FATAL("Failed to open the status file %s for writing", status_file);
    }

    OptimizationTrait<Method>::InitializeMethod(module_, method_);
    
  }
  void Destruct() {
    if (method_!=NULL) {
      delete method_;
      method_=NULL;
    }
    if (nlp_!=NULL) {
      delete nlp_;
      nlp_=NULL;
    }
    if (compound_constraint_!=NULL) {
      delete compound_constraint_;
      compound_constraint_=NULL;
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
  typename OptimizationTrait<Method>::NlpType *nlp_;
  OPTPP::CompoundConstraint *compound_constraint_;

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
    } 
    if (mode & OPTPP::NLPGradient) {
       Vector vecg;
       vecg.Alias(gx.data(), ndim);
       objective_->ComputeGradient(vecx, &vecg);
       result = OPTPP::NLPGradient;   
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
    }
    if (mode & OPTPP::NLPGradient) {
      Vector vecg;
      vecg.Alias(gx.data(), ndim);
      objective_->ComputeGradient(vecx, &vecg); 
      result=OPTPP::NLPGradient;
    }
    if (mode & OPTPP::NLPHessian) {
      Matrix hessian;
      hessian.Alias(hx.data(), ndim, ndim);  
      objective_->ComputeHessian(vecx, hessian);  
      result = OPTPP::NLPHessian;
    }
  } 

  static void ComputeNonLinearEqualityConstraints(int mode, 
      const NEWMAT::ColumnVector &x, NEWMAT::ColumnVector &cvalue, 
      int &result) {
    Vector vecx;  
    vecx.Alias(x.data(), x.Nrows());
    Vector vecc;
    vecc.Alias(cvalue.data(), cvalue.Nrows());
    objective_->ComputeNonLinearEqualityConstraints(vecx, &vecc);
    result = OPTPP::NLPConstraint;
  
  }
  static void ComputeNonLinearEqualityConstraints(int mode, 
      int ndim, const NEWMAT::ColumnVector &x,  
      NEWMAT::ColumnVector &cvalue, NEWMAT::Matrix &cJacobian,
      int &result) {
    Vector vecx;  
    vecx.Alias((double *)x.data(), ndim);
    if (mode & OPTPP::NLPConstraint) {
      Vector vecc;
      vecc.Alias(cvalue.data(), ndim);
      objective_->ComputeNonLinearEqualityConstraints(vecx, &vecc);
      result = OPTPP::NLPConstraint;
    } 
    if (mode & OPTPP::NLPCJacobian) {
       Matrix cjacob;
       cjacob.Alias(cJacobian.data(), cJacobian.Nrows(), cJacobian.Ncols());
       objective_->ComputeNonLinearEqualityConstraintsJacobian(vecx, &cjacob);
       result = OPTPP::NLPCJacobian;   
    } 
  }
  static void ComputeNonLinearEqualityConstraints(int mode, 
      int ndim, const NEWMAT::ColumnVector &x,  
      NEWMAT::ColumnVector &cvalue, NEWMAT::Matrix &cJacobian,
      OPTPP::OptppArray<NEWMAT::SymmetricMatrix> &cHessian,
      int &result) {
    FATAL("Not supported yet");
  }
 
  static void ComputeNonLinearInequalityConstraints(int mode, 
      const NEWMAT::ColumnVector &x, NEWMAT::ColumnVector &cvalue, 
      int &result) {
    Vector vecx;  
    vecx.Alias(x.data(), x.Nrows());
    Vector vecc;
    vecc.Alias(cvalue.data(), x.Nrows());
    objective_->ComputeNonLinearInequalityConstraints(vecx, &vecc);
    result = OPTPP::NLPConstraint;  
  }

  // We still have to test wether cJacobian and Hessian etc get initialized
  // properly
  static void ComputeNonLinearInequalityConstraints(int mode, 
      int ncon, const NEWMAT::ColumnVector &x,  
      NEWMAT::ColumnVector &cvalue, NEWMAT::Matrix &cJacobian,
      int &result) {
    Vector vecx;  
    vecx.Alias((double *)x.data(), x.Nrows());
    if (mode & OPTPP::NLPConstraint) {
      Vector vecc;
      vecc.Alias(cvalue.data(), x.Nrows());
      objective_->ComputeNonLinearInequalityConstraints(vecx, &vecc);
      result = OPTPP::NLPConstraint;
    } 
    if (mode & OPTPP::NLPCJacobian) {
       Matrix cjacob;
       cjacob.Alias(cJacobian.data(), cJacobian.Nrows(), cJacobian.Ncols());
       objective_->ComputeNonLinearInequalityConstraintsJacobian(vecx, 
           &cjacob);
       result = OPTPP::NLPCJacobian;   
    }
  
  }
  static void ComputeNonLinearInequalityConstraints(int mode, 
      int ndim, const NEWMAT::ColumnVector &x,  
      NEWMAT::ColumnVector &cvalue, NEWMAT::Matrix &cJacobian,
      OPTPP::OptppArray<NEWMAT::SymmetricMatrix> &cHessian,
      int &result) {
    FATAL("Not supported yet");
  }

};

template<typename Method, typename Objective, ConstraintType Constraint>
  Objective *StaticOptppOptimizer<Method, Objective, 
      Constraint>::objective_=NULL;

}; // Namespace optim
#endif
