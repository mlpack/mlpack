/** @file optimizer.h
 *
 *  @author Nikolaos Vasiloglou (nick)
 *
 * 
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
#include "include/OptBCNewton.h"
#include "include/OptConstrNewton.h"
#include "include/OptQNewton.h"
#include "include/BoundConstraint.h"
#include "include/CompoundConstraint.h"
#include "include/LinearEquation.h"
#include "include/LinearInequality.h"
#include "include/NonLinearEquation.h"
#include "include/NonLinearInequality.h"

/**
 *  @brief Namespace optim
 *  Here you can find all the classes you need for optimization
 *
 */

namespace optim {
 namespace optpp {
/**
 * @brief  Definitions for the op++ library
 * @code
 *   typedef OPTPP::OptCG       CG;      // for conjugate gradient
 *   typedef OPTPP::OptQNewton  QNewton; // for quasi newton method
 *   typedef OPTPP::OptQNewton  BFGS;    // for BFGS
 *   typedef OPTPP::OptFDNewton FDNewton;// for newton with finite differences
 *   typedef OPTPP::OptNewton   Newton;  // for standard newton
 *   typedef OPTPP::BCNewton    BCNewton;// for bound constraint newton
 *   typedef OPTPP::OptConstrNewton CNewton; //for constrained Newton
 *   typedef OPTPP::OptLBFGS    LBFGS;   // for limited BFGS
 * @endcode
 */  

typedef OPTPP::OptCG       CG;
typedef OPTPP::OptQNewton  QNewton;
typedef OPTPP::OptQNewton  BFGS;
typedef OPTPP::OptFDNewton FDNewton;
typedef OPTPP::OptNewton   Newton;
typedef OPTPP::OptBCNewton BCNewton;
typedef OPTPP::OptConstrNewton CNewton;
typedef OPTPP::OptLBFGS    LBFGS;

/**
 * @brief
 * opt++ supports different constraint types:
 * if you want a combination of the constraints then
 * you just add them. For example if you have bound constraints  and 
 * linear inequalities and nonlinear equalities then you use:
 * BoundConstraint + LinearEquality + NonlinearEquality
 * @code
 *   enum ConstraintType {
 *     NoConstraint       = 0,  // unconstrained optimization
 *     BoundConstraint    = 2,  // box constraints
 *     LinearEquality     = 4,  // linear equalities
 *     LinearInequality   = 8,  // linear inequalities
 *     NonLinearEquality  = 16, // nonlinear equalities
 *     NonLinearInequality= 32  // nonlinear inequalities
 *   };
 * @endcode
 */
enum ConstraintType {
  NoConstraint       = 0,
  BoundConstraint    = 2,
  LinearEquality     = 4,
  LinearInequality   = 8,
  NonLinearEquality  = 16,
  NonLinearInequality= 32
};

/**
 * @brief This is the core class for running optimization. It supports
 * everything that opt++ offer, constrained and unconstrained optimization, 
 * linear/nonlinear equalities/ineqialities  as well as bound constraints. The
 * algorithms include zero order methods (derivative free) up to second order
 * (Newton's method). 
 *
 * @class  StaticOptppOptimizer
 * template <typename Method, typename Objective, ConstraintType Constraint>
 *           class StaticOptppOptimizer;
 *  @param[Method the optimization method we are going to use, it can be any of 
 *   the following: 
 *    @code
 *      typedef OPTPP::OptCG       CG;      // for conjugate gradient
 *      typedef OPTPP::OptQNewton  QNewton; // for quasi newton method
 *      typedef OPTPP::OptQNewton  BFGS;    // for BFGS
 *      typedef OPTPP::OptFDNewton FDNewton;// for newton with finite differences
 *      typedef OPTPP::OptNewton   Newton;  // for standard newton
 *      typedef OPTPP::OptLBFGS    LBFGS;   // for limited BFGS
 *    @endcode
 *  @param[Constraint It can be any sum of the following enums
 *    @code
 *      enum ConstraintType {
 *        NoConstraint       = 0,  // unconstrained optimization
 *        BoundConstraint    = 2,  // box constraints
 *        LinearEquality     = 4,  // linear equalities
 *        LinearInequality   = 8,  // linear inequalities
 *        NonLinearEquality  = 16, // nonlinear equalities
 *        NonLinearInequality= 32  // nonlinear inequalities
 *      };
 *    @endcode
  *  @param[Objective This is the class that defines the objective ant the
 *   constraints, as well as initializations. You can use any class that defines
 *   the following member functions:
 *   @note Not every function has to be defined. For example if you are using
 *   a first order method  that requires gradient only, you don't need to define
 *   ComputeHessian. Another example, if you are using only linear equality
 *   constraints you don't need to define the functions that have to do with
 *   nonlinear equality or inequality constraints.
 *   @code
 *    class MyOptimizationProblem {
 *      public:
 *       // returns the dimensionality of the optimization
 *       //   also known as the number of the optimization variable
 *       index_t dimension(); 
 *       
 *       // fills the vector with initial value
 *       void GiveInit(Vector *vec); 
 *       
 *       // Computes the objective f(x) and returns it to value
 *       void ComputeObjective(Vector &x, double *value);
 *       
 *       // Computes the gradient g=f'(x)
 *       void ComputeGradient(Vector &x, Vector *g);
 *       
 *       // Computes the Hessian  h=f''(x);
 *       void ComputeHessian(Vector &x, Matrix *h);
 *     
 *       // Defines the Bound constraints for the variables
 *       void GetBoundConstraint(Vector *lower_bound, Vector *upper_bound);
 *    @endcode   
 *       @note \f$  \vec{l} \leq \vec{x} \leq \vec{u} \f$
 *        The Vectors lower_bound and upper_bound contain the lower 
 *        and upper bounds for all the constraints. If you want to have 
 *        only lower bounds then leave the upper_bound vector unchanged. If you 
 *        want only upper bounds then leave the lower bound unchanged. 
 *    @code 
 *       // Fills the matrix A and vector b for equality constraints 
 *       void GetLinearEquality(Matrix *a_mat, Vector *b_vec);
 *    @endcode   
 *       @note  The constraint is expressed in the form \f$ A \vec{x} = \vec{b} \f$
 *    @code 
 *       // Fills the matrix A, lower and upper bounds for the linear
 *       // inequalities
 *       void GetLinearInequality(Matrix *a_mat, Vector *l_vec, Vector *u_vec);
 *    @endcode   
 *       @note The constraint is expressed in the form \f$ \vec{l} \leq A \vec{x} \leq \vec{u} \f$
 *        if you have only left or right hand side inequality, just do nothing about
 *        the one  you don't care.
 *    @code   
 *       // Returns the number of nonlinear equalities
 *       index_t num_of_non_linear_equalities()    
 *
 *       // Writes on vecc the evaluation of nonlinear equality constraints on
 *       // vecx
 *       void ComputeNonLinearEqualityConstraints(Vector &vecx, Vector *vecc);
 *       
 *       // Writes on cjacob  the evaluation of the jacobian nonlinear equality 
 *       // constraints on vecx
 *       void ComputeNonLinearEqualityConstraintsJacobian(Vector &vecx, Matrix *cjacob);
 *       
 *       // Returns the number of nonlinear inequalities
 *       index_t num_of_non_linear_inequalities()    
 *       
 *       // Writes on vecc the evaluation of nonlinear inequality constraints on
 *       // vecx
 *       void ComputeNonLinearInequalityConstraints(Vector &vecx, Vector *vecc);
 *       
 *       // Writes on cjacob  the evaluation of the jacobian nonlinear equality 
 *       // constraints on vecx
 *       void ComputeNonLinearInequalityConstraintsJacobian(Vector &vecx, Matirx *cjacob);
 *       
 *       // Writes on both vectors the lower and upper bounds of nonlinear
 *       // inequalities 
 *       void GetNonLinearInequalityConstraintBounds(Vector *lower, Vector *upper);
 *    @endcode   
 *      @note The constraint is expressed in the form \f$ \vec{l} \leq f(\vec{x}) \leq \vec{u}\f$ , 
 *      if you want only one side inequality then ignore the side you are not
 *            interested in. If you don't specify any of \f$ \vec{l}, \vec{u} \f$ then 
 *            it assumes the inequality  \f$ f(\vec{x}) \leq 0 \f$ .
 *   @code
 *    };
 *   @endcode
 *
 */
template <typename Method, 
         typename Objective, 
         ConstraintType Constraint=NoConstraint>
class StaticOptppOptimizer;

/**
 * @brief Here we define a trait for the optimization, we need this trait to do
 * the necessary initializations
 *        
 */
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

/**
 * @brief trait specialization for Newton method
 */ 
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

/**
 *
 * @brief trait specialization for Quasi-Newton with finite difference
 * approximation of the Hessian 
 */
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

/**
 * @brief trait specialization for the Quasi-Newton BFGS
 *
 */
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

/**
 * @brief trait specialization for the Newton
 *
 */
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

// In this section we define the traits for  constraints


/**
 * @brief This trait is usefull for handling bound constraints
 *
 *    template<typename Method, typename Objective, bool Applicable>
 *          class BoundConstraintTrait;
 * @brief The default behaviour when the bound constraint is not applicable
 * is to do nothing
 */
template<typename Method, typename Objective, bool Applicable>
class BoundConstraintTrait {
 public:
  static void UpdateConstraints(Objective *object, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {  
  }
};

/**
 * @brief When the bound constraint is applicable then it updated the
 * constraints accordingly. The objective must provide a GetBoundConstraints
 * public member function defined as:
 * @code
 *   class Objective {
 *    public:
 *     void GetBoundConstraint(Vector *lower_bound, Vector *upper_bound);
 *   };
 * @endcode
 *  \f$  \vec{l} \leq \vec{x} \leq \vec{u}\f$
 * The Vectors lower_bound and upper_bound contain the lower 
 *  and upper bounds for all the constraints. If you want to have 
 *  only lower bounds then leave the upper_bound vector unchanged. If you 
 *  want only upper bounds then leave the lower bound unchanged. 
 */
template<typename Method, typename Objective>
class BoundConstraintTrait<Method, Objective, true> {
 public:
  static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
    index_t dimension = objective->dimension();
    OPTPP::Constraint bc;
    Vector lower_bound;
    Vector upper_bound;
    objective->GetBoundConstraint(&lower_bound, &upper_bound);
    if (lower_bound.length()!=BIG_BAD_NUMBER and 
        upper_bound.length()==BIG_BAD_NUMBER) {
      NEWMAT::ColumnVector lower(lower_bound.ptr(), lower_bound.length());
      bc = new OPTPP::BoundConstraint(dimension, lower); 
    } else {
      if (lower_bound.length()==BIG_BAD_NUMBER and 
          upper_bound.length()!=BIG_BAD_NUMBER){
        NEWMAT::ColumnVector upper(upper_bound.ptr(), upper_bound.length());
        bc = new OPTPP::BoundConstraint(dimension, upper, false);
      } else {
        NEWMAT::ColumnVector lower(lower_bound.ptr(), lower_bound.length());
        NEWMAT::ColumnVector upper(upper_bound.ptr(), upper_bound.length());
        bc = new OPTPP::BoundConstraint(dimension, lower, upper);
      }
    }
    constraint_array->append(bc);
  }

};
/**
 * @class LinearEqualityTrait
 *   template<typename Method, typename Objective, bool Applicable>
 *         class LinearEqualityTrait;
 * @brief This trait is usefull for handling linear equalities
 * @class template<typename Method, typename Objective, bool Applicable>
 *          class LinearEqualityTrait;
 * @brief The default behaviour when the linear equality constraint is not applicable
 * is to do nothing
 */

template<typename Method, typename Objective, bool Applicable>
class LinearEqualityTrait {
 public:
  static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
  
  }

};

/**
 * @class LinearEqualityTrait<Method, Objective, true>
 *   template<typename Method, typename Objective>
 *          class LinearEqualityTrait<Method, Objective, true> 
 * @brief When linear equalities are applicable then it updates the
 * constraints accordingly. The objective must provide a GetLinearEquality
 * public member function defined as:
 * @code
 *   class Objective {
 *    public:
 *     void GetLinearEquality(Matrix *a_mat, Vector *b_vec);
 *   };
 * @endcode
 * The constraint is expressed in the form \f$ A \vec{x} = \vec{b} \f$
 */
template<typename Method, typename Objective>
class LinearEqualityTrait<Method, Objective, true> {
 public:
   static void UpdateConstraints(Objective *objective, 
       OPTPP::OptppArray<OPTPP::Constraint> *constraint_array) {
     index_t dimension = objective->dimension();
     Matrix a_mat;
     Vector b_vec;
     objective->GetLinearEquality(&a_mat, &b_vec);
     NEWMAT::Matrix alpha(a_mat.ptr(), a_mat.n_rows(), a_mat.n_cols());
     NEWMAT::ColumnVector beta(b_vec.ptr(), b_vec.length());  
     OPTPP::Constraint  leq = new OPTPP::LinearEquation(alpha, beta);
     constraint_array->append(leq);
  }

};

/**
 * @class LinearInequalityTrait 
 *   template<typename Method, typename Objective, bool Applicable>
 *            class LinearInequalityTrait 
 * @brief When linear inequalities are applicable then it updates the
 * constraints accordingly. The objective must provide a GetLinearInequality
 * public member function defined as:
 * @code
 *   class Objective {
 *    public:
 *     void GetLinearInequality(Matrix *a_mat, Vector *l_vec, Vector *u_vec);
 *   };
 * @endcode
 * The constraint is expressed in the form \f$ \vec{l} \leq A \vec{x} \leq \vec{u} \f$
 * if you have only left or right hand side inequality, just do nothing about
 * the one  you don't care.
 */

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
    Matrix a_mat;
    Vector left_b;
    Vector right_b;
    objective->GetLinearInequalityConstraint(&a_mat, &left_b, &right_b);
    NEWMAT::Matrix alpha(a_mat.ptr(), a_mat.n_rows(), a_mat.n_cols());
    DEBUG_ASSERT(a_mat.n_rows()!=BIG_BAD_NUMBER);
    DEBUG_ASSERT(left_b.length()!=BIG_BAD_NUMBER or 
        right_b.length()!=BIG_BAD_NUMBER); 
    if (left_b.length()!=BIG_BAD_NUMBER and right_b.length()==BIG_BAD_NUMBER) {
      NEWMAT::ColumnVector left(left_b.ptr(), left_b.length());
      lineq = new OPTPP::LinearInequality(alpha, left); 
    } else {
      if (left_b.length()==BIG_BAD_NUMBER and 
          right_b.length()!=BIG_BAD_NUMBER){
        NEWMAT::ColumnVector right(right_b.ptr(), right_b.length());
        lineq = new OPTPP::LinearInequality(alpha, right, false);
      } else {
        NEWMAT::ColumnVector left(left_b.ptr(), left_b.length());
        NEWMAT::ColumnVector right(right_b.ptr(), right_b.length());
        lineq = new OPTPP::LinearInequality(alpha, left, right);
      }
    }
    constraint_array->append(lineq);
  }
};

/**
 * @class NonLinearEqualityTrait
 *   template<typename Method, typename Objective, bool Applicable>
 *            class NonLinearEqualityTrait;
 * @brief When the nonlinear constraints are  applicable then it updates the
 * constraints accordingly. The objective must provide a 
 * public member function defined as:
 * @code
 *   class Objective {
 *    public:
 *     double num_of_non_linear_equalities());
 *   };
 * @endcode
 * The constraint is expressed in the form \f$ f(\vec{x})=0 \f$
 */

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
        objective->num_of_non_linear_equalities());
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

/**
 * @class NonLinearInequalityTrait 
 *   template<typename Method, typename Objective, bool Applicable>
 *            class NonLinearInequalityTrait 
 * @brief When the nonlineare inequality constraints are applicable then it updates the
 * constraints accordingly. The objective must provide the 
 * public member functions defined as:
 * @code
 *   class Objective {
 *    public:
 *     void GetNonLinearInequalityConstraintBounds(Vector *lower, Vector *upper);
 *     double num_of_non_linear_inequalities());
 *   };
 * @endcode
 * The constraint is expressed in the form \f$ \vec{l} \leq f(\vec{x}) \leq \vec{u}\f$
 * if you want only one side inequality then ignore the side you are not
 * interested in. If you don't specify any of \f$ \vec{l}, \vec{u} \f$ then 
 * it assumes the inequality  \f$ f(\vec{x}) \leq 0 \f$.
 */

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
    Vector left_b;
    Vector right_b;
    objective->GetNonLinearInequalityConstraintBounds(&left_b, &right_b);
    OPTPP::Constraint ineq; 
    if (left_b.length()==BIG_BAD_NUMBER and 
        right_b.length()==BIG_BAD_NUMBER) {
      ineq = new OPTPP::NonLinearInequality(nlp, 
          objective->num_of_non_linear_inequalities());
    } else {
      if (left_b.length()!=BIG_BAD_NUMBER and 
          right_b.length()==BIG_BAD_NUMBER) {
        NEWMAT::ColumnVector left(left_b.ptr(), left_b.length());    
        ineq = new OPTPP::NonLinearInequality(nlp, left,
            objective->num_of_non_linear_inequalities());  
      } else {   
        if (left_b.length()==BIG_BAD_NUMBER and right_b.length()!=BIG_BAD_NUMBER) {
          NEWMAT::ColumnVector right(right_b.ptr(), right_b.length());    
          ineq = new OPTPP::NonLinearInequality(nlp, right, false,
              objective->num_of_non_linear_inequalities());  
   
        } else {
          if (left_b.length()!=BIG_BAD_NUMBER and
              right_b.length()!=BIG_BAD_NUMBER) {
            NEWMAT::ColumnVector right(right_b.ptr(), right_b.length());    
            NEWMAT::ColumnVector left(left_b.ptr(), left_b.length());    
            ineq = new OPTPP::NonLinearInequality(nlp, left, right,
                objective->num_of_non_linear_inequalities());
          
          }
        }
      }
    } 
  } 
};
template<typename Method>
class SearchStrategyTrait {
 public:
   static void SetSearchStrategy(Method *method, fx_module *module) {
   
   }
};

template<>
class SearchStrategyTrait<OPTPP::OptConstrNewtonLike> {
 public: 
  static void SetSearchStrategy(OPTPP::OptConstrNewtonLike *method, fx_module *module) {
    if (fx_param_exists(module, "strategy")) {
      std::string strategy=fx_param_str_req(module, "strategy");
      if (strategy=="linesearch") {
        method->setSearchStrategy(OPTPP::LineSearch);
      } else {
        if (strategy=="trustregion") {
          method->setSearchStrategy(OPTPP::TrustRegion);
        } else {
          if (strategy=="trustpds") {
            method->setSearchStrategy(OPTPP::TrustPDS);
          }
        }
      }    
    }
  }
};

template<>
class SearchStrategyTrait<OPTPP::OptBCNewton> {
 public: 
  static void SetSearchStrategy(OPTPP::OptBCNewtonLike *method, fx_module *module) {
    if (fx_param_exists(module, "strategy")) {
      std::string strategy=fx_param_str_req(module, "strategy");
      if (strategy=="linesearch") {
        method->setSearchStrategy(OPTPP::LineSearch);
      } else {
        if (strategy=="trustregion") {
          method->setSearchStrategy(OPTPP::TrustRegion);
        } else {
          if (strategy=="trustpds") {
            method->setSearchStrategy(OPTPP::TrustPDS);
          }
        }
      }    
    }
  }
};


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
    // Load the constraints
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
    SearchStrategyTrait<Method>::SetSearchStrategy(method_, module_);
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
};  //Namespace optpp
}; // Namespace optim
#endif
