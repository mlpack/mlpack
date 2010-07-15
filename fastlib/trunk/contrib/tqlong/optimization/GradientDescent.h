#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <fastlib/fastlib.h>

#include "BFGS.h"

#ifndef BEGIN_OPTIM_NAMESPACE
#define BEGIN_OPTIM_NAMESPACE namespace optim {
#endif
#ifndef END_OPTIM_NAMESPACE
#define END_OPTIM_NAMESPACE }
#endif

BEGIN_OPTIM_NAMESPACE;

/**********************************************************************
  Implement 1-order Gradient Descent optimization method with Wolfe line search method
  Function: CalculateValue(const variable_type&)
            CalculateGradient(const variable_type&, variable_type& gradient)
  Function::variable_type:
    implement la::AddExpert(double, const variable_type&, variable_type*)
              la::ScaleOverwrite(double, const variable_type&, variable_type*);
              la::LengthEuclidean(const variable_type&)
              la::Dot(const variable_type&, const variable_type&)
              variable_type.CopyValues(const variable_type&)
  Parameter:
    General params
      maxIter, rTol, aTol
    Specific for Gradient Descent & Wolfe line search
      c1, c2, beta
**********************************************************************/
template<typename Function> class GradientDescent : public BFGS<Function>
{
public:
  typedef Function function_type;
  typedef typename Function::variable_type variable_type;
  typedef double* OptimizationParameters;
    //maximum # of iterations  : maxIter = (int) param[0];
    //relative tolerance       : rTol = param[1];
    //absolute tolerance       : aTol = param[2];
    //c1 : Wolfe 1st condition : c1 = param[3];
    //c2 : Wolfe 2nd condition : c2 = param[4];
    //beta : scale parameter   : beta = param[5];
protected:
  static double default_gradient_descent_parameter[6]; // = {100, 0.001, 0.01, 1e-4, 0.9, 0.5};
public:
  GradientDescent(function_type& f_, OptimizationParameters param_ = default_gradient_descent_parameter);
  double optimize(variable_type& sol);
protected:
};

template<typename F>
GradientDescent<F>::GradientDescent(function_type& f_, OptimizationParameters param_)
  : BFGS<F>(f_, param_)
{
}

template<typename F>
double GradientDescent<F>::optimize(variable_type &sol) {
  variable_type x; // the current search variable
  variable_type grad, d; // the current search direction

  this->f.Init(&x);      // initialize search variable
  this->f.Init(&d);      // and direction
  this->f.Init(&grad);      // and gradient place holder

  x.CopyValues(this->x0);
  double val = CalculateValue(x);
  CalculateGradient(x, grad); // Calculate gradient
  double r0 = la::LengthEuclidean(grad);

  sol.CopyValues(this->x0);

  this->history.Clear();
  this->iter = 0;
  this->n_evals = 0;
  this->n_grads = 0;
  this->best_val = val;
  this->residual = r0;
  this->recordProgress();

  for (this->iter = 1; this->iter < this->maxIter; this->iter++) {
    // Calculate search direction: negative gradient
    la::ScaleOverwrite(-1.0, grad, &d);
    // Calculate step size by Wolfe's conditions
    double lambda = WolfeStep(x, val, grad, d);
    if (lambda == 0.0) return this->best_val;
    // Calculate new variable
    la::AddExpert(lambda, d, &x);
    // Update best value
    val = this->val_xp;
    if (val < this->best_val) {
      this->best_val = val;
      sol.CopyValues(x);
    }
    // Calculate gradient and check termination condition
    grad.CopyValues(this->grad_p);
    this->residual = la::LengthEuclidean(grad);
    this->recordProgress();
    if (this->residual < this->rTol*r0+this->aTol) break;
  }
  return this->best_val;
}

END_OPTIM_NAMESPACE;

#endif // GRADIENTDESCENT_H
