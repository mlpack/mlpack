#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <fastlib/fastlib.h>

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
template<typename Function> class GradientDescent
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
  struct HistoryRecord {
    int iter;
    int n_evals;
    int n_grads;
    double best_val;
    double residual;
    OT_DEF(HistoryRecord) {
      OT_MY_OBJECT(iter);
      OT_MY_OBJECT(n_evals);
      OT_MY_OBJECT(n_grads);
      OT_MY_OBJECT(best_val);
      OT_MY_OBJECT(residual);
    }
  public:
    HistoryRecord(int iter_, int n_evals_, int n_grads_, double best_val_, double residual_)
      : iter(iter_), n_evals(n_evals_), n_grads(n_grads_), best_val(best_val_), residual(residual_) {}
  };
protected:
  static double default_gradient_descent_parameter[6]; // = {100, 0.001, 0.01, 1e-4, 0.9, 0.5};
public:
  GradientDescent(function_type& f_, OptimizationParameters param_ = default_gradient_descent_parameter);
  void setParam(OptimizationParameters param);
  void setX0(const variable_type& x0_);
  double optimize(variable_type& sol);
  void printHistory();

  ArrayList<HistoryRecord> history;
protected:
  function_type& f;
  variable_type x0;

  // parameters
  OptimizationParameters param;
  int maxIter;
  double aTol, rTol;
  double c1, c2, beta;

  // progress
  int iter;
  int n_evals;
  int n_grads;
  double best_val;
  double residual;

  double WolfeStep(const variable_type& x, double val_x, const variable_type& grad, const variable_type& p);
  double CalculateValue(const variable_type& x);
  void CalculateGradient(const variable_type& x, variable_type& grad);
  void recordProgress();
  void printProgress();
};

template<typename F>
double GradientDescent<F>::CalculateValue(const variable_type& x) {
  double val = f.CalculateValue(x);
  n_evals++;
  return val;
}

template<typename F>
void GradientDescent<F>::CalculateGradient(const variable_type& x, variable_type& grad) {
  f.CalculateGradient(x, grad);
  n_grads++;
}

template<typename F>
void GradientDescent<F>::recordProgress() {
  history.PushBackCopy(HistoryRecord(iter, n_evals, n_grads, best_val, residual));
  printProgress();
}

template<typename F>
void GradientDescent<F>::printProgress() {
  if (history.size() == 0) return;
  printf("iter = %d n_evals = %d n_grads = %d best_val = %f residual = %f\n",
         history.back().iter, history.back().n_evals, history.back().n_grads,
         history.back().best_val, history.back().residual);
}

template<typename F>
void GradientDescent<F>::printHistory() {
  printf("History:\n");
  for (int i = 0; i < history.size(); i++) {
    printf("iter = %d n_evals = %d n_grads = %d best_val = %f residual = %f\n",
           history[i].iter, history[i].n_evals, history[i].n_grads, history[i].best_val, history[i].residual);
  }
}

template<typename F>
double GradientDescent<F>::default_gradient_descent_parameter[6] = {100, 0.001, 0.01, 1e-4, 0.9, 0.5};

template<typename F>
GradientDescent<F>::GradientDescent(function_type &f_, OptimizationParameters param_)
  : f(f_), param(param_) {
  setParam(param);
  f.Init(&x0);
  history.Init();
}

template<typename F>
void GradientDescent<F>::setParam(OptimizationParameters param) {
  maxIter = (int) param[0];
  rTol = param[1];
  aTol = param[2];
  c1 = param[3];
  c2 = param[4];
  beta = param[5];
}

template<typename F>
void GradientDescent<F>::setX0(const variable_type& x0_) {
  x0.CopyValues(x0_);
}

template<typename F>
double GradientDescent<F>::optimize(variable_type &sol) {
  history.Clear();
  iter = 0;
  n_evals = 0;
  n_grads = 0;

  variable_type x; // the current search variable
  variable_type grad, d; // the current search direction

  f.Init(&x);      // initialize search variable
  f.Init(&d);      // and direction
  f.Init(&grad);      // and direction

  x.CopyValues(x0);
  double val = CalculateValue(x);
  CalculateGradient(x, grad); // Calculate gradient
  double r0 = la::LengthEuclidean(grad);

  sol.CopyValues(x0);
  best_val = val;
  residual = r0;
  recordProgress();
  for (iter = 1; iter < maxIter; iter++) {
    // Calculate search direction: negative gradient
    la::ScaleOverwrite(-1.0, grad, &d);
    // Calculate step size by Wolfe's conditions
    double lambda = WolfeStep(x, val, grad, d);
    // Calculate new variable
    la::AddExpert(lambda, d, &x);
    // Update best value
    val = CalculateValue(x);
    if (val < best_val) {
      best_val = val;
      sol.CopyValues(x);
    }
    // Calculate gradient and check termination condition
    CalculateGradient(x, grad);
    residual = la::LengthEuclidean(grad);
    recordProgress();
    if (residual < rTol*r0+aTol) break;
  }
  return best_val;
}

template<typename F>
double GradientDescent<F>::WolfeStep(const variable_type& x, double val_x, const variable_type& grad, const variable_type& p) {
  double lambda = 1.0/beta, val_xp;
  double dot_grad_p;
  variable_type x_p, grad_p;
  f.Init(&x_p);
  f.Init(&grad_p);

  dot_grad_p = la::Dot(grad, p);
  while (1) {
    lambda *= beta;
    if (lambda < 1e-10) {
      printf("Line search results in a too small step size, try increasing c2 (param[4]).\n");
      break;
    }
    x_p.CopyValues(x);
    la::AddExpert(lambda, p, &x_p);     // x_p = x + lambda*p

    val_xp = CalculateValue(x_p);     // f(x_p)

    // first Wolfe condition
    if (val_xp - val_x <= c1*lambda*dot_grad_p) {  // f(x_p) - f(x) <= c1 * lambda * <grad, p>
      // second Wolfe condition
      CalculateGradient(x_p, grad_p);   // grad f(x_p)
      double dot_grad_x_p = la::Dot(grad_p, p);
      if (dot_grad_x_p >= c2*dot_grad_p) break;          // <p,grad f(x_p)> >= c2 * <p, grad>
    }
  }
  return lambda;
}

END_OPTIM_NAMESPACE;

#endif // GRADIENTDESCENT_H
