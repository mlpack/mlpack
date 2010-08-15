#ifndef LBFGS_H
#define LBFGS_H

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
  Implement Quasi-Newton L-BFGS optimization method with Wolfe line search method
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
    Specific for LBFGS
      mem
    Specific Wolfe line search
      c1, c2, beta
**********************************************************************/
template<typename Function> class LBFGS : public BFGS<Function> {
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
    //mem : memory size        : mem = (int) param[6];
protected:
  static double default_lbfgs_parameter[7]; // = {100, 0.001, 0.01, 1e-4, 0.9, 0.5, 3};
public:
  LBFGS(function_type& f_, OptimizationParameters param_ = default_lbfgs_parameter);
  void setParam(OptimizationParameters param);
  double optimize(variable_type& sol);
  // inherit setX0() from BFGS

protected:
  int mem;
  ArrayList<int> memIndex;
  ArrayList<double> gamma;

  void LBFGSDirection(variable_type& d);
  void MemoryUpdate(const variable_type& new_s, const variable_type& new_y);
};

template<typename F>
double LBFGS<F>::default_lbfgs_parameter[7] = {100, 0.001, 0.01, 1e-4, 0.9, 0.5, 3};

template<typename F>
LBFGS<F>::LBFGS(function_type &f_, OptimizationParameters param_)
  : BFGS<F>(f_, param_) {
  setParam(param_);
  memIndex.Init();
  gamma.Init();
}

template<typename F>
void LBFGS<F>::setParam(OptimizationParameters param) {
  BFGS<F>::setParam(param);
  mem = (int) param[6];
  DEBUG_ASSERT(mem >= 3);              // should have at least 3 memory slots
}

template<typename F>
double LBFGS<F>::optimize(variable_type &sol) {
  variable_type x; // the current search variable
  variable_type grad, d; // the current search direction
  variable_type new_s, new_y;  // s = x_n - x_n-1,  y = g_n - g_n-1

  this->f.Init(&x);      // initialize search variable
  this->f.Init(&d);      // and direction
  this->f.Init(&grad);   // and gradient place holder
  this->f.Init(&new_s);  // and s & y (for BFGS)
  this->f.Init(&new_y);

  x.CopyValues(this->x0);     // Start optimize at x0
  double val = CalculateValue(x);
  CalculateGradient(x, grad); // Calculate gradient
  double r0 = la::LengthEuclidean(grad);  // the beginning residual

  sol.CopyValues(this->x0);

  this->history.Clear();
  this->iter = 0;
  this->n_evals = 0;
  this->n_grads = 0;
  this->best_val = val;
  this->residual = r0;
  this->recordProgress();

  bool need_fixed = false;   // Clear all memory
  this->s.Clear();
  this->y.Clear();
  memIndex.Clear();
  gamma.Clear();
  for (this->iter = 1; this->iter < this->maxIter; this->iter++) {
    // Calculate search direction: negative gradient
    la::ScaleOverwrite(-1.0, grad, &d);
    if (!need_fixed) LBFGSDirection(d);     // la::Dot(s,y) > 0, use BFGSDirection
    // Calculate step size by Wolfe's conditions,
    // x_p, val_xp and grad_p are new search point, its function value and gradient
    double lambda = WolfeStep(x, val, grad, d);
    if (lambda == 0.0) return this->best_val;     // line search failed
    // Calculate new variable
    la::AddExpert(lambda, d, &x);
    // Update best value
    val = this->val_xp;
    if (val < this->best_val) {
      this->best_val = val;
      sol.CopyValues(x);
    }
    // Update BFGS memory
    la::ScaleOverwrite(lambda, d, &new_s);     //  s = x_n - x_n-1 = lambda * d

    la::ScaleOverwrite(-1.0, grad, &new_y);
    grad.CopyValues(this->grad_p);
    la::AddExpert(1.0, grad, &new_y);          //  y = g_n - g_n-1
    need_fixed = la::Dot(new_s, new_y) < 0;    //  if <s,y> >= 0 then use LBFGSDirection

    MemoryUpdate(new_s, new_y);

    // Check termination condition
    this->residual = la::LengthEuclidean(grad);
    this->recordProgress();
    if (this->residual < this->rTol*r0+this->aTol) break;
  }
  return this->best_val;
}

template<typename F>
void LBFGS<F>::MemoryUpdate(const variable_type& new_s, const variable_type& new_y) {
  if (this->s.size() < mem) {
    memIndex.PushBackCopy(this->iter);
    gamma.PushBackCopy(la::Dot(new_s, new_y));
    this->s.PushBackCopy(new_s);
    this->y.PushBackCopy(new_y);
  }
  else {  // memory is full, replace the oldest memory slot
    index_t min_i = 0;
    for (int i = 0; i < memIndex.size(); i++)
      if (memIndex[i] < memIndex[min_i]) min_i = i;
    memIndex[min_i] = this->iter;
    gamma[min_i] = la::Dot(new_s, new_y);
    this->s[min_i].CopyValues(new_s);
    this->y[min_i].CopyValues(new_y);
  }
}

template<typename F>
void LBFGS<F>::LBFGSDirection(variable_type& d) {
  DEBUG_ASSERT(memIndex.size() == gamma.size() &&
               memIndex.size() == this->s.size() &&
               memIndex.size() == this->y.size());
  if (this->s.size() == 0) return;
  // Find the latest memory slot
  int max_i = 0;
  for (int i = 0; i < memIndex.size(); i++)
    if (memIndex[i] > memIndex[max_i]) max_i = i;
  ArrayList<double> alpha;
  alpha.Init(memIndex.size());
  // backward loop
  int i = max_i;
  do {
    alpha[i] = gamma[i]*la::Dot(this->s[i],d);
    la::AddExpert(-alpha[i], this->y[i], &d);
    if (--i < 0) i = memIndex.size()-1;
  } while (i != max_i);
  // forward loop, start with i == max_i
  double beta;
  do {
    if (++i >= memIndex.size()) i = 0;
    beta = gamma[i]*la::Dot(this->y[i],d);
    la::AddExpert(alpha[i]-beta, this->s[i], &d);
  } while (i != max_i);
}

END_OPTIM_NAMESPACE;

#endif // LBFGS_H
