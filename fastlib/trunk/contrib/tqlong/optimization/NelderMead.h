#ifndef NELDERMEAD_H
#define NELDERMEAD_H

#include <fastlib/fastlib.h>

#ifndef BEGIN_OPTIM_NAMESPACE
#define BEGIN_OPTIM_NAMESPACE namespace optim {
#endif
#ifndef END_OPTIM_NAMESPACE
#define END_OPTIM_NAMESPACE }
#endif

#ifndef PARAMETER
#define PARAMETER
struct OptimizationParameter {
  int maxIter;  // maximum number of iterations
  double rTol;  // relative tolerance
  double aTol;  // absolute tolerance
  OptimizationParameter(int maxIter_, double rTol_, double aTol_)
    : maxIter(maxIter_), rTol(rTol_), aTol(aTol_) {}
};
#endif

BEGIN_OPTIM_NAMESPACE;

/**********************************************************************
  Implement 0-order Nelder-Mead simplex optimization method
  Function: implement CalculateValue(const variable_type&)
  Function::variable_type:
    implement la::AddExpert(double, const variable_type&, variable_type*)
              Copy(const variable_type&), CopyValues(const variable_type&)
              for this type
  Parameter:
    General params
      maxIter, rTol, aTol
    Specific for Nelder-Mead
      alpha : Reflection  (1.0)
      gamma : Expansion   (2.0)
      rho   : Contraction (0.5)
      sigma : Reduction   (0.5)
**********************************************************************/
template<typename Function> class NelderMead
{
public:
  typedef Function function_type;
  typedef typename Function::variable_type variable_type;
public:
  NelderMead(function_type& f_, OptimizationParameter param_ = OptimizationParameter(100, 0.01, 0.01));
  void setParam(double alpha_, double gamma_, double rho_, double sigma_) {
    alpha = alpha_; gamma = gamma_; rho = rho_; sigma = sigma_;
  }
  void addSeed(const ArrayList<variable_type>& vX);
  void add(const variable_type& x);
  double optimize(variable_type& sol);
protected:
  function_type& f;
  double alpha, gamma, rho, sigma;
  OptimizationParameter param;

  ArrayList<variable_type> memory;
  ArrayList<double> val;
  variable_type center;
  double center_val;

  void add(const variable_type& x, double v);
  void replace(int pos, const variable_type& x, double v);
  void updateAll();
  int findPos(double v);
  void updateCenter(const variable_type& x);
  void updateCenter(const variable_type& x1, const variable_type& x2);
};

template<typename F>
NelderMead<F>::NelderMead(function_type& f_, OptimizationParameter param_)
  : f(f_), param(param_)
{
  memory.Init();
  val.Init();
  f.Init(&center);
  alpha = 1.5;
  gamma = 2.0;
  rho = 0.5;
  sigma = 0.5;
}

template<typename F>
void NelderMead<F>::add(const variable_type &x)
{
  double v = f.CalculateValue(x);
  add(x, v);
}

template<typename F>
void NelderMead<F>::addSeed(const ArrayList<variable_type>& vX) {
  for (int i = 0; i < vX.size(); i++) add(vX[i]);
}

template<typename F>
void NelderMead<F>::add(const variable_type &x, double v)
{
  double pos = findPos(v);
  if (pos < memory.size()) {
    memory.InsertCopy(pos, x);
    val.InsertCopy(pos, v);
  }
  else {
    memory.PushBackCopy(x);
    val.PushBackCopy(v);
  }
  updateCenter(x);
}

template<typename F>
int NelderMead<F>::findPos(double v)
{
  int i = 0;
  while (i < val.size() && val[i] < v) i++;
  return i;
}

template<typename F>
void NelderMead<F>::updateCenter(const variable_type &x)
{
  if (memory.size() <= 1)
    center.CopyValues(x);
  else {
    variable_type tmp;
    tmp.Copy(center);
    int n = memory.size();
    la::AddExpert(-1.0/n, tmp, &center);
    la::AddExpert(1.0/n, x, &center);
  }
}

template<typename F>
void NelderMead<F>::updateCenter(const variable_type &x1, const variable_type &x2)
{
  int n = memory.size();
  DEBUG_ASSERT(n > 0);
  la::AddExpert(-1.0/n, x1, &center);
  la::AddExpert(1.0/n, x2, &center);
}

template<typename F>
double NelderMead<F>::optimize(variable_type &sol)
{
  int n = memory.size();
  DEBUG_ASSERT(n >= 3);

  double best_val = INFINITY;

  for (int iter = 0; iter < param.maxIter; iter++) {
    // Reflection step: x_ref = center + alpha ( center - worst )
    variable_type x_ref, x_diff;
    x_diff.Copy(center); x_ref.Copy(center);
    la::AddExpert(-1.0, memory[n-1], &x_diff);
    la::AddExpert(alpha, x_diff, &x_ref);
    double v_ref = f.CalculateValue(x_ref);

    if (val[0] <= v_ref && v_ref < val[n-2]) { // v_ref between the current best and second worst
      //printf("Reflect ... ");
      replace(n-1, x_ref, v_ref);
    } else if (v_ref < val[0]) { // Expansion step : v_ref better than the current best
      //printf("Expand  ... ");
      variable_type x_exp;
      x_exp.Copy(center);
      la::AddExpert(gamma, x_diff, &x_exp);
      double v_exp = f.CalculateValue(x_exp);
      if (v_exp < v_ref)
        replace(n-1, x_exp, v_exp);
      else
        replace(n-1, x_ref, v_ref);
    } else {                      // Contraction step : v_ref worse than the second worst
      //printf("Contrac ... ");
      variable_type x_con;
      x_con.Copy(memory[n-1]);
      la::AddExpert(rho, x_diff, &x_con);
      double v_con = f.CalculateValue(x_con);
      if (v_con < val[n-1])     // v_con better than the worst
        replace(n-1, x_con, v_con);
      else {                    // Reduction step: v_con worse than the current worst
        //printf("Reduct  ... ");
        for (int i = 1; i < n; i++) {
          variable_type tmp;
          tmp.Copy(memory[i]);
          la::AddExpert(1.0-sigma, memory[i], &tmp);
          la::AddExpert(1.0-sigma, memory[0], &tmp);
          memory[i].CopyValues(tmp);
          val[i] = f.CalculateValue(tmp);
        }
        updateAll();            // sort all values & recalculate center
      }
    } // all steps done
    if (best_val > val[0]) {
      sol.CopyValues(memory[0]);
      best_val = val[0];
    }
    // report progress
    //printf("iter = %d best_val = %f n_memory = %d\n", iter, best_val, memory.size());
  }
  return best_val;
}

template<typename F>
void NelderMead<F>::replace(int pos, const variable_type &x, double v)
{
  int n = memory.size();
  DEBUG_ASSERT(pos < n);
  updateCenter(memory[pos], x);
  if (pos > 0 && v < val[pos-1]) {  // insert on the left
    while (pos > 0 && v < val[pos-1]) {
      memory[pos] = memory[pos-1];
      val[pos] = val[pos-1];
      pos--;
    }
  } else if (pos < n-1 && v > val[pos+1]) { // insert on the right
    while (pos < n-1 && v > val[pos+1]) {
      memory[pos] = memory[pos+1];
      val[pos] = val[pos+1];
      pos++;
    }
  }
  memory[pos] = x;
  val[pos] = v;
}

template<typename F>
void NelderMead<F>::updateAll()
{
  int n = memory.size();
  DEBUG_ASSERT(n > 0);

  // First update the center
  center.CopyValues(memory[0]);
  la::AddExpert(1.0-1.0/n, memory[0], &center);
  for (int i = 1; i < n; i++)
    la::AddExpert(1.0/n, memory[i], &center);

  // Sort according to function values
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) if (val[i] > val[j]) {
      variable_type tmp; tmp.Copy(memory[i]); memory[i].CopyValues(memory[j]); memory[j].CopyValues(tmp);
      double tmp_val; tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
    }
  }
}

END_OPTIM_NAMESPACE;

#endif // NELDERMEAD_H
