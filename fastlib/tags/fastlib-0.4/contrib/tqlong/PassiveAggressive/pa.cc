
#include <fastlib/fastlib.h>
#include "pa.h"

namespace la {
  double Dot(const KernelizedWeight& weight, const Vector& x) {
    double s = 0;
    for (index_t i_vec = 0; i_vec < weight.n_vectors(); i_vec++) {
      s += weight.m_lCoefficients[i_vec] *
        weight.m_fKernel(weight.m_lSupportVectors[i_vec], x);
    }
    return s;
  }
};

double LengthEuclideanSquare(const Vector& x) {
  double s = 0;
  for (index_t i = 0; i < x.length(); i++)
    s += x[i]*x[i];
  return s;
}

/** Implement a PA update on the ``weight'' 
    when seeing a sample and its label (x, y)
    - loss is a hinge loss
    - weight remains unchanged if no loss occur
    - otherwise, weight changes to nearest point that has zero loss on sample.
    Return: the loss
 */

double PA_Update(fx_module* module, const Vector& w_t,
                 const Vector& x_t, double y_t, Vector& w_out) {
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / LengthEuclideanSquare(x_t);
  w_out.Copy(w_t);
  la::AddExpert(tau*y_t, x_t, &w_out);
  return loss_t;
}

double PA_Update(fx_module* module, const Vector& w_t,
                 double* x_t, double y_t, Vector& w_out) {
  index_t n = w_t.length();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return PA_Update(module, w_t, X_t, y_t, w_out);
}

double PA_Update_Overwrite(fx_module* module, Vector& w_t,
			   const Vector& x_t, double y_t) {
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / LengthEuclideanSquare(x_t);
  la::AddExpert(tau*y_t, x_t, &w_t);
  return loss_t;
}

double PA_I_Update(fx_module* module, const Vector& w_t,
                   const Vector& x_t, double y_t, Vector& w_out) {
  double C = fx_param_double(module, "C", -1);
  DEBUG_ASSERT(C >= 0);
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / LengthEuclideanSquare(x_t);
  if (tau > C) tau = C;
  w_out.Copy(w_t);
  la::AddExpert(tau*y_t, x_t, &w_out);
  return loss_t;
}

double PA_I_Update(fx_module* module, const Vector& w_t,
                   double* x_t, double y_t, Vector& w_out) {
  index_t n = w_t.length();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return PA_I_Update(module, w_t, X_t, y_t, w_out);
}

double PA_I_Update_Overwrite(fx_module* module, Vector& w_t,
			     const Vector& x_t, double y_t) {
  double C = fx_param_double(module, "C", -1);
  DEBUG_ASSERT(C >= 0);
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / LengthEuclideanSquare(x_t);
  if (tau > C) tau = C;
  la::AddExpert(tau*y_t, x_t, &w_t);
  return loss_t;
}

double PA_II_Update(fx_module* module, const Vector& w_t,
                    const Vector& x_t, double y_t, Vector& w_out) {
  double C = fx_param_double(module, "C", -1);
  DEBUG_ASSERT(C >= 0);
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / (LengthEuclideanSquare(x_t) + 0.5/C);
  w_out.Copy(w_t);
  la::AddExpert(tau*y_t, x_t, &w_out);
  return loss_t;
}

double PA_II_Update(fx_module* module, const Vector& w_t,
                    double* x_t, double y_t, Vector& w_out) {
  index_t n = w_t.length();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return PA_II_Update(module, w_t, X_t, y_t, w_out);
}

double PA_II_Update_Overwrite(fx_module* module, Vector& w_t,
			      const Vector& x_t, double y_t) {
  double C = fx_param_double(module, "C", -1);
  DEBUG_ASSERT(C >= 0);
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / (LengthEuclideanSquare(x_t) + 0.5/C);
  la::AddExpert(tau*y_t, x_t, &w_t);
  return loss_t;
}

double Kernelized_PA_Update(fx_module* module, KernelizedWeight& w_t,
                            const Vector& x_t, double y_t) {
  double loss_t = hinge_loss(w_t, x_t, y_t);
  if (IS_ZERO(loss_t)) return 0;               // no change when zero loss
  double tau = loss_t / w_t.m_fKernel(x_t, x_t);
  w_t.AddSupportVector(x_t, tau*y_t);
  return loss_t;
}

double Kernelized_PA_Update(fx_module* module, KernelizedWeight& w_t,
                            double* x_t, double y_t) {
  index_t n = w_t.n_dim();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return Kernelized_PA_Update(module, w_t, X_t, y_t);
}

double Kernelized_PA_I_Update(fx_module* module, KernelizedWeight& w_t,
                              const Vector& x_t, double y_t) {
  double C = fx_param_double(module, "C", -1);
  DEBUG_ASSERT(C >= 0);
  double loss_t = hinge_loss(w_t, x_t, y_t);
  if (IS_ZERO(loss_t)) return 0;               // no change when zero loss
  double tau = loss_t / w_t.m_fKernel(x_t, x_t);
  if (tau > C) tau = C;
  w_t.AddSupportVector(x_t, tau*y_t);
  return loss_t;
}

double Kernelized_PA_I_Update(fx_module* module, KernelizedWeight& w_t,
                              double* x_t, double y_t) {
  index_t n = w_t.n_dim();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return Kernelized_PA_I_Update(module, w_t, X_t, y_t);
}

double Kernelized_PA_II_Update(fx_module* module, KernelizedWeight& w_t,
                               const Vector& x_t, double y_t) {
  double C = fx_param_double(module, "C", -1);
  DEBUG_ASSERT(C >= 0);
  double loss_t = hinge_loss(w_t, x_t, y_t);
  if (IS_ZERO(loss_t)) return 0;               // no change when zero loss
  double tau = loss_t / (w_t.m_fKernel(x_t, x_t) + 0.5/C);
  w_t.AddSupportVector(x_t, tau*y_t);
  return loss_t;
}

double Kernelized_PA_II_Update(fx_module* module, KernelizedWeight& w_t,
                               double* x_t, double y_t) {
  index_t n = w_t.n_dim();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return Kernelized_PA_II_Update(module, w_t, X_t, y_t);
}
