
#ifndef PASSIVE_AGGRESSIVE_H
#define PASSIVE_AGGRESSIVE_H

#include <fastlib/fastlib.h>

#ifndef IS_ZERO
#define IS_ZERO(x) ( (x) < 1e-16 && (x) > -1e-16 )
#endif

typedef double (*LossFunction)(const Vector& weight,const Vector& x,double y);

#include "kernel.h"               // Kernel definitions
#include "dataGenerator.h"        // Abstraction for generator of online data

/** Kernelized Weight: weight vector in RKHS represented by support vectors
    and corresponding coefficients
*/
struct KernelizedWeight {
  ArrayList<Vector> m_lSupportVectors;
  ArrayList<double> m_lCoefficients;
  KernelFunction& m_fKernel;
  index_t m_iDim;
public:
  KernelizedWeight(index_t dim, KernelFunction& func) : m_fKernel(func) {
    m_lSupportVectors.Init();
    m_lCoefficients.Init();
    //m_fKernel = func;
    m_iDim = dim;
  }
  index_t n_vectors() const { return m_lSupportVectors.size(); }
  index_t n_dim() const { return m_iDim; }
  void AddSupportVector(const Vector& x, double coeff) {
    DEBUG_ASSERT(m_iDim == x.length());
    m_lSupportVectors.PushBackCopy(x);
    m_lCoefficients.PushBackCopy(coeff);
  }
};

/** The dot product in RKHS between the weight and the sample
 */
namespace la {
  double Dot(const KernelizedWeight& weight, const Vector& x);
};

/** hinge_loss template function
    - using dot product of two vectors
    - using dot product in RKHS of two vectors
*/
template <class WEIGHT_TYPE>
double hinge_loss(const WEIGHT_TYPE& weight,const Vector& x,double y) {
  double loss = 1 - la::Dot(weight, x) * y;
  return (loss > 0) ? loss : 0;
}

/** Implement a PA, PA_I, PA_II updates on the ``weight'' 
    when seeing a sample and its label (x, y)
    - loss is a hinge loss
    - weight remains unchanged if no loss occur
    - otherwise,
      + PA: weight changes to nearest point that has zero loss on sample.
      + PA_I: add a L_1 soft loss, penalized by C
      + PA_II: add a L_2 soft loss, penalized by C
    Return: the loss
*/
double PA_Update(fx_module* module, const Vector& w_t,
                 const Vector& x_t, double y_t, Vector& w_out);

double PA_Update(fx_module* module, const Vector& w_t,
                 double* x_t, double y_t, Vector& w_out);

double PA_I_Update(fx_module* module, const Vector& w_t,
                   const Vector& x_t, double y_t, Vector& w_out);

double PA_I_Update(fx_module* module, const Vector& w_t,
                   double* x_t, double y_t, Vector& w_out);

double PA_II_Update(fx_module* module, const Vector& w_t,
                    const Vector& x_t, double y_t, Vector& w_out);

double PA_II_Update(fx_module* module, const Vector& w_t,
                    double* x_t, double y_t, Vector& w_out);

double Kernelized_PA_Update(fx_module* module, KernelizedWeight& w_t,
                            const Vector& x_t, double y_t);

double Kernelized_PA_Update(fx_module* module, KernelizedWeight& w_t,
                            double* x_t, double y_t);

double Kernelized_PA_I_Update(fx_module* module, KernelizedWeight& w_t,
                              const Vector& x_t, double y_t);

double Kernelized_PA_I_Update(fx_module* module, KernelizedWeight& w_t,
                              double* x_t, double y_t);

double Kernelized_PA_II_Update(fx_module* module, KernelizedWeight& w_t,
                               const Vector& x_t, double y_t);

double Kernelized_PA_II_Update(fx_module* module, KernelizedWeight& w_t,
                               double* x_t, double y_t);

#endif /* PASSIVE_AGGRESSIVE_H */
