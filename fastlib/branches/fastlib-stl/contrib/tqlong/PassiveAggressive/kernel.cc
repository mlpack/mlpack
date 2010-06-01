
#include <fastlib/fastlib.h>
#include "kernel.h"

double LinearKernel::operator()(const Vector& x, const Vector& y) {
  return la::Dot(x, y);
}

double PolynomialKernel::operator()(const Vector& x, const Vector& y) {
  return pow(la::Dot(x, y)+(m_bHomogeneous ? 0 : 1), m_iPolyOrder);
}

double Gaussian2Kernel::operator()(const Vector& x, const Vector& y) {
  return exp(-0.5*la::DistanceSqEuclidean(x, y)/m_dSigma2);
}

