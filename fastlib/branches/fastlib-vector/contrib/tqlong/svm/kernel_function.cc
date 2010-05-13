#include <fastlib/fastlib.h>
#include "svm.h"

namespace SVMLib {

KernelFunction::KernelFunction(TYPE type_, double param) : type(type_) {
  switch (type) {
  case LINEAR:
    break;
  case POLYNOMIAL:
    d = param;
    break;
  case RBF:
    sigma2 = -0.5/(param*param);
    printf("sigma2 = %f\n", sigma2);
    break;
  default:
    DEBUG_ASSERT(0); // TODO
    break;
  }
}

KernelFunction::KernelFunction(KernelFunc kfunc_) : type(CUSTOM) {
  kfunc = kfunc_;
}

double KernelFunction::operator() (const Vector& x, const Vector& y) {
  DEBUG_ASSERT(y.length() == x.length());
  double s = 0;
  switch (type) {
  case LINEAR:
    return la::Dot(x, y);
    break;
  case POLYNOMIAL:
    s = la::Dot(x, y);
    return s*pow(s+1, d-1);
    break;
  case RBF:
    for (index_t i = 0; i < x.length(); i++) s += (y[i]-x[i])*(y[i]-x[i]);
    return exp(s*sigma2);
    break;
  case CUSTOM:
    return kfunc(x, y);
    break;
  default:
    DEBUG_ASSERT(0); // TODO
    return 0;
    break;
  }
}

};
