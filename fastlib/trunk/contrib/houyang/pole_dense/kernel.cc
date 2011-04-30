
#include <cmath>
#include <armadillo>

#include "kernel.h"

//------------------Linear Kernel------------------//

LinearKernel::LinearKernel() {
  type_ = "linear";
}

LinearKernel::~LinearKernel() {
}

double LinearKernel::Eval(const Col<T_VAL>& a, const Col<T_VAL>& b) {
  return dot(a, b);
}

//-------------------RBF Kernel-------------------//
RBFKernel::RBFKernel() : sigma_(1.0){
  type_ = "rbf";
}

RBFKernel::RBFKernel(double sigma) : sigma_(sigma) {
  type_ = "rbf";
}

RBFKernel::~RBFKernel() {
}

double RBFKernel::Eval(const Col<T_VAL>& a, const Col<T_VAL>& b) {
  return exp( pow(norm(a-b, 2), 2) / (-2*pow(sigma_, 2)));
}
