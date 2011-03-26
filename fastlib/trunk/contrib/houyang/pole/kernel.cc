
#include <cmath>

#include "kernel.h"

Kernel::Kernel() {
}

Kernel::~Kernel() {
}

//------------------Linear Kernel------------------//

LinearKernel::LinearKernel() {
  type_ = "linear";
}

LinearKernel::~LinearKernel() {
}

double LinearKernel::Eval(const Svector& a, const Svector& b) {
  return a.SparseDot(b);
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

double RBFKernel::Eval(const Svector& a, const Svector& b) {
  return exp( a.SparseSqEuclideanDistance(b) / (-2*pow(sigma_, 2)));
}
