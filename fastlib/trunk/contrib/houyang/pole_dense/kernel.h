#ifndef KERNEL_H
#define KERNEL_H

#include <cstring>
#include <armadillo>

#include "datatypes.h"

using namespace std;
using namespace arma;

/////////////////////////////////////////////
// Linear Kernel
// k(a,b) = a^T * b
/////////////////////////////////////////////
//class LinearKernel : public Kernel {
class LinearKernel {
 public:
  double sigma_; // dummy
  string type_;
 public:
  LinearKernel();
  ~LinearKernel();
  double Eval(const Col<T_VAL>& a, const Col<T_VAL>& b);
};

/////////////////////////////////////////////
// Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
//class RBFKernel : public Kernel {
class RBFKernel {
 public:
  double sigma_;
  string type_;
 public:
  RBFKernel();
  RBFKernel(double sigma);
  ~RBFKernel();
  double Eval(const Col<T_VAL>& a, const Col<T_VAL>& b);
};

#endif
