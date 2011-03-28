#ifndef KERNEL_H
#define KERNEL_H

#include "sparsela.h"

/////////////////////////////////////////////
// Linear Kernel
// k(a,b) = a^T * b
/////////////////////////////////////////////
//class LinearKernel : public Kernel {
class LinearKernel {
 public:
  string type_;
 public:
  LinearKernel();
  ~LinearKernel();
  double Eval(const Svector& a, const Svector& b);
};

/////////////////////////////////////////////
// Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
//class RBFKernel : public Kernel {
class RBFKernel {
 private:
  string type_;
  double sigma_;
 public:
  RBFKernel();
  RBFKernel(double sigma);
  ~RBFKernel();
  double Eval(const Svector& a, const Svector& b);
};

#endif
