#ifndef KERNEL_H
#define KERNEL_H

#include "sparsela.h"

class Kernel {
 public:
  string type_;
 public:
  Kernel();
  ~Kernel();
  
  virtual double Eval(Svector *a, Svector *b) = 0;
};

/////////////////////////////////////////////
// Linear Kernel
// k(a,b) = a^T * b
/////////////////////////////////////////////
class LinearKernel : public Kernel {
 public:
  LinearKernel();
  ~LinearKernel();
  double Eval(Svector *a, Svector *b);
};

/////////////////////////////////////////////
// Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
class RBFKernel : public Kernel {
 private:
  double sigma_;
 public:
  RBFKernel();
  RBFKernel(double sigma);
  ~RBFKernel();
  double Eval(Svector *a, Svector *b);
};

#endif
