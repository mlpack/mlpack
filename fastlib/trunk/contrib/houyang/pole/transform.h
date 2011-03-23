#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "sparsela.h"

class Transform {
 public:
  string type_;
  size_t d_;
 public:
  Transform();
  ~Transform();
  
  virtual void Tr(Svector *src, Svector *dest) {};
};

/////////////////////////////////////////////
// Fourier Transform of Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
class FourierRBFTransform : public Transform {
 public:
  FourierRBFTransform();
  FourierRBFTransform(size_t d);
  ~FourierRBFTransform();
  void Tr(Svector *src, Svector *dest);
};

#endif
