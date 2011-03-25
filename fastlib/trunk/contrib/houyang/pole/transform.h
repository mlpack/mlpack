#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "sparsela.h"
#include "maths.h"

class Transform {
 public:
  string type_;
  size_t d_; // dimension of original data
  size_t D_; // number of w samples, for i=1~D, w_i's dim is d_
  vector<Svector> w_;
 public:
  Transform();
  ~Transform();
  virtual void SampleW() {};
  virtual void Tr(Svector *src, Svector *dest) {};
};

/////////////////////////////////////////////
// Fourier Transform of Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
class FourierRBFTransform : public Transform {
 private:
  RandomNumber r_;
 public:
  FourierRBFTransform();
  FourierRBFTransform(size_t d, size_t D);
  ~FourierRBFTransform();
  void SampleW();
  void Tr(Svector *src, Svector *dest);
};

#endif
