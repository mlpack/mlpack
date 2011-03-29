#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "sparsela.h"
#include "maths.h"

class Transform {
 public:
  string type_;
  T_IDX d_; // dimension of original data
  T_IDX D_; // number of w samples, for i=1~D, w_i's dim is d_
  vector<Svector> w_;
 public:
  Transform();
  ~Transform();
};

/////////////////////////////////////////////
// Fourier Transform of Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
class FourierRBFTransform : public Transform {
 private:
  double sigma_;
  RandomNumber r_;
 public:
  FourierRBFTransform();
  FourierRBFTransform(T_IDX d, T_IDX D, double sigma);
  ~FourierRBFTransform();
  void SampleW();
  void Tr(const Svector &src, Svector &dest) const;
};

#endif
