#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <cstring>
#include <armadillo>

#include "maths.h"
#include "datatypes.h"

using namespace std;
using namespace arma;

class Transform {
 public:
  string type_;
  T_IDX d_; // dimension of original data
  T_IDX D_; // number of w samples, for i=1~D, w_i's dim is d_
  Mat<T_VAL> w_;
 public:
  Transform();
  ~Transform();
};

/////////////////////////////////////////////
// Fourier Transform of Gaussian RBF Kernel
// k(a,b) = exp(-||a-b||_2^2/(2sigma^2))
/////////////////////////////////////////////
class FourierRBFTransform : public Transform {
 public:
  double sigma_;
 private:
  RandomNumber r_;
 public:
  FourierRBFTransform();
  FourierRBFTransform(T_IDX d, T_IDX D, double sigma);
  ~FourierRBFTransform();
  void SampleW();
  void Tr(const Col<T_VAL> &src, Col<T_VAL> &dest) const;
};

#endif
