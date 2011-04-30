
#include <cmath>

#include "transform.h"

Transform::Transform() {
  d_ = 0;
  D_ = 0;
}

Transform::~Transform() {
}

//-------------------Transform RBF Kernel-------------------//
FourierRBFTransform::FourierRBFTransform() {
  type_ = "fourier_rbf";
  sigma_ = 1.0; // default
}

FourierRBFTransform::FourierRBFTransform(T_IDX d, T_IDX D, double sigma) {
  type_ = "fourier_rbf";
  d_ = d;
  D_ = D;
  sigma_ = sigma;
}

FourierRBFTransform::~FourierRBFTransform() {
}

///////////////////////////////////////////////////////////
// Sample D d-dim w vectors from Gaussian pdf N(0, 1/sigma)
///////////////////////////////////////////////////////////
void FourierRBFTransform::SampleW() {
  double inv_sigma = 1.0 / sigma_;
  w_.set_size(d_, D_);
  for (T_IDX i=0; i<D_; i++) {
    for (T_IDX j=0; j<d_; j++) {
      w_(j, i) = (T_VAL)r_.RandGaussian(inv_sigma);
    }
  }
}

//////////////////////////////////////////
// Transform d-dim vector to D-dim vector
//////////////////////////////////////////
void FourierRBFTransform::Tr(const Col<T_VAL>& src, Col<T_VAL>& dest) const{
  T_IDX ws = w_.n_cols;
  dest.set_size(2 * ws);
  for (T_IDX i=0; i<ws; i++) {
    double dp = dot(src, w_.col(i));
    dest(i) = cos(dp);
    dest(ws + i) = sin(dp);
  }
}
