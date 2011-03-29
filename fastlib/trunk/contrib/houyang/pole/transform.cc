
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
  double inv_gamma = 1.0 / sigma_;
  w_.resize(D_);
  for (T_IDX i=0; i<D_; i++) {
    w_[i].SetAllResize(d_, 0.0);
    for (T_IDX j=0; j<d_; j++) {
      w_[i].Fs_[j].i_ = j;
      w_[i].Fs_[j].v_ = (T_VAL)r_.RandGaussian(inv_gamma);
    }
  }
}

//////////////////////////////////////////
// Transform d-dim vector to D-dim vector
//////////////////////////////////////////
void FourierRBFTransform::Tr(const Svector& src, Svector& dest) const{
  T_IDX ws = w_.size();
  dest.Resize(2 * ws);
  for (T_IDX i=0; i<ws; i++) {
    double dp = src.SparseDot(w_[i]); // src is typically sparser than w_[i]
    dest[i] = Feature(i, cos(dp));
    dest[ws + i] = Feature(ws+i, sin(dp));
  }
}
