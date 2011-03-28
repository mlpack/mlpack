
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
}

FourierRBFTransform::FourierRBFTransform(size_t d, size_t D) {
  type_ = "fourier_rbf";
  d_ = d;
  D_ = D;
}

FourierRBFTransform::~FourierRBFTransform() {
}

///////////////////////////////////////////////////////
// Sample D d-dim w vectors from Gaussian pdf N(0, 1)
///////////////////////////////////////////////////////
void FourierRBFTransform::SampleW() {
  w_.resize(D_);
  for (size_t i=0; i<D_; i++) {
    w_[i].SetAllResize(d_, 0.0);
    for (size_t j=0; j<d_; j++) {
      w_[i].Fs_[j].i_ = j;
      w_[i].Fs_[j].v_ = r_.RandGaussian(1.0);
    }
  }
}

//////////////////////////////////////////
// Transform d-dim vector to D-dim vector
//////////////////////////////////////////
void FourierRBFTransform::Tr(const Svector& src, Svector& dest) const{
  Ullong ws = w_.size();
  dest.Resize(2 * ws);
  for (Ullong i=0; i<ws; i++) {
    double dp = src.SparseDot(dest);
    dest[i] = Feature(i, cos(dp));
    dest[ws + i] = Feature(ws+i, sin(dp));
  }
}

