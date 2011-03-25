
#include <cmath>

#include "transform.h"

Transform::Transform() {
}

Transform::~Transform() {
}

//-------------------Transform RBF Kernel-------------------//
FourierRBFTransform::FourierRBFTransform() {
  type_ = "fourier_rbf";
  d_ = 0;
  D_ = 0;
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
void FourierRBFTransform::Tr(Svector *src, Svector *dest) {
  
}
