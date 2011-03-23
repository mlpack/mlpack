
#include <cmath>

#include "transform.h"

Transform::Transform() {
}

Transform::~Transform() {
}

//-------------------Transform RBF Kernel-------------------//
FourierRBFTransform::FourierRBFTransform() {
  type_ = "fourier_rbf";
  d_ = 1000;
}

FourierRBFTransform::FourierRBFTransform(size_t d) {
  type_ = "fourier_rbf";
  d_ = d;
}

FourierRBFTransform::~FourierRBFTransform() {
}

void FourierRBFTransform::Tr(Svector *src, Svector *dest) {
  
}
