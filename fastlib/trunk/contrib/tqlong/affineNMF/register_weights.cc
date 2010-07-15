#include <fastlib/fastlib.h>
#include "image_register.h"

void ComputeContribution(const ImageType& X, const ImageType& Ij, 
			 index_t iX, index_t iI, double wj, double& dwj) {
  PointType wpj(Ij.pList[iI]);
  wpj.f *= wj;
  double dr, dc, df;
  d_exp_kernel(X.pList[iX], wpj, dr, dc, df);
  dwj += df * Ij.pList[iI].f;
}

void register_weights(const ImageType& X, const ArrayList<ImageType>& I,
		      const Vector& wInit, Vector& wOut) {
  DEBUG_ASSERT(wInit.length() == I.size());
  wOut.Copy(wInit);
  index_t maxIter = 1;
  double lambda = 0.1;
  index_t n_basis = I.size();
  for (index_t iter = 0; iter < maxIter; iter++) {
    Vector dw;
    dw.Init(n_basis); dw.SetZero();
    for (index_t j = 0; j < n_basis; j++)
      for (index_t iX = 0; iX < X.n_points(); iX++)
	for (index_t iI = 0; iI < I[j].n_points(); iI++)
	  ComputeContribution(X, I[j], iX, iI, wOut[j], dw[j]);
    la::AddExpert(-lambda, dw, &wOut);
    for (index_t j = 0; j < n_basis; j++) // projection onto R+
      if (wOut[j] < 0.0) wOut[j] = 0.0;
  }
}

void register_weights(const ImageType& X, const ArrayList<ImageType>& I,
		      Vector& wOut) {
  DEBUG_ASSERT(wOut.length() == I.size());
  index_t maxIter = 10;
  double lambda = 0.1;
  index_t n_basis = I.size();
  for (index_t iter = 0; iter < maxIter; iter++) {
    Vector dw;
    dw.Init(n_basis); dw.SetZero();
    for (index_t j = 0; j < n_basis; j++)
      for (index_t iX = 0; iX < X.n_points(); iX++)
	for (index_t iI = 0; iI < I[j].n_points(); iI++)
	  ComputeContribution(X, I[j], iX, iI, wOut[j], dw[j]);
    la::AddExpert(-lambda, dw, &wOut);
    for (index_t j = 0; j < n_basis; j++) // projection onto R+
      if (wOut[j] < 0.0) wOut[j] = 0.0;
  }
}
