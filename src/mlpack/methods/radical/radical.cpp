/**
 * @file radical.cpp
 * @author Nishant Mehta
 *
 * Implementation of Radical class
 */

#include "radical.hpp"

using namespace arma;
using namespace std;

Radical::Radical() {
}

void Radical::Init(double noiseStdDev, u32 nReplicates, u32 nAngles, u32 nSweeps,
		   const mat& X) {
  this->noiseStdDev = noiseStdDev;
  this->nReplicates = nReplicates;
  this->nAngles = nAngles;
  this->nSweeps = nSweeps;
  this->X = mat(X); // is this the same as this.X = X ?
  m = floor(sqrt(X.n_rows));
}

mat Radical::GetX() {
  return X;
}

void Radical::WhitenX(mat& Whitening) {
  mat U, V;
  vec s;
  svd(U, s, V, cov(X));
  Whitening = U * diagmat(pow(s, -0.5)) * trans(V);
  Whitening.print("Whitening");
  X = X * Whitening;
}

void Radical::CopyAndPerturb(mat& XNew, const mat& X) {
  XNew = repmat(X, nReplicates, 1);
  XNew = XNew + noiseStdDev * randn(XNew.n_rows, XNew.n_cols);
}

double Radical::Vasicek(const vec& z) {
  vec v = sort(z);
  u32 nPoints = v.n_elem;
  vec logs = log(v.subvec(m, nPoints - 1) - v.subvec(0, nPoints - 1 - m));
  return (double) sum(logs);
}

double Radical::DoRadical2D(const mat& X) {
  mat XMod;
  CopyAndPerturb(XMod, X);

  double theta;
  double value;
  mat Jacobi(2,2);
  mat candidateY;

  double thetaOpt = 0;
  double valueOpt = 1e99;
  
  for(u32 i = 0; i < nAngles; i++) {
    theta = ((double) i) / ((double) nAngles) * M_PI / 2;
    Jacobi(0,0) = cos(theta);
    Jacobi(0,1) = sin(theta);
    Jacobi(1,0) = -sin(theta);
    Jacobi(1,1) = cos(theta);
    
    candidateY = XMod * Jacobi;
    value = Vasicek(candidateY.col(0)) + Vasicek(candidateY.col(1));
    if(value < valueOpt) {
      valueOpt = value;
      thetaOpt = theta;
    }
  }
  
  return thetaOpt;
}


void Radical::DoRadical(mat& Y, mat& W) {

  // X is nPoints by nDims (although less intuitive, this is for computational efficiency)
  
  u32 nDims = X.n_cols;
  
  mat Whitening;
  WhitenX(Whitening);
  
  // in the RADICAL code, they do not copy and perturb initial, although the
  // paper does. we follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima
  //GeneratePerturbedX(X, X);
  
  Y = X;
  W = eye(nDims, nDims);
  
  mat YSubspace(X.n_rows, 2);
  
  for(u32 sweepNum = 0; sweepNum < nSweeps; sweepNum++) {
    for(u32 i = 0; i < nDims - 1; i++) {
      for(u32 j = i + 1; j < nDims; j++) {
	YSubspace.col(0) = Y.col(i);
	YSubspace.col(1) = Y.col(j);
	mat WSubspace;
	double thetaOpt = DoRadical2D(YSubspace);
	mat J = eye(nDims, nDims);
	J(i,i) = cos(thetaOpt);
	J(i,j) = sin(thetaOpt);
	J(j,i) = -sin(thetaOpt);
	J(j,j) = cos(thetaOpt);
	W = W * J;
	Y = Y * W;
      }
    }
  }
  
  // the final transpose provides W in the typical form from the ICA literature
  W = trans(W * Whitening);
}
