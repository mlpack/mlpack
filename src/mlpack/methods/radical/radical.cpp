/**
 * @file radical.cpp
 * @author Nishant Mehta
 *
 * Implementation of Radical class
 */

#include "radical.hpp"

using namespace std;
using namespace arma;

namespace mlpack {
namespace radical {


Radical::Radical(double noiseStdDev, size_t nReplicates, size_t nAngles, 
		 size_t nSweeps) :
  noiseStdDev(noiseStdDev),
  nReplicates(nReplicates),
  nAngles(nAngles),
  nSweeps(nSweeps),
  m(0)
{
  // nothing to do here
}

Radical::Radical(double noiseStdDev, size_t nReplicates, size_t nAngles, size_t nSweeps,
		 size_t m) :
  noiseStdDev(noiseStdDev),
  nReplicates(nReplicates),
  nAngles(nAngles),
  nSweeps(nSweeps),
  m(m)
{
  // nothing to do here
}
  

void Radical::CopyAndPerturb(mat& matXNew, const mat& matX) {
  matXNew = repmat(matX, nReplicates, 1);
  matXNew = matXNew + noiseStdDev * randn(matXNew.n_rows, matXNew.n_cols);
}


double Radical::Vasicek(const vec& z) {
  vec v = sort(z);
  size_t nPoints = v.n_elem;
  vec logs = log(v.subvec(m, nPoints - 1) - v.subvec(0, nPoints - 1 - m));
  return (double) sum(logs);
}


double Radical::DoRadical2D(const mat& matX) {
  mat matXMod;
  CopyAndPerturb(matXMod, matX);

  double theta;
  double value;
  mat matJacobi(2,2);
  mat candidateY;

  double thetaOpt = 0;
  double valueOpt = 1e99;
  
  for(size_t i = 0; i < nAngles; i++) {
    theta = ((double) i) / ((double) nAngles) * math::pi() / 2;
    matJacobi(0,0) = cos(theta);
    matJacobi(0,1) = sin(theta);
    matJacobi(1,0) = -sin(theta);
    matJacobi(1,1) = cos(theta);
    
    candidateY = matXMod * matJacobi;
    value = Vasicek(candidateY.col(0)) + Vasicek(candidateY.col(1));
    if(value < valueOpt) {
      valueOpt = value;
      thetaOpt = theta;
    }
  }
  
  return thetaOpt;
}


void Radical::DoRadical(const mat& matXT, mat& matY, mat& matW) {
  
  // matX is nPoints by nDims (although less intuitive than columns being 
  // points, and although this is the transpose of the ICA literature, this
  // choice is for computational efficiency when repeatedly generating 
  // two-dimensional coordinate projections for Radical2D
  mat matX = trans(matXT);
  
  // if m was not specified, initialize m as recommended in 
  // (Learned-Miller and Fisher, 2003)
  if(m < 1) {
    m = floor(sqrt(matX.n_rows));
  }
  
  
  const size_t nDims = matX.n_cols;
  const size_t nPoints = matX.n_rows;
  
  mat matXWhitened;
  mat matWhitening;
  WhitenFeatureMajorMatrix(matX, matXWhitened, matWhitening);
  
  // in the RADICAL code, they do not copy and perturb initially, although the
  // paper does. we follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima
  //GeneratePerturbedX(X, X);
  
  matY = matXWhitened;
  matW = eye(nDims, nDims);
  
  mat matYSubspace(nPoints, 2);
  
  for(size_t sweepNum = 0; sweepNum < nSweeps; sweepNum++) {
    for(size_t i = 0; i < nDims - 1; i++) {
      for(size_t j = i + 1; j < nDims; j++) {
	matYSubspace.col(0) = matY.col(i);
	matYSubspace.col(1) = matY.col(j);
	mat matWSubspace;
	double thetaOpt = DoRadical2D(matYSubspace);
	mat matJ = eye(nDims, nDims);
	matJ(i,i) = cos(thetaOpt);
	matJ(i,j) = sin(thetaOpt);
	matJ(j,i) = -sin(thetaOpt);
	matJ(j,j) = cos(thetaOpt);
	matW = matW * matJ;
	matY = matXWhitened * matW;
      }
    }
  }
  
  // the final transposes provide W and Y in the typical form from the ICA literature
  matW = trans(matWhitening * matW);
  matY = trans(matY);
}


void WhitenFeatureMajorMatrix(const mat& matX,
			      mat& matXWhitened,
			      mat& matWhitening) {
  mat matU, matV;
  vec s;
  svd(matU, s, matV, cov(matX));
  matWhitening = matU * diagmat(1 / sqrt(s)) * trans(matV);
  matXWhitened = matX * matWhitening;
}


}; // namespace radical
}; // namespace mlpack
