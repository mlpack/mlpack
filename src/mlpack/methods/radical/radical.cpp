/**
 * @file radical.cpp
 * @author Nishant Mehta
 *
 * Implementation of Radical class
 */

#include "radical.hpp"

//using namespace arma;
using namespace std;


Radical::Radical(double noiseStdDev, size_t nReplicates, size_t nAngles, 
		 size_t nSweeps) :
  noiseStdDev(noiseStdDev),
  nReplicates(nReplicates),
  nAngles(nAngles),
  nSweeps(nSweeps),
  m(-1)
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
  

void Radical::CopyAndPerturb(arma::mat& matXNew, const arma::mat& matX) {
  matXNew = arma::repmat(matX, nReplicates, 1);
  matXNew = matXNew + noiseStdDev * arma::randn(matXNew.n_rows, matXNew.n_cols);
}


double Radical::Vasicek(const arma::vec& z) {
  arma::vec v = sort(z);
  size_t nPoints = v.n_elem;
  arma::vec logs = arma::log(v.subvec(m, nPoints - 1) - v.subvec(0, nPoints - 1 - m));
  return (double) arma::sum(logs);
}


double Radical::DoRadical2D(const arma::mat& matX) {
  arma::mat matXMod;
  CopyAndPerturb(matXMod, matX);

  double theta;
  double value;
  arma::mat matJacobi(2,2);
  arma::mat candidateY;

  double thetaOpt = 0;
  double valueOpt = 1e99;
  
  for(size_t i = 0; i < nAngles; i++) {
    theta = ((double) i) / ((double) nAngles) * arma::math::pi() / 2;
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


void Radical::DoRadical(const arma::mat& matX, arma::mat& matY, arma::mat& matW) {
  
  // matX is nPoints by nDims (although less intuitive than columns being points,
  // and although this is the transpose of the ICA literature, this choice is 
  // for computational efficiency)
  
  if(m < 1) {
    m = floor(sqrt(matX.n_rows));
  }
  
  
  const size_t nDims = matX.n_cols;
  const size_t nPoints = matX.n_rows;
  
  arma::mat matXWhitened;
  arma::mat matWhitening;
  WhitenFeatureMajorMatrix(matX, matXWhitened, matWhitening);
  
  // in the RADICAL code, they do not copy and perturb initially, although the
  // paper does. we follow the code as it should match their reported results
  // and likely does a better job bouncing out of local optima
  //GeneratePerturbedX(X, X);
  
  matY = matXWhitened;
  matW = arma::eye(nDims, nDims);
  
  arma::mat matYSubspace(nPoints, 2);
  
  for(size_t sweepNum = 0; sweepNum < nSweeps; sweepNum++) {
    for(size_t i = 0; i < nDims - 1; i++) {
      for(size_t j = i + 1; j < nDims; j++) {
	matYSubspace.col(0) = matY.col(i);
	matYSubspace.col(1) = matY.col(j);
	arma::mat matWSubspace;
	double thetaOpt = DoRadical2D(matYSubspace);
	arma::mat matJ = arma::eye(nDims, nDims);
	matJ(i,i) = cos(thetaOpt);
	matJ(i,j) = sin(thetaOpt);
	matJ(j,i) = -sin(thetaOpt);
	matJ(j,j) = cos(thetaOpt);
	matW = matW * matJ;
	matY = matXWhitened * matW;
      }
    }
  }
  
  // the final transpose provides W in the typical form from the ICA literature
  matW = arma::trans(matWhitening * matW);
}


void Radical::WhitenFeatureMajorMatrix(const arma::mat& matX,
				       arma::mat& matXWhitened,
				       arma::mat& matWhitening) {
  arma::mat matU, matV;
  arma::vec s;
  arma::svd(matU, s, matV, cov(matX));
  matWhitening = matU * arma::diagmat(1 / sqrt(s)) * arma::trans(matV);
  matXWhitened = matX * matWhitening;
}
