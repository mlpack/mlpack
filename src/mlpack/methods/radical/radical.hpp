/**
 * @file radical.hpp
 * @author Nishant Mehta
 *
 * Declaration of Radical class (RADICAL is Robust, Accurate, Direct ICA aLgorithm)
 * Note: upper case variables correspond to matrices. do not convert them to 
 *   camelCase because that would make them appear to be vectors (which is bad!)
 */

#ifndef __MLPACK_METHODS_RADICAL_RADICAL_HPP
#define __MLPACK_METHODS_RADICAL_RADICAL_HPP

#include<armadillo>
#include <stdio.h>
#include<float.h>

using namespace std;

class Radical {
public:
  Radical(double noiseStdDev, size_t nReplicates, size_t nAngles, size_t nSweeps,
	  const arma::mat& matX);

  arma::mat X() {
    return matX;
  }
  
  void CopyAndPerturb(arma::mat& XNew, const arma::mat& matX);
  double Vasicek(const arma::vec& x);
  double DoRadical2D(const arma::mat& matX);
  void DoRadical(arma::mat& matY, arma::mat& matW);
  
private:
  double noiseStdDev;
  size_t nReplicates;
  size_t nAngles;
  size_t nSweeps;
  size_t m; // for Vasicek's m-spacing estimator
  
  arma::mat matX;
  
  void WhitenX(arma::mat& matWhitening);
  
};

#endif
