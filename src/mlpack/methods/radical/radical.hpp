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
  Radical(double noiseStdDev, size_t nReplicates, size_t nAngles,
	  size_t nSweeps);
  Radical(double noiseStdDev, size_t nReplicates, size_t nAngles,
	  size_t nSweeps, size_t m);
  
  //const arma::mat X() const { return matX; }
  
  void CopyAndPerturb(arma::mat& XNew, const arma::mat& matX);
  double Vasicek(const arma::vec& x);
  double DoRadical2D(const arma::mat& matX);
  void DoRadical(const arma::mat& matX, arma::mat& matY, arma::mat& matW);

  static void WhitenFeatureMajorMatrix(const arma::mat& matX,
				       arma::mat& matXWhitened,
				       arma::mat& matWhitening);
  
private:
  /**
   * standard deviation of the Gaussian noise added to the replicates of
   * the data points during Radical2D
   */
  double noiseStdDev;
  
  /**
   * Number of Gaussian-perturbed replicates to use (per point) in Radical2D
   */
  size_t nReplicates;
  
  /**
   * Number of angles to consider in brute-force search during Radical2D
   */
  size_t nAngles;
  
  /**
   * Number of sweeps
   *  - Each sweep calls Radical2D once for each pair of dimensions
   */
  size_t nSweeps;
  
  /**
   * m to use for Vasicek's m-spacing estimator of entropy
   */
  size_t m;
  
};

#endif
