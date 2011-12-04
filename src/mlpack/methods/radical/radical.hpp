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

using namespace arma;
using namespace std;

class Radical {
public:
  Radical();
  
  void Init(double noiseStdDev, u32 nReplicates, u32 nAngles, u32 nSweeps,
	    const mat& X);
  mat GetX();
  void WhitenX(mat& Whitening);
  void CopyAndPerturb(mat& XNew, const mat& X);
  double Vasicek(const vec& x);
  double DoRadical2D(const mat& X);
  void DoRadical(mat& Y, mat& W);
  
private:
  double noiseStdDev;
  u32 nReplicates;
  u32 nAngles;
  u32 nSweeps;
  u32 m; // for Vasicek's m-spacing estimator
  
  mat X;

};

#endif
