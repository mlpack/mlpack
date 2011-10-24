/** @file pegasos.h
 *
 * This file implements Pegasos: Primal Estimated sub-GrAdient SOlver for SVM
 * For now, we do not implement the mini-batch version
 *
 * @author Nishant Mehta (niche)
 * @bug No known bugs
 */

#ifndef PEGASOS_H
#define PEGASOS_H

#define INSIDE_PEGASOS_H


using namespace arma;
using namespace std;


class Pegasos{
 private:
  vec w_;
  mat X_;
  vec y_;
  
  u32 n_points_;

  double lambda_;
  u32 T_;
  u32 k_;
  
  
 public:
  Pegasos();
  ~Pegasos() { }

  void Init(const mat& X, const vec& y,
	    double lambda, u32 T);
  
  void Init(const mat& X, const vec& y,
	    double lambda, u32 T, u32 k);
  
  void DoPegasos();
  
  void DoPegasosTrivialBatch();
  
  void DoPegasosMiniBatch();
  
  vec GetW();

  void Shuffle(uvec& numbers);
  
};

#include "pegasos_impl.h"
#undef INSIDE_PEGASOS_H

#endif
