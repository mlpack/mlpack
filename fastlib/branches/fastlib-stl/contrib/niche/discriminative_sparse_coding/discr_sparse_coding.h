/** @file discr_sparse_coding.h
 *
 *  This file implements Stochastic Gradient Algorithms for Discriminative Sparse Coding and Discriminative Local Coordinate Coding
 *
 *  @author Nishant Mehta (niche)
 *  @bug No known bugs, but no known completed code
 */

#ifndef DISCR_SPARSE_CODING_H
#define DISCR_SPARSE_CODING_H

#include <contrib/niche/lars/lars.h>

using namespace arma;
using namespace std;


class DiscrSparseCoding {
 private:
  mat X_;
  vec y_;
  u32 n_dims_;
  u32 n_points_;

  
  mat D_; // dictionary: each column is an atom
  vec w_; // hypothesis vector
  u32 n_atoms_;
  
  double lambda_1_; // l_1 regularization term
  double lambda_2_; // l_2 regularization term
  
 public:
  DiscrSparseCoding() {
  }
  
  
  ~DiscrSparseCoding() { }

  
  void Init(const mat& X, const vec& y, u32 n_atoms,
	    double lambda_1, double lambda_2);
  
  void InitDictionary();
  
  void RandomInitDictionary();
  
  void KMeansInitDictionary();
  
  void SGDOptimize(u32 n_iterations, double step_size);
  
  void SGDStep(const vec& x, double step_size);
};

#endif
