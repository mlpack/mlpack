/** @file discr_sparse_coding.h
 *
 *  This file implements Stochastic Gradient Algorithms for Discriminative Sparse Coding and Discriminative Local Coordinate Coding
 *
 *  @author Nishant Mehta (niche)
 *  @bug No known bugs, but no known completed code
 */

#ifndef DISCR_SPARSE_CODING_H
#define DISCR_SPARSE_CODING_H

#define INSIDE_DISCR_SPARSE_CODING_H


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
  double lambda_w_; // regularization term for norm of hypothesis vector w
  
 public:
  DiscrSparseCoding() {
  }
  
  
  ~DiscrSparseCoding() { }

  
  void Init(const mat& X, const vec& y, u32 n_atoms,
	    double lambda_1, double lambda_2,
	    double lambda_w);
  
  void SetDictionary(const mat& D);
  
  void SetW(const vec& w);
  
  void InitDictionary();

  void InitDictionary(const char* dictionary_filename);
  
  void RandomInitDictionary();
  
  void KMeansInitDictionary();

  void InitW();

  void InitW(const char* w_filename);
  
  void SGDOptimize(u32 n_iterations, double step_size);
  
  void SGDStep(const vec& x, double y, double step_size);
  
  void ProjectW();

  void PrintDictionary();

  void GetDictionary(mat& D);

  void PrintW();
  
  void GetW(vec& w);
};

#include "discr_sparse_coding_impl.h"
#undef INSIDE_DISCR_SPARSE_CODING_H

#endif
