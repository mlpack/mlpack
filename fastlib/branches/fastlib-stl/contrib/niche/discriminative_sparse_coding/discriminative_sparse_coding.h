/** @file discriminative_sparse_coding.h
 *
 *  This file implements Stochastic Gradient Algorithms for Discriminative Sparse Coding and Discriminative Local Coordinate Coding
 *
 *  @author Nishant Mehta (niche)
 *  @bug No known bugs, but no known completed code
 */

#ifndef DISCRIMINATIVE_SPARSE_CODING_H
#define DISCRIMINATIVE_SPARSE_CODING_H

#include <contrib/niche/lars/lars.h>

using namespace arma;
using namespace std;


class DiscriminativeSparseCoding {
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
  DiscriminativeSparseCoding() {
  }
  
  
  ~DiscriminativeSparseCoding() { }

  
  void Init(const mat& X, const vec& y, u32 n_atoms,
	    double lambda_1, double lambda_2) {
    X_ = X;
    y_ = y;
    
    n_dims_ = X.n_rows;
    n_points_ = X.n_cols;
    
    n_atoms_ = n_atoms;
    D_ = mat(n_dims_, n_atoms_);
    
    lambda_1_ = lambda_1;
    lambda_2_ = lambda_2;
  }
  
  
  void InitDictionary() {
    RandomInitDictionary();
  }
  
  
  void RandomInitDictionary() {
    for(u32 j = 0; j < n_atoms_; j++) {
      D_.col(j) = randu(n_dims_);
      D_.col(j) /= norm(D_.col(j), 2);
    }
  }
  
  
  void KMeansInitDictionary() {
    // need a constrained k-means algorithm to ensure each cluster is assigned at least one point
  }
  


  
  void SGDOptimize(u32 n_iterations, double step_size) {
    for(u32 t = 0; t < n_iterations; t++) {
      u32 ind = rand() % n_points_;
      SGDStep(X_.col(ind), step_size);
      // modify step size in some way
    }
  }
  

  void SGDStep(const vec& x, double step_size) {
    Lars lars;
    lars.Init(D_, x, true, lambda_1_, lambda_2_);
    lars.DoLARS();
    vec v;
    lars.Solution(v);
    
    
    // update to Dictionary
    
    

  }
};

#endif
