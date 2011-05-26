/** @file sparse_coding.h
 *
 *  This file implements the Sparse Coding (via l1-regularization)  (without the elastic net variation, for now...)
 *
 *  @author Nishant Mehta (niche)
 *  @bug Could be lots, let's see!
 */

#ifndef SPARSE_CODING_H
#define SPARSE_CODING_H

#define INSIDE_SPARSE_CODING_H


#include <contrib/niche/lars/lars.h>
#include <contrib/niche/tools/tools.h>

using namespace arma;
using namespace std;

class SparseCoding {
 private:
  u32 n_dims_;
  u32 n_atoms_;
  u32 n_points_;

  mat X_;

  mat D_;
  mat V_;
  
  double lambda_; // l_1 regularization term

 public:
  SparseCoding() { }

  ~SparseCoding() { }
  
  void Init(const mat& X, u32 n_atoms, double lambda);
  
  void SetDictionary(const mat& D);
  
  void InitDictionary();
  
  void InitDictionary(const char* dictionary_filename);
  
  void RandomInitDictionary();

  void DataDependentRandomInitDictionary();
  
  void RandomAtom(vec& atom);

  void DoSparseCoding(u32 n_iterations);

  void OptimizeCode();
  
  void OptimizeDictionary(uvec adjacencies);
  
  void ProjectDictionary();
  
  double Objective(uvec adjacencies);
  
  void GetDictionary(mat& D);

  void PrintDictionary();
    
  void GetCoding(mat& V);
  
  void PrintCoding();
  
};

#include "sparse_coding_impl.h"
#undef INSIDE_SPARSE_CODING_H

#endif
