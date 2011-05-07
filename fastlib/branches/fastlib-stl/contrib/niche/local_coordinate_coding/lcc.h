/** @file lcc.h
 *
 *  This file implements the original Local Coordinate Coding algorithm (without the elastic net variation, for now...)
 *
 *  @author Nishant Mehta (niche)
 *  @bug Could be lots, let's see!
 */

#ifndef LCC_H
#define LCC_H

#define INSIDE_LCC_H


#include <contrib/niche/lars/lars.h>
#include <contrib/niche/tools/tools.h>

using namespace arma;
using namespace std;

class LocalCoordinateCoding {
 private:
  u32 n_dims_;
  u32 n_atoms_;
  u32 n_points_;

  mat X_;

  mat D_;
  mat V_; // should the code vectors be part of the class? they sort of are just transient...
  
  double lambda_; // l_1 regularization term

 public:
  LocalCoordinateCoding() { }

  ~LocalCoordinateCoding() { }
  
  void Init(const mat& X, u32 n_atoms, double lambda);
  
  void SetDictionary(mat D);
  
  void InitDictionary();
  
  void InitDictionary(const char* dictionary_filename);
  
  void RandomInitDictionary();

  void DataDependentRandomInitDictionary();
  
  void RandomAtom(vec& atom);

  void DoLCC(u32 n_iterations);

  void OptimizeCode();
  
  void OptimizeDictionary(uvec adjacencies);
  
  double Objective(uvec adjacencies);
  
  void GetDictionary(mat& D);

  void PrintDictionary();
    
  void GetCoding(mat& V);
  
  void PrintCoding();
  
};

#include "lcc_impl.h"
#undef INSIDE_LCC_H

#endif
