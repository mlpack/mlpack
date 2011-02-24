#ifndef SPARSELA_H
#define SPARSELA_H

#include <cstdlib>

typedef unsigned long T_IDX; // type for feature indices
typedef double T_VAL; // type for feature values
typedef float T_LBL; // type for lables

class Feature {
 public:
  T_IDX widx_; // starts from 1
  T_VAL wval_;
};

class Example {
 public:
  // for a general sparse vector
  Feature *F_; // features
  char *ud_; // user defined info
  size_t n_nz_f_; // Number of nonzero features
  T_LBL label_;
  // for parallel computing
  bool in_use_;
};

#endif
