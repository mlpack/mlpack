#ifndef SPARSELA_H
#define SPARSELA_H

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

typedef unsigned long T_IDX; // type for feature indices
typedef double        T_VAL; // type for feature values
typedef float         T_LBL; // type for lables

//-------------------------Feature----------------------------//
class Feature {
 public:
  T_IDX i_; // starts from 0
  T_VAL v_;
 public:
  Feature();
  Feature(const T_IDX i, const T_VAL v);
  ~Feature();
};

//---------------------General Sparse Vector-----------------//
class Svector {
 public:
  // a general sparse vector
  vector<Feature> Fs_; // features
 public:
  Svector();
  Svector(const vector<Feature> Fs);
  Svector(const T_IDX n_f, const T_VAL c);
  ~Svector();
  // basics
  size_t Size();
  void SetAll(const T_VAL v);
  void SetAllResize(const T_IDX n_f, const T_VAL v);
  void PushBack(const Feature& F);
  void InsertOne(const T_IDX p, const Feature& F);
  void EraseOne(const T_IDX p);
  void Clear();
  void Print();
  // linear algebra
  Svector& operator=(const Svector& x);
  Svector& operator+=(const Svector& x);
  Svector& operator-=(const Svector& x);
  Svector& operator*=(const Svector& x);
  Svector& operator*=(const double a);
  Svector& operator/=(const double a);
  Svector& operator^=(const double p);
  double SparseDot(const Svector& x) const;
  double SparseSqL2Norm() const;
  void   SparseAddExpertOverwrite(double a, Svector *x);  
  void   SparseSubtract(Svector& p, Svector& n);
  void   SparseExpMultiplyOverwrite(Svector *x);
  void   SparseNegExpMultiplyOverwrite(Svector *x);
  // metrics
  double SparseSqEuclideanDistance(const Svector& x) const;
  // misc
  void   Shrink(double threshold);
};

//---------------------------Example-------------------------//
class Example : public Svector{
 public:
  T_LBL  y_; // label
  bool   in_use_; // for parallel computing
  string ud_; // user defined info
 public:
  Example();
  Example(const vector<Feature> Fs, const T_LBL y);
  Example(const vector<Feature> Fs, const T_LBL y, const string ud);
  ~Example();

  Example& operator=(const Example& x);
  void Clear();
  void Print();
};

//------------------Row Indexed Sparse Matrix---------------//
class SmatrixR {
 public:
  SmatrixR();
  ~SmatrixR();
  // a general sparse matrix, indexed by rows
  vector<Svector> Vs_; // rows
};

#endif
