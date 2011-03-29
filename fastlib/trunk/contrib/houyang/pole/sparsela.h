#ifndef SPARSELA_H
#define SPARSELA_H

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

#include "datatypes.h"

using namespace std;

//-------------------------Feature----------------------------//
class Feature {
 public:
  T_IDX i_; // starts from 0
  T_VAL v_;
 public:
  Feature();
  Feature(T_IDX i, T_VAL v);
  ~Feature();
};

//---------------------General Sparse Vector-----------------//
class Svector {
 public:
  // a general sparse vector
  vector<Feature> Fs_; // features
 public:
  Svector();
  Svector(const vector<Feature> &Fs);
  Svector(T_IDX n_f, T_VAL c);
  ~Svector();
  // basics
  T_IDX Size() const;
  void SetAll(T_VAL v);
  void SetAllResize(T_IDX n_f, T_VAL v);
  void Resize(T_IDX n_f);
  void Reserve(T_IDX n_f);
  void PushBack(const Feature &F);
  void InsertOne(T_IDX p, const Feature &F);
  void EraseOne(T_IDX p);
  void Clear();
  void Print();
  // linear algebra
  Svector& operator=(const Svector &x);
  Svector& operator+=(const Svector &x);
  Svector& operator-=(const Svector &x);
  Svector& operator*=(const Svector &x);
  Svector& operator*=(double a);
  Svector& operator/=(double a);
  Svector& operator^=(double p);
  Feature& operator[](T_IDX);
  void   SparseAddExpertOverwrite(double a, const Svector &x);  
  void   SparseSubtract(const Svector& p, const Svector &n);
  double SparseDot(const Svector &x) const;
  void   SparseExpMultiplyOverwrite(const Svector &x);
  void   SparseNegExpMultiplyOverwrite(const Svector &x);
  // metrics
  double SparseSqL2Norm() const;
  double SparseSqEuclideanDistance(const Svector &x) const;
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
  Example(const vector<Feature> &Fs, T_LBL y);
  Example(const vector<Feature> &Fs, T_LBL y, const string &ud);
  ~Example();

  Example &operator=(const Example &x);
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
