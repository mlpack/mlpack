#ifndef SPARSELA_H
#define SPARSELA_H

#include <cstdio>
#include <cstdlib>
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
  Svector(vector<Feature> Fs);
  Svector(T_IDX n_f, T_VAL c);
  ~Svector();

  void Copy(Svector V);
  void SetAll(T_VAL v);
  void SetAllResize(T_IDX n_f, T_VAL v);
  void PushBack(Feature F);
  void InsertOne(T_IDX p, Feature F);
  void EraseOne(T_IDX p);
  void Clear();
  void Print();
  // linear algebra
  void   SparseScaleOverwrite(double a);
  double SparseDot(Svector *v);
  double SparseSqL2Norm();
  void   SparseAddExpertOverwrite(double a, Svector *x);
  void   SparseAddOverwrite(Svector *x);
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
  Example(vector<Feature> Fs, T_LBL y);
  Example(vector<Feature> Fs, T_LBL y, string ud);
  ~Example();

  void Copy(Example X);
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
