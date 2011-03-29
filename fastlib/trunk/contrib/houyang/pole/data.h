#ifndef DATA_H
#define DATA_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "sparsela.h"

using namespace std;

enum FileFormat {
  csv, arff, svmlight, unknown
};

class Data {
 public:
  // input parameters
  T_IDX n_source_;
  string fn_;
  FILE   *fp_;
  T_IDX port_;
  bool   random_;
  // data properties
  vector<Example> EXs_; // examples
  vector<T_IDX> rnd_i_; // index for randomly permuted examples
  T_IDX used_ct_; // counter for number of times examples are used
  // data file properties
  FileFormat ff_; 
  T_IDX n_ln_; // number of lines of example file
  T_IDX max_ft_idx_; // max index of features
  // for svmlight format
  T_IDX max_n_nz_ft_; // maximum number of non-0 features
  T_IDX max_l_ln_; // maximum length of a text line
  // data stream properties
 private:
  void InitFromSvmlight();
  void InitFromCsv();
  void InitFromArff();
  bool SorN(int c);
 public:
  Data(string fn, T_IDX port, bool random);
  T_IDX Size();
  bool ReadFileInfo();
  void ReadFromFile();
  void ReadFromPort();
  void RandomPermute();
  Example* GetExample(T_IDX idx); 
  void Print();
};


#endif
