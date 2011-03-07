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
  size_t n_source_;
  string fn_;
  FILE   *fp_;
  size_t port_;
  bool   random_;
  // data properties
  vector<Example> EXs_; // examples
  vector<size_t> rnd_i_; // index for randomly permuted examples
  size_t used_ct_; // counter for number of times examples are used
  // data file properties
  FileFormat ff_; 
  size_t n_ln_; // number of lines of example file
  T_IDX max_ft_idx_; // max index of features
  // for svmlight format
  size_t max_n_nz_ft_; // maximum number of non-0 features
  size_t max_l_ln_; // maximum length of a text line
  // data stream properties
 private:
  void InitFromSvmlight();
  void InitFromCsv();
  void InitFromArff();
  bool SorN(int c);
 public:
  Data(string fn, size_t port, bool random);
  size_t Size();
  bool ReadFileInfo();
  void ReadFromFile();
  void ReadFromPort();
  void RandomPermute();
  Example* GetExample(size_t idx); 
  void Print();
};


#endif
