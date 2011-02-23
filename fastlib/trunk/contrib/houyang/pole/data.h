#ifndef DATA_H
#define DATA_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

using namespace std;

enum FileFormat {
  csv, arff, svmlight, unknown
};

//---------------------Dataset---------------------------//
class Dataset {
 public:
  // input parameters
  size_t n_source_;
  string fn_;
  FILE   *fp_;
  FileFormat ff_; 
  size_t port_;
  bool   random_;
  // data file properties
  size_t n_sp_; // number of examples
  size_t max_n_ft_; // maximum number of features
  size_t max_l_ln_; // maximum length of a text line
  // data stream properties
 public:
  Dataset(string fn, size_t port, bool random);
  bool ReadFileInfo();
  void InitFromSvmlight();
  void InitFromCsv();
  void InitFromArff();

  void ReadFromFile();
  void ReadFromPort();
  
  bool GetExample();
};

//---------------------Data---------------------------//
class Data {
 public:
  Dataset *TR_; // training set
  Dataset *VA_; // validation set
  Dataset *TE_; // testing set
  
 public:
  Data();
  ~Data();
};

#endif
