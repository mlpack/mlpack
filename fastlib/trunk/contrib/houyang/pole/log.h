#ifndef LOG_H
#define LOG_H

#include <vector>
#include <iostream>

#include "sparsela.h"

using namespace std;

typedef vector<size_t> Vec_d;
typedef vector<double> Vec_f;

class Log {
 public:
  size_t n_thread_;
  size_t n_log_;
  size_t t_int_;
  bool calc_loss_;
  string type_;
  // statistics
  vector<size_t> ct_t_;  // counters for round t
  vector<size_t> ct_lp_; // counters for log points
  vector<Vec_d>  err_;  // number of errors
  vector<Vec_f>  loss_;  // loss
  // for expert-advice learners
  vector<Vec_d>  err_exp_; // number of errors for experts
 public:
  Log();
  Log(size_t n_thread, size_t n_log, size_t t_int, 
      size_t n_expert, string opt_method);
};

#endif
