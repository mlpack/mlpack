#ifndef LOG_H
#define LOG_H

#include <vector>
#include <iostream>

#include "sparsela.h"

using namespace std;

typedef vector<T_IDX> Vec_d;
typedef vector<double> Vec_f;

class Log {
 public:
  T_IDX  n_thread_;
  T_IDX  n_log_;
  T_IDX  t_int_;
  bool   calc_loss_;
  string type_;
  // statistics
  vector<T_IDX> ct_t_;  // counters for round t
  vector<T_IDX> ct_lp_; // counters for log points
  vector<Vec_d>  err_;  // number of errors
  vector<Vec_f>  loss_;  // loss
  // for expert-advice learners
  vector<Vec_d>  err_exp_; // number of errors for experts
 public:
  Log();
  Log(T_IDX n_thread, T_IDX n_log, T_IDX t_int, 
      T_IDX n_expert, string opt_method);
};

#endif
