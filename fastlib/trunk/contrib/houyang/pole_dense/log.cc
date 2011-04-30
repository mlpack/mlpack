
#include "log.h"

Log::Log() : n_thread_(0), n_log_(0), t_int_(0) {
}

Log::Log(T_IDX n_thread, T_IDX n_log, T_IDX t_int, 
         T_IDX n_expert, string opt_name) : 
  n_thread_(n_thread), n_log_(n_log), t_int_(t_int), 
  ct_t_(n_thread, 0), ct_lp_(n_thread, 0), 
  err_(n_thread, Vec_d(n_log, 0)),
  loss_(n_thread, Vec_f(n_log, 0.0)) {
  // init for expert-advice learners
  if (opt_name == "dwm_i" || opt_name ==  "dwm_a") {
    err_exp_.resize(n_expert);
    for (T_IDX i=0; i<n_expert; i++) {
      err_exp_[i] = Vec_d(n_log, 0);
    }
  }
}

