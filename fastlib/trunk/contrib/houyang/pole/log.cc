
#include "log.h"

Log::Log() : n_thread_(0), n_log_(0), t_int_(0) {
}

Log::Log(size_t n_thread, size_t n_log, size_t t_int) : 
  n_thread_(n_thread), n_log_(n_log), t_int_(t_int), 
  ct_t_(n_thread, 0), ct_lp_(n_thread, 0), 
  err_(n_thread, Vec_d(n_log, 0)),
  loss_(n_thread, Vec_f(n_log, 0.0)) {
}

