#ifndef OPT_OEG_H
#define OPT_OEG_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "learner.h"

class OEG : public Learner {
 public:
  vector<Svector> w_p_pool_; // shared memory for weight vectors of each thread
  vector<Svector> w_n_pool_;
  vector<Svector> m_p_pool_; // shared memory for messages
  vector<Svector> m_n_pool_; // shared memory for messages
  vector<double>  b_p_pool_; // shared memory for bias term
  vector<double>  b_n_pool_; // shared memory for bias term
 private:
  double eta0_, t_init_;
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  OEG();
  //~OEG();
  void Learn();
  void Test();
 private:
  static void* OegThread(void *par);
  void OegCommUpdate(T_IDX tid);
  void MakeLog(T_IDX tid, Svector *w, double bias, Example *x, double pred_val);
  void SaveLog();
};

#endif

