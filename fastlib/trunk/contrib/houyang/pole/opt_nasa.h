#ifndef OPT_NASA_H
#define OPT_NASA_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "learner.h"

class NASA : public Learner {
 public:
  vector<Svector> w_pool_; // shared memory for weight vectors of each thread
  vector<Svector> w_avg_pool_; // shared memory for averaged weight vec over iterations
  vector<Svector> m_pool_; // shared memory for messages
  vector<double>  b_pool_; // shared memory for bias term
  vector<double>  sum_gdsq_pool_;
  vector<double>  max_gdsq_pool_;
  vector<Svector> avg_gd_; // sample-averaged subgradient
  vector<double>  sum_deltasq_pool_;
  vector<double>  max_deltasq_pool_;
 private:
  //double eta0_, t_init_;
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  NASA();
  //~NASA();
  void Learn();
  void Test();
 private:
  static void* LearnThread(void *par); // for learning
  static void* TestThread(void *par); // for prediction
  void CommUpdate(T_IDX tid);
  void MakeLearnLog(T_IDX tid, Example *x, double pred_val);
  void MakeTestLog(T_IDX tid, Example *x, double pred_val);
  void SaveLearnLog();
  void SaveTestLog();
};

#endif
