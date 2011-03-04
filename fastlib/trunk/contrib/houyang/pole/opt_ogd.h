#ifndef OPT_OGD_H
#define OPT_OGD_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "learner.h"

class OGD : public Learner {
 public:
  vector<Svector> w_pool_; // shared memory for weight vectors of each thread
  vector<Svector> m_pool_; // shared memory for messages
  vector<double>  b_pool_; // shared memory for bias term
 private:
  double eta0_, t_init_;
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  OGD();
  void Learn();
  void Test();
 private:
  static void* OgdThread(void *par);
  void OgdCommUpdate(size_t tid);
  void MakeLog(size_t tid, Example *x, double pred_val);
  void SaveLog();
};

#endif
