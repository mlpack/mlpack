#ifndef OPT_OGD_H
#define OPT_OGD_H

#include "learner.h"

class OGD : public Learner {
 public:
  vector<Svector> w_pool_; // a pool that contains weight vectors for each thread
  vector<double> b_pool_; // a pool for bias term
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
};

#endif
