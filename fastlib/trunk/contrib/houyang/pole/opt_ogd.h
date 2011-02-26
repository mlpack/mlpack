#ifndef OPT_OGD_H
#define OPT_OGD_H

#include "learner.h"

class OGD : public Learner {
 private:
  double eta0_, t_init_;
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  OGD();
  void Learn();
  void Test();
  void *OgdThread(void *in_par);
};

#endif
