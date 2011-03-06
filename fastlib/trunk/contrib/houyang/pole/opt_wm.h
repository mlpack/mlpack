#ifndef OPT_WM_H
#define OPT_WM_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "learner.h"
#include "weak_learner.h"

class WM : public Learner {
 public:
  vector<Svector> w_pool_; // shared memory for weight vectors of each thread
  vector<Svector> m_pool_; // shared memory for messages
  vector<WeakLearner*> WLs_; // weak learners
 private:
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  WM();
  ~WM();
  void Learn();
  void Test();
 private:
  void TrainWeak();
  static void* WmThread(void *par);
  void WmCommUpdate(size_t tid);
  void MakeLog(size_t tid, T_LBL true_lbl, T_LBL pred_lbl, 
               vector<T_LBL> &exp_pred);
  void SaveLog();
};

#endif
