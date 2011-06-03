#ifndef OPT_RWM_H
#define OPT_RWM_H

#include <armadillo>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "learner.h"
#include "weak_learner.h"

using namespace arma;

class RWM : public Learner {
 public:
  vector<Svector> w_pool_; // shared memory for weight vectors of each thread
  vector<Svector> m_pool_; // shared memory for messages
  vector<Svector> old_m_pool_; // shared memory for messages received in last iteration
  vector<Svector> accu_msg_; // accumulated msgs
  //  int **adj_m_; // adjacency matrix
  //vector< vector<int> > adj_m_;
 private:
  Mat<int> adj_m_;
  vector<WeakLearner*> WLs_; // weak learners
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  RWM();
  //~RWM();
  void Learn();
  void Test();
 private:
  void TrainWeak();
  static void* RwmThread(void *par);
  void RwmCommUpdate(T_IDX tid);
  void MakeLog(T_IDX tid, T_LBL true_lbl, T_LBL pred_lbl, 
               vector<T_LBL> &exp_pred);
  void SaveLog();
};

#endif
