
#include "opt_wm.h"

struct thread_par {
  size_t id_;
  WM *Lp_;
};

WM::WM() {
  cout << "---Distributed Weighted Majority---" << endl;
}

void WM::WmCommUpdate(size_t tid) {
}

// In Distributed WM, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
void* WM::WmThread(void *in_par) {
  return NULL;
}

void WM::Learn() {
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  // init parameters
  w_pool_.resize(n_thread_);
  m_pool_.resize(n_thread_);

  thread_par pars[n_thread_];
  for (size_t t = 0; t < n_thread_; t++) {
    // init thread parameters and statistics
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    w_pool_[t].Clear();
    t_state_[t] = 0;
    t_n_it_[t] = 0;
    t_n_used_examples_[t] = 0;
    t_loss_[t] = 0;
    t_err_[t] = 0;
    t_exp_err_[t] = 0;
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &WM::WmThread, (void*)&pars[t]);
  }

  FinishThreads();
  SaveLog();
  
}

void WM::Test() {
}

void WM::MakeLog(size_t tid, Example *x, double pred_val) {
}

void WM::SaveLog() {
}
