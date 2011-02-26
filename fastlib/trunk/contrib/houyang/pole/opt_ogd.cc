
#include "opt_ogd.h"

OGD::OGD() {
  cout << "Online Gradient Descent" << endl;
}

// In OGD, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update

void OGD::*OgdThread(void *in_par) {
  return NULL;
}

void OGD::Learn() {
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  // initial learning rate
  eta0_ = sqrt(TR_->n_ex_);
  t_init_ = 1.0 / (eta0_ * reg_factor_);

  for (size_t t = 0; t < n_thread_; t++) {
    // init thread parameters
    thd_par_[t].id_ = t;
    thd_par_[t].state_ = 0;
    /*
    // init thread weights and messages
    l1.w_vec_pool[t] = CreateEmptySvector();
    l1.w_n_vec_pool[t] = CreateEmptySvector(); // not used; only for EG.
    l1.bias_pool[t] = 0.0;
    l1.bias_n_pool[t] = 0.0; // not used; only for EG.
    l1.msg_pool[t] = CreateEmptySvector();
    
    l1.t_pool[t] = t_init;
    l1.scale_pool[t] = 1.0;
    l1.num_used_exp[t] = 0;
    l1.total_loss_pool[t] = 0.0;
    l1.total_misp_pool[t] = 0;
    */
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, OGD::OgdThread, (void*)&thd_par_[t]);
  }

  FinishThreads();
}

void OGD::Test() {
}
