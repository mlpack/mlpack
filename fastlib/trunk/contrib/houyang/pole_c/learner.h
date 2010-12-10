#ifndef LEARNER_H
#define LEARNER_H

#include "loss_functions.h"
#include "sparsela.h"

struct learner {
  SVEC** w_vec_pool; // a pool that contains weight vectors for each thread
  SVEC** msg_pool; // a pool of messages. each thread put its message in and read other's from
  double* t_pool; // time t for SGD
  double* scale_pool; // scales for SGD
  size_t* num_used_exp; // count the number of examples used by a thread
  loss_function *loss_func;
  int reg; // Which regularization term to use; 1:L1, 2:squared L2(default), -1: no regularization
  double reg_factor; // regularization weight ('lambda' in avg_loss + lambda * regularization)
  size_t num_threads;
  size_t num_epoches;
  string loss_name;
  double* total_loss_pool; // a pool of total loss for each thread;
};

T_LBL LinearPredict(SVEC *wvec, EXAMPLE *ex) {
  double sum = 0.0;
  sum = SparseDot(wvec, ex);

  if (sum > 0.0)
    return (T_LBL)1;
  else
    return (T_LBL)-1;
}

void FinishLearner(learner &l, size_t ts) {
  if(l.w_vec_pool) {
    for (size_t t=0; t<ts; t++) {
      print_svec(l.w_vec_pool[t]);
      //FreeSvec(l.w_vec_pool[t]);
    }
    free(l.w_vec_pool);
  }
  if(l.msg_pool) {
    for (size_t t=0; t<ts; t++) {
      //FreeSvec(l.w_vec_pool[t]);
    }
    free(l.msg_pool);
  }
  free(l.t_pool);
  free(l.scale_pool);
  free(l.num_used_exp);
  free(l.total_loss_pool);
}

void TickWait(size_t ticks) {
  clock_t t = ticks + clock();
  while (t > clock());
}

#endif
