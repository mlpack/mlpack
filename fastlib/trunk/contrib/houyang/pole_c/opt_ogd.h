// Online subGradient Descent

#ifndef OGD_H
#define OGD_H

#include "parallel.h"
#include "sparsela.h"
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>


void OgdUpdate(SVEC *wvec, double &t, size_t tid) {
  SVEC *exp_sum;
  t = t + 1.0;
  double eta; // learning rate
  if (l1.reg == 2) {
    eta= 1.0 / (2* l1.reg_factor * t);
  }
  else {
    eta = 1.0 / sqrt(t);
  }
  exp_sum = CreateEmptySvector(NULL);
  /*
  for (size_t t=0; t<global.num_threads; t++) {
    SparseAddOverwrite(exp_sum, l1.msg_pool[t]);
  }
  */
  SparseAddOverwrite(exp_sum, l1.msg_pool[tid]);
  SparseScaleOverwrite(exp_sum, eta/global.num_threads);
  SparseAddOverwrite(wvec, exp_sum);
  // dummy updating time
  boost::this_thread::sleep(boost::posix_time::microseconds(1));
}

void *OgdThread(void *in_par) {
  thread_param* par = (thread_param*) in_par;
  size_t tid = par->thread_id;
  EXAMPLE *ex;
  T_VAL pred;
  double update;

  //ptime ts, te;
  //time_duration d;
  
  while (true) {
    switch (par->thread_state) {
    case 0: // waiting to read data
      if ( GetImmedExample(&ex, tid, l1) != 0 ) { // new example read
	//cout << "state 0, tid: " << tid << ", feat0: " << ex->feats[0].wval <<endl;
	par->thread_state = 1;
      }
      else { // all epoches finished
	return NULL;
      }
      break;
    case 1:
      pred = LinearPredict(l1.w_vec_pool[tid], ex);
      cout << ex->label << " : " << pred << ", "<< l1.loss_func->getLoss((double)pred, (double)ex->label) << endl;
      // want to calculate total loss ?
      if (global.calc_loss) {
	l1.total_loss_pool[tid] = l1.total_loss_pool[tid] + l1.loss_func->getLoss((double)pred, (double)ex->label);
	if (l1.reg == 2) {
	  l1.total_loss_pool[tid] = l1.total_loss_pool[tid] + l1.reg_factor * SparseSqL2Norm(l1.w_vec_pool[tid]);
	}
      }
      update = l1.loss_func->getUpdate(pred,(double)ex->label);
      // update message: [y_t x_t]^+_i
      SparseScale(l1.msg_pool[tid], update, ex);
      if (l1.reg == 2) {
	// [-2\lambda w_i^t]
	SparseAddExpertOverwrite(l1.msg_pool[tid], -2*l1.reg_factor, l1.w_vec_pool[tid]);
      }
      //cout << "state 1, tid: " << tid << endl;
      // wait till all threads send their messages
      pthread_barrier_wait(&barrier_msg_all_sent);
      
      par->thread_state = 2;
      break;
    case 2:
      // update using messages
      OgdUpdate(l1.w_vec_pool[tid], l1.t_pool[tid], tid);
      //cout << "state 2, tid: " << tid << endl;
      // wait till all threads used messages they received
      pthread_barrier_wait(&barrier_msg_all_used);
      // communication done
      par->thread_state = 0;
      break;
    default:
      cout << "ERROR! Unknown thread state number !" << endl;
      return NULL;
    }
  }

}

void Ogd(learner &l) {
  size_t n_threads = l.num_threads;
  size_t t;

  threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
  t_par = (thread_param**)calloc(n_threads, sizeof(thread_param*));
  
  pthread_barrier_init(&barrier_msg_all_sent, NULL, n_threads);
  pthread_barrier_init(&barrier_msg_all_used, NULL, n_threads);

  for (t = 0; t < n_threads; t++) {
    // init thread parameters
    t_par[t] = (thread_param*)calloc(1, sizeof(thread_param));
    t_par[t]->thread_id = t;
    t_par[t]->l = &l;
    t_par[t]->thread_state = 0;
    // init thread weights and messages
    l1.w_vec_pool[t] = CreateEmptySvector(NULL);
    l1.msg_pool[t] = CreateEmptySvector(NULL);
    l1.t_pool[t] = 0.001;
    l1.scale_pool[t] = 1.0;
    l1.num_used_exp[t] = 0;
    l1.total_loss_pool[t] = 0.0;
    // begin learning iterations
    pthread_create(&threads[t], NULL, OgdThread, (void*)t_par[t]);
  }

  FinishThreads(n_threads);

  double t_l = 0.0;
  for (t = 0; t < n_threads; t++) {
    t_l += l1.total_loss_pool[t];
    cout << "t"<< t << ": " << l1.num_used_exp[t] << " samples processed. Loss: " << l1.total_loss_pool[t]<< endl;
  }
  cout << "Total loss: " << t_l << endl;
  

  FinishLearner(l1, n_threads);
  FinishData();
}


#endif
