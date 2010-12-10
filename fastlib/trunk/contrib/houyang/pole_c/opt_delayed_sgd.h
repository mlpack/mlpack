#ifndef DELAYED_SGD_H
#define DELAYED_SGD_H

#include <stdlib.h>
#include <pthread.h>
#include <iostream>
#include <float.h>
#include "parallel.h"
#include "learner.h"

T_VAL partial_inner_product(EXAMPLE *ex, size_t thread) {
  T_VAL prediction = 0.0;

  T_VAL *partial_weights = l1.weight_vec[thread];
  size_t thread_mask = global.thread_mask;
  for (size_t* i = ec->indices.begin; i != ec->indices.end; i++) 
    prediction += sd_add(weights,thread_mask,ec->subsets[*i][thread], ec->subsets[*i][thread+1]);
  
  for (vector<string>::iterator i = global.pairs.begin(); i != global.pairs.end();i++) 
    {
      if (ec->subsets[(int)(*i)[0]].index() > 0)
	{
	  v_array<feature> temp = ec->atomics[(int)(*i)[0]];
	  temp.begin = ec->subsets[(int)(*i)[0]][thread];
	  temp.end = ec->subsets[(int)(*i)[0]][thread+1];
	  for (; temp.begin != temp.end; temp.begin++)
	    prediction += one_pf_quad_predict(weights,*temp.begin,
					      ec->atomics[(int)(*i)[1]],thread_mask);
	}
    }
  
  if ( thread == 0 ) 
    prediction += weights[constant & thread_mask];
  
  return prediction;
}

T_VAL finalize_prediction(T_VAL ret, size_t num_features, float &norm) {
  if (num_features > 0)
    norm = 1. / sqrtf(num_features);
  else 
    norm = 1.;
  ret *= norm;
  if (isnan(ret))
    return 0.5;
  if ( ret > global.max_label )
    return global.max_label;
  if (ret < global.min_label)
    return global.min_label;
  return ret;
}

void final_predict(example* ec, size_t num_threads) {
  float norm;
  ec->final_pred = finalize_prediction(ec->partial_prediction, ec->num_features, norm);

  /*prediction pred = {ec->final_prediction, ec->example_counter}; 
  send_prediction(global.local_prediction, pred);
  if (global.unique_id == 0) {
    size_t len = sizeof(ld->label) + sizeof(ld->weight);
    char c[len];
    bufcache_simple_label(ld,c);
    write(global.local_prediction,c,len);
  }
  */

  ec->loss = reg.loss->getLoss(ec->final_prediction, ld->label) * ld->weight;
  vars.t += ld->weight;
  
  ec->eta_round = vars.eta/pow(vars.t,vars.power_t)
    * (ld->label - ec->final_prediction)
    * norm * ld->weight;
  
  float example_update = reg.loss->getUpdate(ec->final_prediction, ld->label) * ld->weight;
  ec->eta_round = vars.eta/pow(vars.t,vars.power_t) * example_update * norm;
}

void predict(EXAMPLE *ex, size_t thread) {
  T_VAL pred_part = partial_inner_product(ex, thread);

  pthread_mutex_lock(&ex->lock);

  ex->partial_pred += pred_part;
  if (--ex->threads_to_finish != 0) {
    while (!ex->pred_done) {
      pthread_cond_wait(&final_pred_done, &ex->lock);
    }
  }
  else { // We are the last thread using this example.
    final_predict(ex, global.num_threads());
    ex->pred_done = true;
    
    pthread_cond_broadcast(&final_pred_done);

    delay_example(ex, l1.num_threads);
    //delay_example(ex,0);
  }
  pthread_mutex_unlock(&ex->lock);
  
  //return ex->final_pred;
}

void *delayed_sgd_thread(void *in_par) {
  thread_param* par = (thread_param*) in_par;
  example *ex;
  cout << "Thread"<< par->thread_id <<", learner: "<< par->l->loss_name << endl;
  while (true) {
    if ((ex = get_delayed_example(par->thread_id)) != NULL) {// for update w
      /*
      inline_train(reg, ec, thread_num, ec->eta_round);
      finish_example(ec);
      */
      cout << "here1" << endl;
      return NULL;
    }
    else if ((ex = get_immed_example(par->thread_id)) != NULL) {// for calculate w^Tx and loss
      //cout << ex->feats[0].wval << endl;
      predict(ex, par->thread_id);
            return NULL;
    }
    else if (thread_done(par->thread_id)){
      /*
      if (global.local_prediction > 0)
	shutdown(global.local_prediction, SHUT_WR);
      return NULL;
      */
    }
    else 
      ;//busywait when we have predicted on all examples but not yet trained on all.
  }
  return NULL;
}

void train_delayed_sgd(learner &l) {
  size_t n_threads = l.num_threads;

  for (size_t c=0; c<l.num_epoches; c++) {
    threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
    t_par = (thread_param**)calloc(n_threads, sizeof(thread_param*));

    for (size_t i = 0; i < n_threads; i++) {
      t_par[i] = (thread_param*)calloc(1, sizeof(thread_param));
      t_par[i]->thread_id = i;
      t_par[i]->l = &l;
      pthread_create(&threads[i], NULL, delayed_sgd_thread, (void*)t_par[i]);
    }
  
    finish_threads(n_threads);
  }
}


#endif
