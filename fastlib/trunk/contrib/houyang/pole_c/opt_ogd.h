// Online subGradient Descent

#ifndef OGD_H
#define OGD_H

#include "parallel.h"
#include "sparsela.h"
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>


void OgdUpdate(SVEC *wvec, double &tin, double &bias, double update, size_t tid) {
  SVEC *exp_sum;
  double eta; // learning rate
  if (l1.reg == 2) {
    eta= 1.0 / (l1.reg_factor * tin);
  }
  else {
    eta = 1.0 / sqrt(tin);
  }
  
  if (l1.reg == 2) {
    // [- \lambda \eta w_i^t]
    //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
    SparseScaleOverwrite(wvec, 1.0-eta * l1.reg_factor);
  }
  
  exp_sum = CreateEmptySvector();
  
  if (global.comm_method == 1) {
    for (size_t h=0; h<global.num_threads; h++) {
      SparseAddOverwrite(exp_sum, l1.msg_pool[h]);
    }
    SparseScaleOverwrite(exp_sum, eta/global.num_threads);
  }
  else if (global.comm_method == 0) {
    SparseAddOverwrite(exp_sum, l1.msg_pool[tid]);
    SparseScaleOverwrite(exp_sum, eta);
  }

  // update bias
  if (global.use_bias) {
    if (l1.reg == 2) {
      bias = bias - eta * l1.reg_factor * bias;
    }
    bias = bias + eta * update;
  }

  SparseAddOverwrite(wvec, exp_sum);

  DestroySvec(exp_sum);
  // dummy updating time
  //boost::this_thread::sleep(boost::posix_time::microseconds(1));

  tin += 1.0;
}

void *OgdThread(void *in_par) {
  thread_param* par = (thread_param*) in_par;
  size_t tid = par->thread_id;
  EXAMPLE **exs;
  double pred;
  double update = 0.0;
  int b;

  exs = (EXAMPLE **)my_malloc( sizeof(EXAMPLE *) * global.mb_size );

  while (true) {
    switch (par->thread_state) {
    case 0: // waiting to read data
      for (b = 0; b<global.mb_size; b++) {
	if ( GetImmedExample(exs+b, tid, l1) != 0 ) { // new example read
	  //print_ex(exs[b]);
	}
	else { // all epoches finished
	  return NULL;
	}
      }
      par->thread_state = 1;
      break;
    case 1:
      EmptyFeatures(l1.msg_pool[tid]);
      for (b = 0; b<global.mb_size; b++) {
	pred = LinearPredictBias(l1.w_vec_pool[tid], exs[b], l1.bias_pool[tid]);

	// ------------for log-------------------
	if (global.calc_loss) {
	  // Calculate loss and number of misclassifications
	  if (l1.type == "classification") {
	    T_LBL pred_lbl = LinearPredictBiasLabel(l1.w_vec_pool[tid], exs[b], l1.bias_pool[tid]);
	    //cout << exs[b]->label << " : " << pred_lbl << ", "<< l1.loss_func->getLoss(pred, (double)exs[b]->label) << endl;
	    if (pred_lbl != exs[b]->label) {
	      l1.total_loss_pool[tid] = l1.total_loss_pool[tid] + l1.loss_func->getLoss(pred, (double)exs[b]->label);
	      l1.total_misp_pool[tid] = l1.total_misp_pool[tid] + 1;
	    }
	    if (l1.reg == 2 && l1.reg_factor != 0) {
	      //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
	      l1.total_loss_pool[tid] = l1.total_loss_pool[tid] + 0.5 * l1.reg_factor * SparseSqL2Norm(l1.w_vec_pool[tid]);
	    }
	    if (l1.num_log > 0) {
	      l1.t_ct[tid]  = l1.t_ct[tid] + 1;
	      if (l1.t_ct[tid] == l1.t_int && l1.lp_ct[tid] < l1.num_log) {
		l1.log_err[tid][l1.lp_ct[tid]] = l1.total_misp_pool[tid];
		l1.log_loss[tid][l1.lp_ct[tid]] = l1.total_loss_pool[tid];
		l1.t_ct[tid] = 0;
		l1.lp_ct[tid] = l1.lp_ct[tid] + 1;
	      }
	    }
	  }
	  // Calculate loss
	  else {
	    l1.total_loss_pool[tid] = l1.total_loss_pool[tid] + l1.loss_func->getLoss(pred, (double)exs[b]->label);
	    if (l1.reg == 2 && l1.reg_factor != 0) {
	      l1.total_loss_pool[tid] = l1.total_loss_pool[tid] + 0.5 * l1.reg_factor * SparseSqL2Norm(l1.w_vec_pool[tid]);
	    }
	  }
	}
	//----------- log end-------------

	update = l1.loss_func->getUpdate(pred,(double)exs[b]->label);
	// update message: [y_t x_t]^+_i
	SparseAddExpertOverwrite(l1.msg_pool[tid], update, exs[b]);

      }
      SparseScaleOverwrite(l1.msg_pool[tid], 1.0/global.mb_size);
      // dummy gradient calc time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));

      // wait till all threads send their messages
      pthread_barrier_wait(&barrier_msg_all_sent);
      
      par->thread_state = 2;
      break;
    case 2:
      // update using messages
      OgdUpdate(l1.w_vec_pool[tid], l1.t_pool[tid], l1.bias_pool[tid], update, tid);
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
  size_t t, k;

  //// for log
  if (l.num_log > 0) {
    l.t_int = (size_t)floor( (global.num_epoches*num_train_exps+global.num_iter_res)/(n_threads * l.num_log) );
    l.t_ct = (size_t*)malloc(n_threads * sizeof(size_t));
    l.lp_ct = (size_t*)malloc(n_threads * sizeof(size_t));
    l.log_err = (size_t**)malloc(n_threads * sizeof(size_t*));
    l.log_loss = (double**)malloc(n_threads * sizeof(double*));
    for (t=0; t<n_threads; t++) {
      l.log_err[t] = (size_t*)malloc(l.num_log * sizeof(size_t));
      l.log_loss[t] = (double*)malloc(l.num_log * sizeof(double));
      l.t_ct[t] = 0;
      l.lp_ct[t] = 0;
    }
    for (t=0; t<n_threads; t++)
      for (k=0; k<l.num_log; k++) {
	l.log_err[t][k] = 0;
	l.log_loss[t][k] = 0.0;
      }
  }
  //// log end

  threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
  t_par = (thread_param**)calloc(n_threads, sizeof(thread_param*));
  
  pthread_barrier_init(&barrier_msg_all_sent, NULL, n_threads);
  pthread_barrier_init(&barrier_msg_all_used, NULL, n_threads);

  // initial learning rate
  double eta0 = sqrt(num_train_exps);
  double t_init = 1.0 / (eta0 * l1.reg_factor);
  
  for (t = 0; t < n_threads; t++) {
    // init thread parameters
    t_par[t] = (thread_param*)calloc(1, sizeof(thread_param));
    t_par[t]->thread_id = t;
    t_par[t]->l = &l;
    t_par[t]->thread_state = 0;
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
    // begin learning iterations
    pthread_create(&threads[t], NULL, OgdThread, (void*)t_par[t]);
  }

  FinishThreads(n_threads);

  //// save log
  if (l.num_log > 0) {
    FILE *fp;
    string log_fn (global.train_data_fn);
    log_fn += ".";
    log_fn += l.opt_method;
    log_fn += ".log";
    if ((fp = fopen (log_fn.c_str(), "w")) == NULL) {
      cerr << "Cannot save log file!"<< endl;
      exit (1);
    }
    fprintf(fp, "Log intervals: %ld. Number of logs: %ld\n\n", l.t_int, l.num_log);
    fprintf(fp, "Errors cumulated:\n");
    for (t=0; t<n_threads; t++) {
      for (k=0; k<l.num_log; k++) {
	fprintf(fp, "%ld", l.log_err[t][k]);
	fprintf(fp, " ");
      }
      fprintf(fp, ";\n");
    }
    fprintf(fp, "\n\nLoss cumulated:\n");
    for (t=0; t<n_threads; t++) {
      for (k=0; k<l.num_log; k++) {
	fprintf(fp, "%lf", l.log_loss[t][k]);
	fprintf(fp, " ");
      }
      fprintf(fp, ";\n");
    }
    fclose(fp);
  }

  double t_l = 0.0;
  for (t = 0; t < n_threads; t++) {
    t_l += l1.total_loss_pool[t];
    cout << "t"<< t << ": " << l1.num_used_exp[t] << " samples processed. Loss: " << l1.total_loss_pool[t]<< endl;
  }
  cout << "Total loss: " << t_l << endl;

  // prediction accuracy for classifications
  if (l1.type == "classification") {
    size_t t_m = 0, t_s = 0;
    for (t = 0; t < n_threads; t++) {
      t_m += l1.total_misp_pool[t];
      t_s += l1.num_used_exp[t];
      cout << "t"<< t << ": " << l1.num_used_exp[t] << " samples processed. Misprediction: " << l1.total_misp_pool[t]<< ", accuracy: "<< 1.0-(double)l1.total_misp_pool[t]/(double)l1.num_used_exp[t] << endl;
    }
    cout << "------ Total mispredictions: " << t_m << ", accuracy: " << 1.0-(double)t_m/(double)t_s<< endl;
  }

  FinishLearner(l1, n_threads);
  FinishData();
}


#endif
