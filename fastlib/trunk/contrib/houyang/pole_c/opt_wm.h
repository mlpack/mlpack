// Online Weighted Majority

#ifndef WM_H
#define WM_H

#include "parallel.h"
#include "sparsela.h"
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

void WmUpdate(SVEC *wvec, size_t tid) {
  SVEC *temp_v = NULL;
  if (global.comm_method == 1) {
    if (l1.opt_method == "dwm_i") {
      temp_v = CreateConstDvector(l1.num_experts, 1.0);
      for (size_t h=0; h<global.num_threads; h++) {
	SparseMultiplyOverwrite(temp_v, l1.w_vec_pool[h]);
      }
      SparsePowerOverwrite(temp_v, 1.0/global.num_threads);
    }
    else { // l1.opt_method == "dwm_a"
      temp_v = CreateConstDvector(l1.num_experts, 0.0);
      for (size_t h=0; h<global.num_threads; h++) {
	SparseAddOverwrite(temp_v, l1.w_vec_pool[h]);
      }
      SparseScaleOverwrite(temp_v, 1.0/global.num_threads);
    }
  }
  else if (global.comm_method == 0) {
    if (l1.opt_method == "dwm_i") {
      temp_v = CreateConstDvector(l1.num_experts, 1.0);
      SparseMultiplyOverwrite(temp_v, l1.w_vec_pool[tid]);
    }
    else { // l1.opt_method == "dwm_a"
      temp_v = CreateConstDvector(l1.num_experts, 0.0);
      SparseAddOverwrite(temp_v, l1.w_vec_pool[tid]);
    }
  }

  CopyFromSvec(l1.w_vec_pool[tid], temp_v);

  DestroySvec(temp_v);
  // dummy updating time
  //boost::this_thread::sleep(boost::posix_time::microseconds(1));
}

void *WmThread(void *in_par) {
  thread_param* par = (thread_param*) in_par;
  size_t tid = par->thread_id;
  EXAMPLE **exs;
  int b;
  vector<T_LBL> exp_pred (l1.num_experts, 0.0);
  T_LBL pred_lbl;
  double sum_weight_pos, sum_weight_neg;

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
      for (b = 0; b<global.mb_size; b++) {
	sum_weight_pos = 0.0; sum_weight_neg = 0.0;
	for (size_t p=0; p<l1.num_experts; p++) {
	  exp_pred[p] = l1.weak_learners[p]->WeakPredictLabel(exs[b]);
	  if (exp_pred[p] ==  1) {
	    sum_weight_pos = sum_weight_pos + l1.w_vec_pool[tid]->feats[p].wval;
	  }
	  else {
	    sum_weight_neg = sum_weight_neg + l1.w_vec_pool[tid]->feats[p].wval;
	  }
	}
	if (sum_weight_pos > sum_weight_neg) {
	  pred_lbl = (T_LBL)1;
	}
	else {
	  pred_lbl = (T_LBL)-1;
	}

	if (global.calc_loss) {
	  // Calculate number of misclassifications
	  if (l1.type == "classification") {
	    //cout << "ex_label: "<< exs[b]->label << ", pred_label: " << pred_lbl << endl;
	    if (pred_lbl != exs[b]->label) {
	      l1.total_misp_pool[tid] = l1.total_misp_pool[tid] + 1;
	    }
	  }
	}

	// local update
	for (size_t p=0; p<l1.num_experts; p++) {
	  if (exp_pred[p] != exs[b]->label) {
	    l1.w_vec_pool[tid]->feats[p].wval = l1.w_vec_pool[tid]->feats[p].wval * l1.alpha;
	  }
	}

      }

      // dummy gradient calc time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));

      // wait till all threads send their messages
      pthread_barrier_wait(&barrier_msg_all_sent);
      
      par->thread_state = 2;
      break;
    case 2:
      // update using messages
      WmUpdate(l1.w_vec_pool[tid], tid);
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


void Wm(learner &l) {
  size_t n_threads = l.num_threads;
  size_t t;

  if (l.num_experts <= 0) {
    cout << "Number of experts not specified for WM! Bailing!" << endl;
    exit(1);
  }
  // Set parameters for weak learners
  if (l.wl_name == "stump") {
    // choose splitting dimensions
    if (l.num_experts > global.max_feature_idx) {
      cout << "Number of experts: " <<l.num_experts << " larger than number of feature dimension: "<< global.max_feature_idx <<" ! Bailing!" << endl;
      exit(1);
    }
    vector<size_t> dims;
    // feature index starts from 1
    for (size_t d=1; d<=global.max_feature_idx; d++)
      dims.push_back(d);
    random_shuffle ( dims.begin(), dims.end() );

    // Train experts
    for (size_t k=0; k< l.num_experts; k++) {
      l.weak_learners[k] = GetWeakLearner( l.wl_name, dims[k], max(200, (int)ceil(num_train_exps/50)) );
      l.weak_learners[k]->WeakTrain(train_exps, num_train_exps, NULL);
    }
  }
  else {
    for (size_t k=0; k< l.num_experts; k++) {
      l.weak_learners[k] = GetWeakLearner(l.wl_name, 0, 0);
    }
  }

  threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
  t_par = (thread_param**)calloc(n_threads, sizeof(thread_param*));
  
  pthread_barrier_init(&barrier_msg_all_sent, NULL, n_threads);
  pthread_barrier_init(&barrier_msg_all_used, NULL, n_threads);

  // initial learning rate  
  for (t = 0; t < n_threads; t++) {
    // init thread parameters
    t_par[t] = (thread_param*)calloc(1, sizeof(thread_param));
    t_par[t]->thread_id = t;
    t_par[t]->l = &l;
    t_par[t]->thread_state = 0;
    // init thread weights
    l1.w_vec_pool[t] = CreateConstDvector(l.num_experts, num_train_exps);
    l1.msg_pool[t] = CreateEmptySvector();
    l1.num_used_exp[t] = 0;

    l1.total_misp_pool[t] = 0;
    // begin learning iterations
    pthread_create(&threads[t], NULL, WmThread, (void*)t_par[t]);
  }

  FinishThreads(n_threads);

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
