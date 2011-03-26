
#include "opt_wm.h"

struct thread_par {
  size_t id_;
  WM *Lp_;
};

WM::WM() {
  cout << "---Distributed Weighted Majority---" << endl;
}

void WM::WmCommUpdate(size_t tid) {
  if (comm_method_ == 1) {
    if (opt_name_ == "dwm_i") {
      for (size_t h=0; h<n_thread_; h++) {
        if (h != tid) {
	  w_pool_[tid] *= m_pool_[h];
        }
      }
      w_pool_[tid] ^= 1.0/n_thread_;
    }
    else if (opt_name_ == "dwm_a") {
      for (size_t h=0; h<n_thread_; h++) {
        if (h != tid) {
          w_pool_[tid] += m_pool_[h];
        }
      }
      w_pool_[tid] /= n_thread_;
    }
    else {
      cout << "Unknown WM method!" << endl;
      exit (1);
    }
  }
  else { // no communication
  }
}

// In Distributed WM, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
void* WM::WmThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  size_t tid = par->id_;
  WM* Lp = (WM *)par->Lp_;
  Example* exs[Lp->mb_size_];
  vector<T_LBL> exp_pred(Lp->n_expert_, 0.0);

  while (true) {
    switch (Lp->t_state_[tid]) {
    case 0: // waiting to read data
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	if ( Lp->GetImmedExample(Lp->TR_, exs+b, tid) ) { // new example read
	  //exs[b]->Print();
	}
	else { // all epoches finished
	  return NULL;
	}
      }
      Lp->t_state_[tid] = 1;
      break;
    case 1: // predict and local update
      double sum_weight_pos, sum_weight_neg;
      T_LBL pred_lbl;
      Lp->t_n_it_[tid] = Lp->t_n_it_[tid] + 1;
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	sum_weight_pos = 0.0; sum_weight_neg = 0.0;
	for (size_t p=0; p<Lp->n_expert_; p++) {
	  exp_pred[p] = Lp->WLs_[p]->PredictLabelBinary(exs[b]);
	  if (exp_pred[p] != exs[b]->y_) {
            Lp->t_exp_err_[p][tid] = Lp->t_exp_err_[p][tid] + 1;
	  }
	  if (exp_pred[p] == 1) {
            sum_weight_pos += Lp->w_pool_[tid].Fs_[p].v_;
	  }
	  else {
	    sum_weight_neg += Lp->w_pool_[tid].Fs_[p].v_;
	  }
	}
	if (sum_weight_pos > sum_weight_neg) {
	  pred_lbl = (T_LBL)1;
	}
	else {
	  pred_lbl = (T_LBL)-1;
	}
        // local update weights
	for (size_t p = 0; p < Lp->n_expert_; p++) {
	  if (exp_pred[p] != exs[b]->y_) {
	    Lp->w_pool_[tid].Fs_[p].v_ = Lp->w_pool_[tid].Fs_[p].v_ * Lp->alpha_;
	  }
	}
	Lp->MakeLog(tid, exs[b]->y_, pred_lbl, exp_pred);
      }
      // dummy calculation time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));
      // send message out
      Lp->m_pool_[tid] = Lp->w_pool_[tid];
      // wait till all threads send their messages
      pthread_barrier_wait(&Lp->barrier_msg_all_sent_);
      Lp->t_state_[tid] = 2;
      break;
    case 2: // communicate and update using received msg
      Lp->WmCommUpdate(tid);
      // wait till all threads used messages they received
      pthread_barrier_wait(&Lp->barrier_msg_all_used_);
      // communication done
      Lp->t_state_[tid] = 0;
      break;
    default:
      cout << "ERROR! Unknown thread state number !" << endl;
      return NULL;
    }
  }
  return NULL;
}

void WM::TrainWeak() {
  WLs_.resize(n_expert_);
  if (wl_name_ == "stump") {
    // choose splitting dimensions
    if (n_expert_ > TR_->max_ft_idx_) {
      cout << "Number of experts: " << n_expert_ << 
        " larger than number of feature dimension: "<< 
        TR_->max_ft_idx_+1 <<" !" << endl;
      exit(1);
    }
    vector<size_t> sd(TR_->max_ft_idx_, 0);
    for (size_t d = 0; d < TR_->max_ft_idx_; d++) {
      sd[d] = d;
    }
    random_shuffle(sd.begin(), sd.end());
    // choose number of iterations
    size_t n_it;
    if (TR_->Size() > 10000)
      n_it = min(200, (int)ceil(TR_->Size()/50));
    else
      n_it = max(200, (int)ceil(TR_->Size()/50));
    // train
    cout << "Training " << n_expert_ <<" experts (weak learners)...";
    for (size_t p = 0; p < n_expert_; p++) {
      WLs_[p] = new DecisionStump(sd[p], n_it);
      WLs_[p]->BatchLearn(TR_);
    }
    cout << "done." << endl;
  }
  else {
    cout << "Invalid weak learner name: " << wl_name_ << "!" << endl;
    exit(1);
  }
}

void WM::Learn() {
  // init
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  w_pool_.resize(n_thread_);
  m_pool_.resize(n_thread_);

  // train weak learners
  TrainWeak();

  thread_par pars[n_thread_];
  for (size_t t = 0; t < n_thread_; t++) {
    // init thread parameters and statistics
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    w_pool_[t].SetAllResize(n_expert_, TR_->Size()*n_epoch_+n_iter_res_);
    t_state_[t] = 0;
    t_n_it_[t] = 0;
    t_n_used_examples_[t] = 0;
    t_loss_[t] = 0;
    t_err_[t] = 0;
    for (size_t p = 0; p < n_expert_; p++) {
      t_exp_err_[p][t] = 0;
    }
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &WM::WmThread, (void*)&pars[t]);
  }

  FinishThreads();
  SaveLog();
}

void WM::Test() {
}

void WM::MakeLog(size_t tid, T_LBL true_lbl, T_LBL pred_lbl, 
                 vector<T_LBL> &exp_pred) {
  if (calc_loss_) {
    // Calc # of misclassifications
    if (type_ == "classification") {
      if (pred_lbl != true_lbl) {
        t_err_[tid] =  t_err_[tid] + 1;
      }
    }
    // intermediate logs
    if (n_log_ > 0) {
      LOG_->ct_t_[tid]  = LOG_->ct_t_[tid] + 1;
      if (LOG_->ct_t_[tid] == LOG_->t_int_ && LOG_->ct_lp_[tid] < n_log_) {
        LOG_->err_[tid][LOG_->ct_lp_[tid]] = t_err_[tid];
        for (size_t p=0; p<n_expert_; p++) {
          if (exp_pred[p] != true_lbl) {
            LOG_->err_exp_[p][LOG_->ct_lp_[tid]] = 
              LOG_->err_exp_[p][LOG_->ct_lp_[tid]] + 1; // TODO
          }
        }
        LOG_->ct_t_[tid] = 0;
        LOG_->ct_lp_[tid] = LOG_->ct_lp_[tid] + 1;
      }
    }
  } 
}

void WM::SaveLog() {
  if (calc_loss_) {
    // intermediate logs
    if (n_log_ > 0) {
      FILE *fp;
      string log_fn(TR_->fn_);
      log_fn += ".";
      log_fn += opt_name_;
      log_fn += ".log";
      if ((fp = fopen (log_fn.c_str(), "w")) == NULL) {
	cerr << "Cannot save log file!"<< endl;
	exit (1);
      }
      fprintf(fp, "Log intervals: %zu. Number of logs: %zu\n\n", 
	      LOG_->t_int_, n_log_);
      fprintf(fp, "Errors cumulated:\n");
      for (size_t t=0; t<n_thread_; t++) {
	for (size_t k=0; k<n_log_; k++) {
	  fprintf(fp, "%zu", LOG_->err_[t][k]);
	  fprintf(fp, " ");
	}
	fprintf(fp, ";\n");
      }
      // TODO: accumulate LOG->err_exp_
      for (size_t p=0; p<n_expert_; p++) {
        for (size_t k=1; k<n_log_; k++) {
          LOG_->err_exp_[p][k] = LOG_->err_exp_[p][k-1] + LOG_->err_exp_[p][k];
        }
      }
      fprintf(fp, "Expert Errors:\n");
      for (size_t p=0; p<n_expert_; p++) {
        for (size_t k=0; k<n_log_; k++) {
          fprintf(fp, "%zu", LOG_->err_exp_[p][k]);
          fprintf(fp, " ");
        }
        fprintf(fp, ";\n");
      }
      fclose(fp);
    }
    
    // final prediction accuracy
    if (type_ == "classification") {
      size_t t_m = 0, t_s = 0;
      for (size_t t = 0; t < n_thread_; t++) {
        t_m += t_err_[t];
        t_s += t_n_used_examples_[t];
        cout << "t"<< t << ": " << t_n_used_examples_[t] 
             << " samples processed. Misprediction: " << t_err_[t] << "; accuracy: " 
             << 1.0-(double)t_err_[t]/(double)t_n_used_examples_[t] << endl;
        /*
        for (p=0; p<n_expert_; p++) {
          cout << "Expert " << p << " made " << t_exp_err_[p][t] 
               << " mispredictions over agent " << t << ". Weight: " 
               << w_pool_[t]->Fs_[k].v_ << "." << endl;
        }
        */
      }
      cout << "Total mispredictions: " << t_m << ", accuracy: " 
           << 1.0-(double)t_m/(double)t_s<< endl;
    }
  }
}
