//***********************************************************
//* Randomized Weighted Majority
//***********************************************************

#include "opt_rwm.h"

struct thread_par {
  T_IDX id_;
  RWM *Lp_;
};

RWM::RWM() {
  cout << "<<<< Randomized Weighted Majority >>>>" << endl;
}

void RWM::CommUpdate(T_IDX tid) {
  accu_msg_[tid] = m_pool_[tid];
  if (comm_method_ == 1) {
    if (opt_name_ == "drwm_i") {
      for (T_IDX h=0; h<n_thread_; h++) { // set c-regular graph here
        if (h != tid) {
	  accu_msg_[tid] += m_pool_[h];
        }
      }
      for (T_IDX p = 0; p < n_expert_; p++) {
	for (T_IDX h=0; h<n_thread_; h++) {
	  if (h != tid) {
	    w_pool_[tid].Fs_[p].v_ = w_pool_[tid].Fs_[p].v_ * 
	      pow(alpha_, (double)accu_msg_[tid].Fs_[p].v_);
	  }
	}
      }
    }
    else if (opt_name_ == "drwm_r") {
      
    }
    else {
      cout << "Unknown RWM method!" << endl;
      exit (1);
    }
  }
  else if (comm_method_ == 2) {
    // using adjacency matrix
    if (opt_name_ == "drwm_i") {
      for (T_IDX h=0; h<n_thread_; h++) { // set c-regular graph here
        if (adj_m_(h,tid) == 1) {
	  accu_msg_[tid] += m_pool_[h];
        }
      }
      for (T_IDX p = 0; p < n_expert_; p++) {
	for (T_IDX h=0; h<n_thread_; h++) {
	  if (adj_m_(h,tid) == 1) {
	    w_pool_[tid].Fs_[p].v_ = w_pool_[tid].Fs_[p].v_ * 
	      pow(alpha_, (double)accu_msg_[tid].Fs_[p].v_);
	  }
	}
      }
    }
    else if (opt_name_ == "drwm_r") {
      
    }
    else {
      cout << "Unknown RWM method!" << endl;
      exit (1);
    }
  }
  else { // no communication
  }
}

// In Distributed RWM, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
void* RWM::LearnThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  T_IDX tid = par->id_;
  RWM* Lp = (RWM *)par->Lp_;
  vector<Example*> exs(Lp->mb_size_, NULL);
  vector<T_LBL> exp_pred(Lp->n_expert_, 0.0);
  RandomNumber rnd;

  while (true) {
    switch (Lp->t_state_[tid]) {
    case 0: // waiting to read data
      for (T_IDX b = 0; b<Lp->mb_size_; b++) {
	if ( Lp->GetTrainExample(Lp->TR_, &exs[b], tid) ) { // new example read
	  //exs[b]->Print();
	}
	else { // all epoches finished
	  return NULL;
	}
      }
      Lp->t_state_[tid] = 1;
      break;
    case 1: // predict and local update
      T_LBL pred_lbl;
      Lp->t_n_it_[tid] = Lp->t_n_it_[tid] + 1;
      for (T_IDX b = 0; b<Lp->mb_size_; b++) {
	// experts predict
	for (T_IDX p=0; p<Lp->n_expert_; p++) {
	  exp_pred[p] = Lp->WLs_[p]->PredictLabelBinary(exs[b]);
	}
	// choose an expert with probability w_p/(sum_p w_p)
	double r = rnd.RandomUniform(); // a double uniformly distributed in the range [0,1)
	double w_sum = 0.0;
	T_IDX chosen_expert = 0;
	for (T_IDX p = 0; p < Lp->n_expert_; p++) {
	  w_sum += Lp->w_pool_[tid].Fs_[p].v_;
	}
	r = r * w_sum;
	double sp = 0.0;
	for (T_IDX p = 0; p < Lp->n_expert_; p++) {
	  sp += Lp->w_pool_[tid].Fs_[p].v_;
	  if (sp >= r) {
	    chosen_expert = p;
	    break;
	  }
	}
	
	// agent predict following this expert
	pred_lbl = exp_pred[chosen_expert];

        // local update weights
	for (T_IDX p = 0; p < Lp->n_expert_; p++) {
	  if (exp_pred[p] != exs[b]->y_) {
	    Lp->w_pool_[tid].Fs_[p].v_ = Lp->w_pool_[tid].Fs_[p].v_ * Lp->alpha_;
	  }
	}

	Lp->MakeLearnLog(tid, exs[b]->y_, pred_lbl, exp_pred);
	
	// send message out	
	Lp->m_pool_[tid].SetAll(0.0);
	for (T_IDX p = 0; p < Lp->n_expert_; p++) {
	  if (exp_pred[p] != exs[b]->y_) {
	    Lp->m_pool_[tid].Fs_[p].v_ = Lp->m_pool_[tid].Fs_[p].v_ + 1.0;
	  }
	}
      }
      // dummy calculation time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));

      // wait till all threads send their messages
      pthread_barrier_wait(&Lp->barrier_msg_all_sent_);
      Lp->t_state_[tid] = 2;
      break;
    case 2: // communicate and update using received msg
      Lp->CommUpdate(tid);
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

void RWM::TrainWeak() {
  WLs_.resize(n_expert_);
  if (wl_name_ == "stump") {
    // choose splitting dimensions
    if (n_expert_ > TR_->max_ft_idx_) {
      cout << "Number of experts: " << n_expert_ << 
        " larger than number of feature dimension: "<< 
        TR_->max_ft_idx_+1 <<" !" << endl;
      exit(1);
    }
    vector<T_IDX> sd(TR_->max_ft_idx_, 0);
    for (T_IDX d = 0; d < TR_->max_ft_idx_; d++) {
      sd[d] = d;
    }
    random_shuffle(sd.begin(), sd.end());
    // choose number of iterations
    T_IDX n_it;
    if (TR_->Size() > 10000)
      n_it = min(200, (int)ceil(TR_->Size()/50));
    else
      n_it = max(200, (int)ceil(TR_->Size()/50));
    // train
    cout << "Training " << n_expert_ <<" experts (weak learners)...";
    for (T_IDX p = 0; p < n_expert_; p++) {
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

void RWM::Learn() {
  // init
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  w_pool_.resize(n_thread_);
  m_pool_.resize(n_thread_);
  old_m_pool_.resize(n_thread_);
  accu_msg_.resize(n_thread_);

  // form adjacency matrix
  adj_m_.zeros(n_thread_,n_thread_);

  // a k-connected cycle
  
  T_IDX hdegree = 45; // half of the degree of a graph
  for (int t = 0; t <(int)n_thread_; t++) {
    for (int d = 1; d <= (int)hdegree; d++) {
      if ( t+d < (int)n_thread_ ) {
	adj_m_(t+d,t) = 1;
      }
      else {
	adj_m_(t+d-n_thread_,t) = 1;
      }
      if ( t-d >= 0 ) {
	adj_m_(t-d,t) = 1;
      }
      else {
	adj_m_(t-d+n_thread_,t) = 1;
      }
    }
  }
  
  /*
  // binary tree
  T_IDX level = 7; // n_thread = 128
  for (unsigned int lv=0; lv<level-1; lv++) {
    for (unsigned int t=0; t<pow(2,lv); t++) {
      unsigned int idx = pow(2,lv-1)+t+1;
      adj_m_(idx+pow(2,lv), idx) = 1;
      adj_m_(idx+pow(2,lv)+1, idx) = 1;
    }
  }
  */

  // train weak learners
  TrainWeak();
  cout << "here" << endl;

  thread_par pars[n_thread_];
  for (T_IDX t = 0; t < n_thread_; t++) {
    // init thread parameters
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    w_pool_[t].SetAllResize(n_expert_, TR_->Size()*n_epoch_+n_iter_res_);
    m_pool_[t].SetAllResize(n_expert_, 0.0);
    for (T_IDX p = 0; p < n_expert_; p++) {
      t_exp_err_[p][t] = 0;
    }
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &RWM::LearnThread, (void*)&pars[t]);
  }

  FinishThreads();
  SaveLearnLog();
}

void RWM::Test() {
}

void RWM::MakeLearnLog(T_IDX tid, T_LBL true_lbl, T_LBL pred_lbl, 
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
        for (T_IDX p=0; p<n_expert_; p++) {
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

void RWM::SaveLearnLog() {
  cout << "-----------------Online Prediction------------------" << endl;
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
      for (T_IDX t=0; t<n_thread_; t++) {
	for (T_IDX k=0; k<n_log_; k++) {
	  fprintf(fp, "%zu", LOG_->err_[t][k]);
	  fprintf(fp, " ");
	}
	fprintf(fp, ";\n");
      }
      // TODO: accumulate LOG->err_exp_
      for (T_IDX p=0; p<n_expert_; p++) {
        for (T_IDX k=1; k<n_log_; k++) {
          LOG_->err_exp_[p][k] = LOG_->err_exp_[p][k-1] + LOG_->err_exp_[p][k];
        }
      }
      fprintf(fp, "Expert Errors:\n");
      for (T_IDX p=0; p<n_expert_; p++) {
        for (T_IDX k=0; k<n_log_; k++) {
          fprintf(fp, "%zu", LOG_->err_exp_[p][k]);
          fprintf(fp, " ");
        }
        fprintf(fp, ";\n");
      }
      fclose(fp);
    }
    
    // final prediction accuracy
    if (type_ == "classification") {
      T_IDX t_m = 0, t_s = 0;
      for (T_IDX t = 0; t < n_thread_; t++) {
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
  else {
    cout << "Online prediction results are not shown." << endl;
  }
}
