
#include "opt_oeg.h"

struct thread_par {
  size_t id_;
  OEG *Lp_;
};

OEG::OEG() {
  cout << "---Online Exponentiated Gradient---" << endl;
}

void OEG::OegCommUpdate(size_t tid) {
  if (comm_method_ == 1) { // fully connected graph
    for (size_t h=0; h<n_thread_; h++) {
      if (h != tid) {
	w_p_pool_[tid].SparseMultiplyOverwrite(&m_p_pool_[h]);
	w_n_pool_[tid].SparseMultiplyOverwrite(&m_n_pool_[h]);
      }
    }
    w_p_pool_[tid].SparsePowerOverwrite(1.0/n_thread_);
    w_n_pool_[tid].SparsePowerOverwrite(1.0/n_thread_);
  }
  else { // no communication
  }

  // Normalization; calc sum_i(w_p_i+w_n_i)
  double w_sum = 0.0;
  for (size_t i=0; i<w_p_pool_[tid].Fs_.size(); i++) {
    w_sum += w_p_pool_[tid].Fs_[i].v_;
  }
  for (size_t i=0; i<w_n_pool_[tid].Fs_.size(); i++) {
    w_sum += w_n_pool_[tid].Fs_[i].v_;
  }
  w_sum = w_sum + b_p_pool_[tid] + b_n_pool_[tid];
  //cout << w_sum << endl;
  if (w_sum > reg_C_) {
    w_p_pool_[tid].SparseScaleOverwrite(reg_C_/w_sum);
    w_n_pool_[tid].SparseScaleOverwrite(reg_C_/w_sum);
    b_p_pool_[tid] = b_p_pool_[tid] * reg_C_ / w_sum;
    b_n_pool_[tid] = b_n_pool_[tid] * reg_C_ / w_sum;
  }
}

// In Distributed OEG, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
void* OEG::OegThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  size_t tid = par->id_;
  OEG* Lp = (OEG *)par->Lp_;
  Example* exs[Lp->mb_size_];
  Svector uv; // update vector
  double ub = 0.0; // for bias

  while (true) {
    switch (Lp->state_[tid]) {
    case 0: // waiting to read data
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	if ( Lp->GetImmedExample(Lp->TR_, exs+b, tid) ) { // new example read
	  //exs[b]->Print();
	}
	else { // all epoches finished
	  return NULL;
	}
      }
      Lp->state_[tid] = 1;
      break;
    case 1: // predict and local update
      double eta;
      Lp->n_it_[tid] = Lp->n_it_[tid] + 1;
      /*
      if (Lp->reg_type_ == 2) {
	eta= 1.0 / (Lp->reg_factor_ * Lp->n_it_[tid]);
      }
      else {
	eta = 1.0 / sqrt(Lp->n_it_[tid]);
      }
      */
      eta = 1.0 / Lp->n_it_[tid];
      //--- local update: subgradient of loss function
      uv.Clear(); ub = 0.0;
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	double bias = Lp->b_p_pool_[tid] - Lp->b_n_pool_[tid];
	Svector w;
	w.SparseSubtract(&Lp->w_p_pool_[tid], &Lp->w_n_pool_[tid]);
	double pred_val = Lp->LinearPredictBias(&w, exs[b], bias);
	Lp->MakeLog(tid, &w, bias, exs[b], pred_val);
	double update = Lp->LF_->GetUpdate(pred_val, (double)exs[b]->y_);
	uv.SparseAddExpertOverwrite(update, exs[b]);
        ub += update;
      }
      uv.SparseScaleOverwrite(eta / Lp->mb_size_);
      // update bias
      if (Lp->use_bias_) {
        Lp->b_p_pool_[tid] = Lp->b_p_pool_[tid] * exp(eta * ub / Lp->mb_size_);
	Lp->b_n_pool_[tid] = Lp->b_n_pool_[tid] / exp(eta * ub / Lp->mb_size_);
      }
      // update w
      Lp->w_p_pool_[tid].SparseExpMultiplyOverwrite(&uv);
      Lp->w_n_pool_[tid].SparseNegExpMultiplyOverwrite(&uv);
      //--- dummy gradient calc time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));
      // send message out
      Lp->m_p_pool_[tid].Copy(Lp->w_p_pool_[tid]);
      Lp->m_n_pool_[tid].Copy(Lp->w_n_pool_[tid]);
      //--- wait till all threads send their messages
      pthread_barrier_wait(&Lp->barrier_msg_all_sent_);
      Lp->state_[tid] = 2;
      break;
    case 2: // communicate and update using other's msg
      // update using received messages
      Lp->OegCommUpdate(tid);
      // wait till all threads used messages they received
      pthread_barrier_wait(&Lp->barrier_msg_all_used_);
      // communication done
      Lp->state_[tid] = 0;
      break;
    default:
      cout << "ERROR! Unknown thread state number !" << endl;
      return NULL;
    }
  }
  return NULL;
}
  
void OEG::Learn() {
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  // init learning rate
  eta0_ = sqrt(TR_->n_ex_);
  t_init_ = 1.0 / (eta0_ * reg_factor_);
  // init parameters
  w_p_pool_.resize(n_thread_);
  w_n_pool_.resize(n_thread_);
  m_p_pool_.resize(n_thread_);
  m_n_pool_.resize(n_thread_);
  b_p_pool_.resize(n_thread_);
  b_n_pool_.resize(n_thread_);

  thread_par pars[n_thread_];
  for (size_t t = 0; t < n_thread_; t++) {
    // init thread parameters and statistics
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    b_p_pool_[t] = 0.5*reg_C_/(TR_->max_ft_idx_+1);
    b_n_pool_[t] = 0.5*reg_C_/(TR_->max_ft_idx_+1);
    w_p_pool_[t].SetAllResize(TR_->max_ft_idx_, 0.5*reg_C_/(TR_->max_ft_idx_+1));
    w_n_pool_[t].SetAllResize(TR_->max_ft_idx_, 0.5*reg_C_/(TR_->max_ft_idx_+1));
    state_[t] = 0;
    n_it_[t] = t_init_;
    thd_n_used_examples_[t] = 0;
    loss_[t] = 0;
    err_[t] = 0;
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &OEG::OegThread, (void*)&pars[t]);
  }

  FinishThreads();
  SaveLog();
}

void OEG::Test() {
}

void OEG::MakeLog(size_t tid, Svector *w, double bias, Example *x, double pred_val) {
  if (calc_loss_) {
    // Calculate loss and number of misclassifications
    if (type_ == "classification") {
      T_LBL pred_lbl = LinearPredictBiasLabelBinary(w, x, bias);
      //cout << x->y_ << " : " << pred_lbl << endl;
      if (pred_lbl != x->y_) {
	loss_[tid] = loss_[tid] + LF_->GetLoss(pred_val, (double)x->y_);
	err_[tid] =  err_[tid] + 1;
      }
      // save intermediate logs
      if (n_log_ > 0) {
	LOG_->ct_t_[tid]  = LOG_->ct_t_[tid] + 1;
	if (LOG_->ct_t_[tid] == LOG_->t_int_ && LOG_->ct_lp_[tid] < n_log_) {
	  LOG_->err_[tid][LOG_->ct_lp_[tid]] = err_[tid];
	  LOG_->loss_[tid][LOG_->ct_lp_[tid]] = loss_[tid];
	  LOG_->ct_t_[tid] = 0;
	  LOG_->ct_lp_[tid] = LOG_->ct_lp_[tid] + 1;
	}
      }
    }
    // Calculate loss only for regression etc.
    else {
      loss_[tid] = loss_[tid] + LF_->GetLoss(pred_val, (double)x->y_);
    }
  }
}

void OEG::SaveLog() {
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
      fprintf(fp, "Log intervals: %ld. Number of logs: %ld\n\n", 
	      LOG_->t_int_, n_log_);
      fprintf(fp, "Errors cumulated:\n");
      for (size_t t=0; t<n_thread_; t++) {
	for (size_t k=0; k<n_log_; k++) {
	  fprintf(fp, "%ld", LOG_->err_[t][k]);
	  fprintf(fp, " ");
	}
	fprintf(fp, ";\n");
      }
      fprintf(fp, "\n\nLoss cumulated:\n");
      for (size_t t=0; t<n_thread_; t++) {
	for (size_t k=0; k<n_log_; k++) {
	  fprintf(fp, "%lf", LOG_->loss_[t][k]);
	  fprintf(fp, " ");
	}
	fprintf(fp, ";\n");
      }
      fclose(fp);
    }

    // final loss
    double t_l = 0.0;
    for (size_t t = 0; t < n_thread_; t++) {
      t_l += loss_[t];
      cout << "t"<< t << ": " << thd_n_used_examples_[t] 
	   << " samples processed. Loss: " << loss_[t]<< endl;
    }
    cout << "Total loss: " << t_l << endl;

    // prediction accuracy for classifications
    if (type_ == "classification") {
      size_t t_m = 0, t_s = 0;
      for (size_t t = 0; t < n_thread_; t++) {
	t_m += err_[t];
	t_s += thd_n_used_examples_[t];
	cout << "t"<< t << ": " << thd_n_used_examples_[t] << 
	  " samples processed. Misprediction: " << err_[t]<< ", accuracy: "<< 
	  1.0-(double)err_[t]/(double)thd_n_used_examples_[t] << endl;
      }
      cout << "Total mispredictions: " << t_m << ", accuracy: " << 
	1.0-(double)t_m/(double)t_s<< endl;
    }
  }
}
