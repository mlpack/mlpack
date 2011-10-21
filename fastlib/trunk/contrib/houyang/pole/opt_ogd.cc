//***********************************************************
//* Online/Stochastic Gradient Descent
//***********************************************************
#include "opt_ogd.h"

struct thread_par {
  T_IDX id_;
  OGD *Lp_;
};

OGD::OGD() {
  cout << "<<<< Online/Stochastic Gradient Descent >>>>" << endl;
}

void OGD::CommUpdate(T_IDX tid) {
  if (comm_method_ == 1) { // fully connected graph
    for (T_IDX h=0; h<n_thread_; h++) {
      if (h != tid) {
	w_pool_[tid] += m_pool_[h];
      }
    }
    w_pool_[tid] *= 1.0/n_thread_;
  }
  else { // no communication
  }
}

////////////////////////////////////////////////////////////////
// Thread for Batch training, or Online learning
//
// In Distributed OGD, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
////////////////////////////////////////////////////////////////
void* OGD::LearnThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  T_IDX tid = par->id_;
  OGD* Lp = (OGD *)par->Lp_;
  Example* exs[Lp->mb_size_];
  double update = 0.0;
  Svector uv; // update vector
  double ub = 0.0; // for bias

  while (true) {
    switch (Lp->t_state_[tid]) {
    case 0: // waiting to read data
      for (T_IDX b = 0; b<Lp->mb_size_; b++) {
        if ( Lp->GetTrainExample(Lp->TR_, exs+b, tid) ) { // new example read
          //exs[b]->Print();
        }
        else { // all epoches finished
          return NULL;
        }
      }
      Lp->t_state_[tid] = 1;
      break;
    case 1: // predict and local update
      double eta;
      Lp->t_n_it_[tid] = Lp->t_n_it_[tid] + 1;
            
      // Make prediction and get loss
      for (T_IDX b = 0; b<Lp->mb_size_; b++) {
        /*
        // calculate w_avg and make logs
        Lp->w_avg_pool_[tid] *= (Lp->t_n_it_[tid] - 1.0);
        Lp->w_avg_pool_[tid] += Lp->w_pool_[tid];
        Lp->w_avg_pool_[tid] *= (1.0/Lp->t_n_it_[tid]);
        */
        Lp->w_avg_pool_[tid] = Lp->w_pool_[tid];
        double pred_val = Lp->LinearPredictBias(Lp->w_avg_pool_[tid], 
                                                *exs[b], Lp->b_pool_[tid]);
        Lp->MakeLearnLog(tid, exs[b], pred_val);
        update = Lp->LF_->GetUpdate(pred_val, (double)exs[b]->y_);
      }

      //----------------- step sizes for OGD ---------------
      // Assuming strong convexity: ogd_str
      if (Lp->opt_name_ == "ogd_str") {
        eta = 1 / (Lp->strongness_*Lp->t_n_it_[tid]);
      }
      // Assuming general convexity: ogd
      else if (Lp->opt_name_ == "ogd") {
        eta = Lp->dbound_ / sqrt(Lp->t_n_it_[tid]);
      }
      else {
        cout << "ERROR! Unkown OGD method."<< endl;
        exit(1);
      }
      //eta = 1.0/(60*Lp->reg_factor_);  // constant step size

      //--- local update: subgradient of loss function
      uv.Clear(); ub = 0.0;
      for (T_IDX b = 0; b<Lp->mb_size_; b++) {
        uv.SparseAddExpertOverwrite(update, *exs[b]);
        ub += update;
      }
      //--- local update: regularization part
      if (Lp->reg_type_ == 2) {
        // [- \lambda \eta w_i^t],  L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
        Lp->w_pool_[tid] *= (1.0 - eta * Lp->reg_factor_);
        // update bias term
        if (Lp->use_bias_) {
          Lp->b_pool_[tid] = Lp->b_pool_[tid] *(1.0 - eta * Lp->reg_factor_);
        }
      }
      // update bias
      if (Lp->use_bias_) {
        Lp->b_pool_[tid] = Lp->b_pool_[tid] + eta * ub / Lp->mb_size_;
      }
      // update w
      Lp->w_pool_[tid].SparseAddExpertOverwrite(eta / Lp->mb_size_, uv);
      //--- dummy gradient calc time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));
      // send message out
      Lp->m_pool_[tid] = Lp->w_pool_[tid];
      
      //--- wait till all threads send their messages
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


void OGD::Learn() {
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  // init learning rate
  //eta0_ = sqrt(TR_->Size());
  //t_init_ = 1.0 / (eta0_ * reg_factor_);
  // init parameters
  w_pool_.resize(n_thread_);
  w_avg_pool_.resize(n_thread_);
  m_pool_.resize(n_thread_);
  b_pool_.resize(n_thread_);

  thread_par pars[n_thread_];
  for (T_IDX t = 0; t < n_thread_; t++) {
    // init thread parameters
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    b_pool_[t] = 0.0;
    w_pool_[t].Clear();
    w_avg_pool_[t].Clear();
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &OGD::LearnThread, (void*)&pars[t]);
  }
  FinishThreads();
  SaveLearnLog();
}

void* OGD::TestThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  T_IDX tid = par->id_;
  OGD* Lp = (OGD *)par->Lp_;
  Example* exs[1];

  while (true) {
    if ( Lp->GetTestExample(Lp->TE_, exs, tid) ) { // new test example read
      // testing using Thread[0]'s w & b. TODO...
      double pred_val = Lp->LinearPredictBias(Lp->w_avg_pool_[0], 
                                              *exs[0], Lp->b_pool_[0]);
      Lp->MakeTestLog(tid, exs[0], pred_val);
    }
    else { // all testing examples are used
      return NULL;
    }
  }
  return NULL;
}

// for batch prediction
void OGD::Test() {
  thread_par pars[n_thread_test_];
  for (T_IDX t = 0; t < n_thread_test_; t++) {
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    pthread_create(&ThreadsTest_[t], NULL, &OGD::TestThread, (void*)&pars[t]);
  }
  FinishThreadsTest();
  SaveTestLog();
}


void OGD::MakeLearnLog(T_IDX tid, Example *x, double pred_val) {
  if (calc_loss_) {
    //cout << t_n_it_[tid] <<" |pred: " << pred_val <<", y: " << (double)x->y_ << endl;
    // Calc loss
    t_loss_[tid] = t_loss_[tid] + LF_->GetLoss(pred_val, (double)x->y_);
    if (reg_type_ == 2 && reg_factor_ != 0) {
      //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
      t_loss_[tid] = t_loss_[tid] + 
        0.5 * reg_factor_ * w_avg_pool_[tid].SparseSqL2Norm();
    }
    // for classification only: calc # of misclassifications
    if (type_ == "classification") {
      T_LBL pred_lbl = LinearPredictBiasLabelBinary(w_avg_pool_[tid], *x, b_pool_[tid]);
      //cout << x->y_ << " : " << pred_lbl << endl;
      if (pred_lbl != x->y_) {
	t_err_[tid] =  t_err_[tid] + 1;
      }
    }
    // intermediate logs
    if (n_log_ > 0) {
      LOG_->ct_t_[tid]  = LOG_->ct_t_[tid] + 1;
      if (LOG_->ct_t_[tid] == LOG_->t_int_ && LOG_->ct_lp_[tid] < n_log_) {
        LOG_->err_[tid][LOG_->ct_lp_[tid]] = t_err_[tid];
        LOG_->loss_[tid][LOG_->ct_lp_[tid]] = t_loss_[tid];
        LOG_->ct_t_[tid] = 0;
        LOG_->ct_lp_[tid] = LOG_->ct_lp_[tid] + 1;
      }
    }
  }
}

void OGD::SaveLearnLog() {
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
      fprintf(fp, "\n\nLoss cumulated:\n");
      for (T_IDX t=0; t<n_thread_; t++) {
	for (T_IDX k=0; k<n_log_; k++) {
	  fprintf(fp, "%lf", LOG_->loss_[t][k]);
	  fprintf(fp, " ");
	}
	fprintf(fp, ";\n");
      }
      fclose(fp);
    }

    // final loss
    double t_l = 0.0;
    for (T_IDX t = 0; t < n_thread_; t++) {
      t_l += t_loss_[t];
      cout << "t"<< t << ": " << t_n_used_examples_[t] 
	   << " samples processed. Loss: " << t_loss_[t]<< endl;
    }
    cout << "Total loss: " << t_l << endl;

    // prediction accuracy for classifications
    if (type_ == "classification") {
      T_IDX t_m = 0, t_s = 0;
      for (T_IDX t = 0; t < n_thread_; t++) {
	t_m += t_err_[t];
	t_s += t_n_used_examples_[t];
	cout << "t"<< t << ": " << t_n_used_examples_[t] << 
	  " samples processed. Misprediction: " << t_err_[t]<< ", accuracy: "
             << 1.0-(double)t_err_[t]/(double)t_n_used_examples_[t] << endl;
      }
      cout << "Total mispredictions: " << t_m << ", accuracy: " << 
	1.0-(double)t_m/(double)t_s<< endl;
    }
  }
  else {
    cout << "Online prediction results are not shown." << endl;
  }
}

void OGD::MakeTestLog(T_IDX tid, Example *x, double pred_val) {
  //cout << "pred: " << pred_val <<", y: " << (double)x->y_ << endl;
  t_test_loss_[tid] = t_test_loss_[tid] + LF_->GetLoss(pred_val, (double)x->y_);
  // testing using Thread[0]'s w. TODO...
  if (reg_type_ == 2 && reg_factor_ != 0) {
    //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
    t_loss_[tid] = t_loss_[tid] + 
      0.5 * reg_factor_ * w_avg_pool_[0].SparseSqL2Norm();
  }
  // for classification only: calc # of misclassifications
  if (type_ == "classification") {
    // testing using Thread[0]'s w & b. TODO...
    T_LBL pred_lbl = LinearPredictBiasLabelBinary(w_avg_pool_[0], *x, b_pool_[0]);
    //cout << x->y_ << " : " << pred_lbl << endl;
    if (pred_lbl != x->y_) {
      t_test_err_[tid] =  t_test_err_[tid] + 1;
    }
  }
}

void OGD::SaveTestLog() {
  // final loss
  double t_l = 0.0;
  for (T_IDX t = 0; t < n_thread_test_; t++) {
    if (t_test_n_used_examples_[t] > 0) {
      t_l += t_test_loss_[t];
      cout << "t"<< t << ": " << t_test_n_used_examples_[t] 
           << " samples processed. Loss: " << t_test_loss_[t]<< endl;
    }
  }
  cout << "Total loss: " << t_l << endl;

  // prediction accuracy for classifications
  if (type_ == "classification") {
    T_IDX t_m = 0, t_s = 0;
    for (T_IDX t = 0; t < n_thread_test_; t++) {
      if (t_test_n_used_examples_[t] > 0) {
        t_m += t_test_err_[t];
        t_s += t_test_n_used_examples_[t];
        cout << "t"<< t << ": " << t_test_n_used_examples_[t] << 
          " samples processed. Misprediction: " << t_test_err_[t]<< ", accuracy: "
             << 1.0-(double)t_test_err_[t]/(double)t_test_n_used_examples_[t] << endl;
      }
    }
    cout << "Total mispredictions: " << t_m << ". Overall accuracy: " << 
      1.0-(double)t_m/(double)t_s<< endl;
  }
}
