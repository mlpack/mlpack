//***********************************************************
//* Online Kernel Gradient Descent with Transformed Features
//* Hua Ouyang 03/23/2011
//***********************************************************
#ifndef OPT_OGD_T_H
#define OPT_OGD_T_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>

#include "learner.h"

template <typename TTransform>
class OGDT : public Learner {
 public:
  struct thread_par {
    size_t id_;
    OGDT<TTransform> *Lp_;
  };
  TTransform T_;
  vector<Svector> w_pool_; // shared memory for weight vectors of each thread
  vector<Svector> m_pool_; // shared memory for messages
  vector<double>  b_pool_; // shared memory for bias term
 private:
  double eta0_, t_init_;
  pthread_barrier_t barrier_msg_all_sent_;
  pthread_barrier_t barrier_msg_all_used_;
 public:
  OGDT();
  void Learn();
  void Test();
 private:
  static void* OgdTThread(void *par);
  void OgdTCommUpdate(size_t tid);
  void MakeLog(size_t tid, Example *x, double pred_val);
  void SaveLog();
};


template <typename TTransform>
OGDT<TTransform>::OGDT() {
  cout << "---Online Kernel Gradient Descent using Transformed Features---" << endl;
}

template <typename TTransform>
void OGDT<TTransform>::OgdTCommUpdate(size_t tid) {
  if (comm_method_ == 1) { // fully connected graph
    for (size_t h=0; h<n_thread_; h++) {
      if (h != tid) {
	w_pool_[tid].SparseAddOverwrite(&m_pool_[h]);
      }
    }
    w_pool_[tid].SparseScaleOverwrite(1.0/n_thread_);
  }
  else { // no communication
  }
}

// In Distributed OGDT, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
template <typename TTransform>
void* OGDT<TTransform>::OgdTThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  size_t tid = par->id_;
  OGDT* Lp = (OGDT *)par->Lp_;
  Example* exs[Lp->mb_size_];
  Svector uv; // update vector
  double ub = 0.0; // for bias

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
      //--- local update: regularization part
      double eta;
      Lp->t_n_it_[tid] = Lp->t_n_it_[tid] + 1;
      if (Lp->reg_type_ == 2) {
	eta= 1.0 / (Lp->reg_factor_ * Lp->t_n_it_[tid]);
      }
      else {
	eta = 1.0 / sqrt(Lp->t_n_it_[tid]);
      }
      if (Lp->reg_type_ == 2) {
	// [- \lambda \eta w_i^t],  L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
	Lp->w_pool_[tid].SparseScaleOverwrite(1.0 - eta * Lp->reg_factor_);
	// update bias term
	if (Lp->use_bias_) {
	  Lp->b_pool_[tid] = Lp->b_pool_[tid] *(1.0 - eta * Lp->reg_factor_);
	}
      }
      //--- local update: subgradient of loss function
      uv.Clear(); ub = 0.0;
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	double pred_val = Lp->LinearPredictBias(&Lp->w_pool_[tid], 
						exs[b], Lp->b_pool_[tid]);
	Lp->MakeLog(tid, exs[b], pred_val);
	double update = Lp->LF_->GetUpdate(pred_val, (double)exs[b]->y_);
	uv.SparseAddExpertOverwrite(update, exs[b]);
        ub += update;
      }
      // update bias
      if (Lp->use_bias_) {
        Lp->b_pool_[tid] = Lp->b_pool_[tid] + eta * ub / Lp->mb_size_;
      }
      // update w
      Lp->w_pool_[tid].SparseAddExpertOverwrite(eta / Lp->mb_size_, &uv);
      //--- dummy gradient calc time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));
      // send message out
      Lp->m_pool_[tid].Copy(Lp->w_pool_[tid]);
      //--- wait till all threads send their messages
      pthread_barrier_wait(&Lp->barrier_msg_all_sent_);
      Lp->t_state_[tid] = 2;
      break;
    case 2: // communicate and update using received msg
      Lp->OgdTCommUpdate(tid);
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

template <typename TTransform>
void OGDT<TTransform>::Learn() {
  pthread_barrier_init(&barrier_msg_all_sent_, NULL, n_thread_);
  pthread_barrier_init(&barrier_msg_all_used_, NULL, n_thread_);
  // init transform
  T_.D_ = trdim_;
  T_.d_ = TR_->max_ft_idx_;
  T_.SampleW();
  // init learning rate
  eta0_ = sqrt(TR_->Size());
  t_init_ = 1.0 / (eta0_ * reg_factor_);
  // init parameters
  w_pool_.resize(n_thread_);
  m_pool_.resize(n_thread_);
  b_pool_.resize(n_thread_);

  thread_par pars[n_thread_];
  for (size_t t = 0; t < n_thread_; t++) {
    // init thread parameters and statistics
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    b_pool_[t] = 0.0;
    w_pool_[t].Clear();
    t_state_[t] = 0;
    t_n_it_[t] = 0;
    t_n_used_examples_[t] = 0;
    t_loss_[t] = 0;
    t_err_[t] = 0;
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &OGDT::OgdTThread, (void*)&pars[t]);
  }

  FinishThreads();
  SaveLog();
}

template <typename TTransform>
void OGDT<TTransform>::Test() {
}

template <typename TTransform>
void OGDT<TTransform>::MakeLog(size_t tid, Example *x, double pred_val) {
  if (calc_loss_) {
    // Calc loss
    t_loss_[tid] = t_loss_[tid] + LF_->GetLoss(pred_val, (double)x->y_);
    if (reg_type_ == 2 && reg_factor_ != 0) {
      //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
      t_loss_[tid] = t_loss_[tid] + 
        0.5 * reg_factor_ * w_pool_[tid].SparseSqL2Norm();
    }
    // Calc # of misclassifications
    if (type_ == "classification") {
      T_LBL pred_lbl = LinearPredictBiasLabelBinary(&w_pool_[tid], x, b_pool_[tid]);
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

template <typename TTransform>
void OGDT<TTransform>::SaveLog() {
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
      t_l += t_loss_[t];
      cout << "t"<< t << ": " << t_n_used_examples_[t] 
	   << " samples processed. Loss: " << t_loss_[t]<< endl;
    }
    cout << "Total loss: " << t_l << endl;

    // prediction accuracy for classifications
    if (type_ == "classification") {
      size_t t_m = 0, t_s = 0;
      for (size_t t = 0; t < n_thread_; t++) {
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
}


#endif
