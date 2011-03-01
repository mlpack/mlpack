
#include "opt_ogd.h"

struct thread_par {
  size_t id_;
  OGD *Lp_;
};

OGD::OGD() {
  cout << "--Online Gradient Descent--" << endl;
}

// In OGD, thread states are defined as:
// 0: waiting to read data
// 1: data read, predict and send message(e.g. calc subgradient)
// 2: msg sent done, waiting to receive messages from other agents and update
void* OGD::OgdThread(void *in_par) {
  thread_par* par = (thread_par*) in_par;
  size_t tid = par->id_;
  OGD* Lp = (OGD *)par->Lp_;
  
  Example* exs[Lp->mb_size_];
  while (true) {
    switch (Lp->state_[tid]) {
    case 0: // waiting to read data
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	if ( Lp->GetImmedExample(Lp->TR_, exs+b, 1) ) { // new example read
	  //exs[b]->Print();
	}
	else { // all epoches finished
	  return NULL;
	}
      }
      Lp->state_[tid] = 1;
      break;
    case 1:
      for (size_t b = 0; b<Lp->mb_size_; b++) {
	double pred_val = Lp->LinearPredictBias(&Lp->w_pool_[tid], exs[b], Lp->b_pool_[tid]);
	// ------------for log-------------------
	if (Lp->calc_loss_) {
	  if (Lp->type_ == "classification") { // Calculate loss and number of misclassifications
	    T_LBL pred_lbl = Lp->LinearPredictBiasLabelBinary(&Lp->w_pool_[tid], exs[b], Lp->b_pool_[tid]);
	    //cout << exs[b]->y_ << " : " << pred_lbl << ", "<< Lp->LF_->GetLoss(pred_val, (double)exs[b]->y_) << endl;
	    if (pred_lbl != exs[b]->y_) {
	      Lp->loss_[tid] = Lp->loss_[tid] + Lp->LF_->GetLoss(pred_val, (double)exs[b]->y_);
	      Lp->err_[tid] =  Lp->err_[tid] + 1;
	    }
	    if (Lp->reg_type_ == 2 && Lp->reg_factor_ != 0) {
	      //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
	      Lp->loss_[tid] = Lp->loss_[tid] + 0.5 * Lp->reg_factor_ * Lp->w_pool_[tid].SparseSqL2Norm();
	    }
	    if (Lp->n_log_ > 0) {
	      Lp->LOG_->ct_t_[tid]  = Lp->LOG_->ct_t_[tid] + 1;
	      if (Lp->LOG_->ct_t_[tid] == Lp->LOG_->t_int_ && Lp->LOG_->ct_lp_[tid] < Lp->n_log_) {
		Lp->LOG_->err_[tid][Lp->LOG_->ct_lp_[tid]] = Lp->err_[tid];
		Lp->LOG_->loss_[tid][Lp->LOG_->ct_lp_[tid]] = Lp->loss_[tid];
		Lp->LOG_->ct_t_[tid] = 0;
		Lp->LOG_->ct_lp_[tid] = Lp->LOG_->ct_lp_[tid] + 1;
	      }
	    }
	  }
	  
	  else { // Calculate loss only
	    Lp->loss_[tid] = Lp->loss_[tid] + Lp->LF_->GetLoss(pred_val, (double)exs[b]->y_);
	    if (Lp->reg_type_ == 2 && Lp->reg_factor_ != 0) {
	      Lp->loss_[tid] = Lp->loss_[tid] + 0.5 * Lp->reg_factor_ * Lp->w_pool_[tid].SparseSqL2Norm();
	    }
	  }
	}//----------- log end-------------

	update = l1.loss_func->getUpdate(pred,(double)exs[b]->label);
	// update message: [y_t x_t]^+_i
	SparseAddExpertOverwrite(l1.msg_pool[tid], update, exs[b]);
      }
      SparseScaleOverwrite(l1.msg_pool[tid], 1.0/global.mb_size);
      // dummy gradient calc time
      //boost::this_thread::sleep(boost::posix_time::microseconds(1));

      // wait till all threads send their messages
      pthread_barrier_wait(&barrier_msg_all_sent);
      Lp->state_[tid] = 2;
      break;
    case 2:
      /*
      // update using messages
      OgdUpdate(l1.w_vec_pool[tid], l1.t_pool[tid], l1.bias_pool[tid], update, tid);
      // wait till all threads used messages they received
      pthread_barrier_wait(&barrier_msg_all_used);
      // communication done
      */
      Lp->state_[tid] = 0;
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
  eta0_ = sqrt(TR_->n_ex_);
  t_init_ = 1.0 / (eta0_ * reg_factor_);
  // init parameters
  w_pool_.resize(n_thread_);
  b_pool_.resize(n_thread_);

  thread_par pars[n_thread_];
  for (size_t t = 0; t < n_thread_; t++) {
    // init thread parameters and statistics
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    state_[t] = 0;
    thd_n_used_examples_[t] = 0;
    loss_[t] = 0;
    err_[t] = 0;
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &OGD::OgdThread, (void*)&pars[t]);
  }

  FinishThreads();
}

void OGD::Test() {
}

