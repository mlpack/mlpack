
#include "opt_ogd.h"

struct thread_par {
  size_t id_;
  Learner *Lp_;
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
  Learner* Lp = (Learner *)par->Lp_;
  
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
      /*
      for (b = 0; b<Lp->mb_size; b++) {
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
      */
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
    // init thread parameters
    pars[t].id_ = t;
    pars[t].Lp_ = this;
    state_[t] = 0;
    // begin learning iterations
    pthread_create(&Threads_[t], NULL, &OGD::OgdThread, (void*)&pars[t]);
  }

  FinishThreads();
}

void OGD::Test() {
}

