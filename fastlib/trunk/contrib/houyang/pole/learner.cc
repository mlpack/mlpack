// Implementation for Learner

#include "learner.h"

Learner::Learner() {
}

Learner::~Learner() {
  if (LOG_)
    delete LOG_;
  if (TR_)
    delete TR_;
  if (VA_)
    delete VA_;
  if (TE_)
    delete TE_;
  if (LF_)
    delete LF_;
}

////////////////////////////
// Parallel Learning
////////////////////////////
void Learner::ParallelLearn() {
  // for logs and statistics
  epoch_ct_ = 0;
  if (n_log_ > 0) {
    size_t t_int = (size_t)floor( (n_epoch_*TR_->n_ex_ + n_iter_res_)/(n_thread_ * n_log_) );
    LOG_ = new Log(n_thread_, n_log_, t_int);
  }
  // for parallelism
  Threads_.resize(n_thread_);
  state_.resize(n_thread_);
  thd_n_used_examples_.resize(n_thread_);
  loss_.resize(n_thread_);
  err_.resize(n_thread_);
  pthread_mutex_init(&mutex_ex_, NULL);

  Learn();
}

///////////////
// Thread join
///////////////
void Learner::FinishThreads() {
  for (size_t i=0; i<n_thread_; i++) {
    pthread_join(Threads_[i], NULL);
  }
}

/*
///////////////////////////////////////
// Serial Learning
//
// Used in case where serial algorithm
// is different from the parallel one
///////////////////////////////////////
void Learner::SerialLearn() {
}
*/

///////////////////
// Online Learning
///////////////////
void Learner::OnlineLearn() {
  if (v_)
    cout << "Online learning" << endl;

  // Get input data
  if (read_port_) {
    TR_ = new Data(NULL, port_, false);
    TR_->ReadFromPort();
  }
  else {
    if (fn_learn_ != "") {
      TR_ = new Data(fn_learn_, 0, random_data_);
      TR_->ReadFromFile();
    }
    else {
      cout << "No input file name provided!" << endl;
      exit(1);
    }
  }
  // Learning
  ParallelLearn();
}

///////////////////////////////
// Batch Learning  and Testing
///////////////////////////////
void Learner::BatchLearn() {
  if (v_)
    cout << "Batch learning" << endl;

  // Training
  if (fn_learn_ != "") {
    TR_ = new Data(fn_learn_, 0, random_data_);
    TR_->ReadFromFile();
  }
  else {
    cout << "No training file provided!" << endl;
    exit(1);
  }
  ParallelLearn();

  // Testing
  if (fn_predict_ != "") {
    if (fn_predict_ == fn_learn_) {
      TE_ = TR_;
    }
    else {
      TE_ = new Data(fn_predict_, 0, false);
      TE_->ReadFromFile();
    }
  }
  else {
    cout << "No testing file provided!" << endl;
    exit(1);
  }
  Test();
}

///////////////////////////////
// Get an immediate example
///////////////////////////////
bool Learner::GetImmedExample(Data *D, Example** x_p, size_t tid) {
  size_t ring_idx = 0;
  pthread_mutex_lock(&mutex_ex_);
  
  if (epoch_ct_ < n_epoch_) {
    ring_idx = D->used_ct_ % D->n_ex_;
    if ( ring_idx == (D->n_ex_-1) ) { // one epoch finished
      epoch_ct_ ++;
      // To mimic the online learning senario, in each epoch, 
      // we randomly permutate the dataset, indexed by old_from_new
      if (D->random_) {
	D->RandomPermute();
      }
    }
    (*x_p) = D->GetExample(ring_idx);
    thd_n_used_examples_[tid] = thd_n_used_examples_[tid] + 1;

    pthread_mutex_unlock(&mutex_ex_);
    return true;
  }
  else if (iter_res_ct_ < n_iter_res_) {
    ring_idx = D->used_ct_ % D->n_ex_;
    (*x_p) = D->GetExample(ring_idx);
    thd_n_used_examples_[tid] = thd_n_used_examples_[tid] + 1;
    iter_res_ct_ ++;
    
    pthread_mutex_unlock(&mutex_ex_);
    return true;
  }
  else {
    pthread_mutex_unlock(&mutex_ex_);
    return false;
  }
}

double Learner::LinearPredictBias(Svector *w, Example *x, double bias) {
  return x->SparseDot(w) + bias;
}

T_LBL Learner::LinearPredictBiasLabelBinary(Svector *w, Example *x, double bias) {
  if (x->SparseDot(w) + bias > 0.0)
    return (T_LBL)1;
  else
    return (T_LBL)-1;
}
