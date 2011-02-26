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
  if (n_log_ > 0) {
    size_t t_int = (size_t)floor( (n_epoch_*TR_->n_ex_ + n_iter_res_)/(n_thread_ * n_log_) );
    LOG_ = new Log(n_thread_, n_log_, t_int);
  }
  // for parallelism
  Threads_.resize(n_thread_);
  thd_par_.resize(n_thread_);

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
