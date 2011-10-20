// Common functions for various learners

#include "learner.h"

///////////////
// Construction
///////////////
Learner::Learner() {
  TR_ = NULL;
  VA_ = NULL;
  TE_ = NULL;
  LF_ = NULL;
  LOG_ = NULL;
}

///////////////
// Destruction
///////////////
Learner::~Learner() {
  if (TR_ && (TE_ == NULL)) { // only training data
    delete TR_;
  }
  else if (TR_ && TE_) {
    if (TR_ == TE_) { // training == testing
      delete TR_;
    }
    else { // both training & testing data loaded
      delete TR_;
      delete TE_;
    }
  }
  else if (TE_ && (TR_ == NULL)) { // only testing data
    delete TE_;
  }
  else {
    // no training nor testing data loaded
  }
  if (VA_)
    delete VA_;
  if (LF_)
    delete LF_;
  if (LOG_)
    delete LOG_;
}

////////////////////////////
// Parallel Learning
////////////////////////////
void Learner::ParallelLearn() {
  // determine number of epochs and residual iterations
  if (n_iter_res_ >= TR_->Size()) {
    n_epoch_ += (T_IDX)(n_iter_res_ / TR_->Size());
    n_iter_res_ %= TR_->Size();
  }
  T_IDX left_ct = (n_epoch_ * TR_->Size() + n_iter_res_) % (n_thread_ * mb_size_);
  if ( left_ct > 0)
    n_iter_res_ += (n_thread_ * mb_size_ - left_ct);
  cout << "n_epo= " << n_epoch_ << ", n_iter_res= " << n_iter_res_ << endl;
  // init for intermediate logs
  epoch_ct_ = 0;
  if (n_log_ > 0) {
    T_IDX t_int = (T_IDX)floor( (n_epoch_*TR_->Size() + n_iter_res_)/(n_thread_ * n_log_) );
    LOG_ = new Log(n_thread_, n_log_, t_int, n_expert_, opt_name_);
  }
  // init for parallelism
  Threads_.resize(n_thread_);
  t_state_.resize(n_thread_);
  t_n_it_.resize(n_thread_);
  t_n_used_examples_.resize(n_thread_);
  t_loss_.resize(n_thread_);
  t_err_.resize(n_thread_);
  pthread_mutex_init(&mutex_ex_, NULL);
  // for expert-advice learning methods
  if (opt_name_ == "dwm_i" || opt_name_ == "dwm_a" || opt_name_ == "drwm_i" || opt_name_ == "drwm_r") {
    t_exp_err_.resize(n_expert_);
    for (T_IDX p=0; p<n_expert_; p++) {
      t_exp_err_[p].resize(n_thread_);
    }
  }
  // begin learning
  Learn();
}


//////////////////////////
// Parallel Testing
//////////////////////////
void Learner::ParallelTest() {
  TE_->used_ct_ = 0; // in case that TR_==TE_
  // init for parallelism
  ThreadsTest_.resize(n_thread_test_);
  t_test_n_used_examples_.resize(n_thread_test_);
  t_test_loss_.resize(n_thread_test_);
  t_test_err_.resize(n_thread_test_);
  pthread_mutex_init(&mutex_ex_test_, NULL);
  
  // begin testing
  Test();
}

////////////////////////
// Training Thread join
////////////////////////
void Learner::FinishThreads() {
  for (T_IDX i=0; i<n_thread_; i++) {
    pthread_join(Threads_[i], NULL);
  }
}

////////////////////////
// Testing Thread join
////////////////////////
void Learner::FinishThreadsTest() {
  for (T_IDX i=0; i<n_thread_test_; i++) {
    pthread_join(ThreadsTest_[i], NULL);
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
      //cout << "Training file is the same as testing file." << endl;
    }
    else {
      TE_ = new Data(fn_predict_, 0, false); // no need to permute testing data
      TE_->ReadFromFile();
    }
  }
  else {
    cout << "No testing file provided!" << endl;
    exit(1);
  }
  ParallelTest();
}

///////////////////////////////
// Get an immediate example
///////////////////////////////
bool Learner::GetImmedExample(Data *D, Example** x_p, T_IDX tid) {
  T_IDX ring_idx = 0;
  pthread_mutex_lock(&mutex_ex_);
  
  if (epoch_ct_ < n_epoch_) {
    ring_idx = D->used_ct_ % D->Size();
    if ( ring_idx == (D->Size()-1) ) { // one epoch finished
      epoch_ct_ ++;
      // To mimic the online learning senario, in each epoch, 
      // we randomly permutate the dataset, indexed by old_from_new
      if (D->random_) {
	D->RandomPermute();
      }
    }
    (*x_p) = D->GetExample(ring_idx);
    t_n_used_examples_[tid] = t_n_used_examples_[tid] + 1;

    pthread_mutex_unlock(&mutex_ex_);
    return true;
  }
  else if (iter_res_ct_ < n_iter_res_) {
    ring_idx = D->used_ct_ % D->Size();
    (*x_p) = D->GetExample(ring_idx);
    t_n_used_examples_[tid] = t_n_used_examples_[tid] + 1;
    iter_res_ct_ ++;
    
    pthread_mutex_unlock(&mutex_ex_);
    return true;
  }
  else {
    pthread_mutex_unlock(&mutex_ex_);
    return false;
  }
}

///////////////////////////////
// Get a testing example
///////////////////////////////
bool Learner::GetTestExample(Data *D, Example** x_p, T_IDX tid) {
  pthread_mutex_lock(&mutex_ex_test_);
  if (D->used_ct_ < D->Size()) {
    (*x_p) = D->GetExample(D->used_ct_);
    t_test_n_used_examples_[tid] = t_test_n_used_examples_[tid] + 1;

    pthread_mutex_unlock(&mutex_ex_test_);
    return true;
  }
  else {
    pthread_mutex_unlock(&mutex_ex_test_);
    return false;
  }
}

double Learner::LinearPredictBias(const Svector &w, const Example &x, double bias) const {
  return x.SparseDot(w) + bias;
}

double Learner::LinearPredictBias(const Svector &w, const Svector &x, double bias) const {
  return x.SparseDot(w) + bias;
}

T_LBL Learner::LinearPredictBiasLabelBinary(const Svector &w, const Example &x, double bias) const {
  if (x.SparseDot(w) + bias > 0.0)
    return (T_LBL)1;
  else
    return (T_LBL)-1;
}

T_LBL Learner::LinearPredictBiasLabelBinary(const Svector &w, const Svector &x, double bias) const {
  if (x.SparseDot(w) + bias > 0.0)
    return (T_LBL)1;
  else
    return (T_LBL)-1;
}

