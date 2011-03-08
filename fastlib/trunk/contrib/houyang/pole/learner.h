#ifndef LEARNER_H
#define LEARNER_H

#include <string>
#include <iostream>
#include <cmath>
#include <pthread.h>

#include "data.h"
#include "loss.h"
#include "log.h"

typedef vector<size_t> Vec_d;

using namespace std;

class Learner {
 public:
  bool v_;
  
  Data *TR_; // training set
  Data *VA_; // validation set
  Data *TE_; // testing set

  bool   random_data_;
  size_t n_source_, port_;
  bool   read_port_; // read data from port or file
  string fn_learn_;
  string fn_predict_;
  // for los function
  Loss   *LF_;
  string lf_name_;
  // for learner logs and statistics
  Log    *LOG_;
  bool   calc_loss_; // calculate total loss
  size_t n_log_; // How many log points
  // thread properties and statistics
  size_t n_thread_; // number of threads for learning
  vector<pthread_t> Threads_;
  vector<size_t> t_state_; // thread state
  vector<size_t> t_n_it_; // nubmer of iterations
  vector<size_t> t_n_used_examples_;
  vector<double> t_loss_; // thread loss
  vector<size_t> t_err_; // thread error
  vector<Vec_d>  t_exp_err_; // expert error on each thread
  // for iterations
  size_t epoch_ct_; // epoch counter
  size_t n_epoch_; // number of learning epochs
  size_t iter_res_ct_; // counter for iter_res
  size_t n_iter_res_; // number of training iterations besides epoches
  // for learning
  string opt_name_; // name of optimization method
  size_t mb_size_; // Size of a mini-batch
  int    reg_type_; // type of regularization term
  double reg_factor_; // Regularization factor ('lambda' in avg_loss + lambda * regularization)
  double reg_C_; // Cost factor C ('C' in regularization + C*avg_loss)
  string type_; // learner type: classification / regression
  bool   use_bias_; // Add a bias term to examples
  size_t n_expert_; // number of experts
  string wl_name_; // name of weak learner
  double alpha_; // Multiplication factor in Weighte Majority
  string kernel_name_; // name of kernel
  double sigma_; // sigma in Gaussian RBF kernel
  int    comm_method_; // How agents communicate with each other
 private:
  pthread_mutex_t mutex_ex_;

 public:
  Learner();
  ~Learner();
  
  void OnlineLearn();
  void BatchLearn();
  void ParallelLearn();
  //void SerialLearn();
  void FinishThreads();

  bool GetImmedExample(Data *D, Example **ex, size_t tid);

  virtual void Learn() {};
  virtual void Test() {};

  double LinearPredictBias(Svector *w, Example *x, double bias);
  T_LBL LinearPredictBiasLabelBinary(Svector *w, Example *x, double bias);
};

#endif
