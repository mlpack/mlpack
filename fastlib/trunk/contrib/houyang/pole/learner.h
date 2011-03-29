#ifndef LEARNER_H
#define LEARNER_H

#include <string>
#include <iostream>
#include <cmath>
#include <pthread.h>

#include "data.h"
#include "loss.h"
#include "log.h"

typedef vector<T_IDX> Vec_d;

using namespace std;

class Learner {
 public:
  bool v_;
  
  Data *TR_; // training set
  Data *VA_; // validation set
  Data *TE_; // testing set

  bool   random_data_;
  T_IDX  n_source_, port_;
  bool   read_port_; // read data from port or file
  string fn_learn_;
  string fn_predict_;
  // for los function
  Loss   *LF_;
  string lf_name_;
  // for learner logs and statistics
  Log    *LOG_;
  bool   calc_loss_; // calculate total loss
  T_IDX  n_log_; // How many log points
  // thread properties and statistics
  T_IDX  n_thread_; // number of threads for learning
  vector<pthread_t> Threads_;
  vector<T_IDX> t_state_; // thread state
  vector<T_IDX> t_n_it_; // nubmer of iterations
  vector<T_IDX> t_n_used_examples_;
  vector<double> t_loss_; // thread loss
  vector<T_IDX> t_err_; // thread error
  vector<Vec_d>  t_exp_err_; // expert error on each thread
  // for iterations
  T_IDX  epoch_ct_; // epoch counter
  T_IDX  n_epoch_; // number of learning epochs
  T_IDX  iter_res_ct_; // counter for iter_res
  T_IDX n_iter_res_; // number of training iterations besides epoches
  // for learning
  string opt_name_; // name of optimization method
  T_IDX  mb_size_; // Size of a mini-batch
  int    reg_type_; // type of regularization term
  double reg_factor_; // Regularization factor ('lambda' in avg_loss + lambda * regularization)
  double reg_C_; // Cost factor C ('C' in regularization + C*avg_loss)
  string type_; // learner type: classification / regression
  bool   use_bias_; // Add a bias term to examples
  T_IDX  n_expert_; // number of experts
  string wl_name_; // name of weak learner
  double alpha_; // Multiplication factor in Weighte Majority
  string kernel_name_; // name of kernel
  double sigma_; // sigma in Gaussian RBF kernel
  T_IDX trdim_; // dimension for transformed features
  int    comm_method_; // How agents communicate with each other
 private:
  pthread_mutex_t mutex_ex_;

 public:
  Learner();
  virtual ~Learner();
  
  void OnlineLearn();
  void BatchLearn();
  void ParallelLearn();
  //void SerialLearn();
  void FinishThreads();

  bool GetImmedExample(Data *D, Example **ex, T_IDX tid);

  virtual void Learn() {};
  virtual void Test() {};

  double LinearPredictBias(const Svector &w, const Example &x, double bias) const;
  double LinearPredictBias(const Svector &w, const Svector &x, double bias) const;
  T_LBL LinearPredictBiasLabelBinary(const Svector &w, const Example &x, double bias)const;
  T_LBL LinearPredictBiasLabelBinary(const Svector &w, const Svector &x, double bias)const;
};

#endif
