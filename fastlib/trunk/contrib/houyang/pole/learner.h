#ifndef LEARNER_H
#define LEARNER_H

#include <string>
#include <iostream>

#include "data.h"
#include "loss.h"

/*
#include "weak_learner.h"
#include "sparse.h"
*/

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

  string lf_name_;
  Loss *LF_;

  string opt_name_;

  //WeakLearner **WL_;

  size_t n_thread_; // number of threads for learning
  size_t n_epoch_; // number of learning epochs
  size_t n_iter_res_; // number of training iterations besides epoches
  int    reg_type_; // type of regularization term
  double reg_factor_; // Regularization factor ('lambda' in avg_loss + lambda * regularization)
  double reg_C_; // Cost factor C ('C' in regularization + C*avg_loss)
  string type_; // learner type: classification / regression
  bool   use_bias_; // Add a bias term to examples
  size_t n_expert_; // number of experts
  string wl_name_; // name of weak learner
  double alpha_; // Multiplication factor in Weighte Majority
  int    comm_method_; // How agents communicate with each other
  size_t mb_size_; // Size of a mini-batch
  bool   calc_loss_; // calculate total loss
  size_t n_log_; // How many log points

 public:
  Learner();
  ~Learner();
  
  void OnlineLearn();
  void BatchLearn();

  virtual void Learn() {};
  virtual void Test() {};

};

#endif
