#ifndef WEAKLEARNER_H_
#define WEAKLEARNER_H_

#include "data.h"
#include "sparsela.h"

//---------------Weak Learner------------------//
class WeakLearner {
 public:
  string method_;
 public:
  WeakLearner() {};
  ~WeakLearner() {};
  // train a weak classifier
  virtual void  BatchLearn(Data *D) {};
  // binary prediction using weak learner
  virtual T_LBL PredictLabelBinary(Example *x) = 0;
};

//---------------Decision Stump------------------//
class DecisionStump : public WeakLearner {
 private:
  size_t sd_; // splitting dimension
  size_t n_it_;
  float  thd_; // threshold for decision
  T_LBL  gl_; // label for > thd
 public:
  DecisionStump(size_t split_dim, size_t num_iter);
  ~DecisionStump();
  
  void  BatchLearn(Data *D);
  T_LBL PredictLabelBinary(Example *x);
};


#endif

