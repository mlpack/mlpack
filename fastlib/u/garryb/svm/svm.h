#ifndef U_SVM_SVM_H
#define U_SVM_SVM_H

#include "smo.h"

#include "fastlib/fastlib.h"

#include <typeinfo>

struct SVMLinearKernel {
  void Init(datanode *node) {}
  
  void Copy(const SVMLinearKernel& other) {}
  
  double Eval(const Vector& a, const Vector& b) const {
    return la::Dot(a, b);
  }
};

template<typename TKernel>
class SVM {
 public:
  typedef TKernel Kernel;
  
 private:
  TKernel kernel_;
  double c_;
  Matrix support_vectors_;
  double thresh_;
  Vector alpha_;
  
 public:
  void InitTrain(const Dataset& dataset, int n_classes, datanode *module);
  
  int Classify(const Vector& vector);
};

template<typename TKernel>
void SVM<TKernel>::InitTrain(
    const Dataset& dataset, int n_classes, datanode *module) {
  DEBUG_ASSERT_MSG(n_classes == 2, "SVM is only a binary classifier");
  
  fx_set_param(module, "kernel_type", typeid(TKernel).name());
  
  kernel_.Init(fx_submodule(module, "kernel", "kernel"));
  
  c_ = fx_param_double(module, "c", 1.0);
  
  SMO<Kernel> smo;
  smo.Init(&dataset, c_);
  smo.kernel().Copy(kernel_);
  smo.Train();
  
  thresh_ = smo.threshold();
  smo.GetSVM(&support_vectors_, &alpha_);
  DEBUG_ASSERT(alpha_.length() != 0);
  DEBUG_ASSERT(alpha_.length() == support_vectors_.n_cols());
  
  DEBUG_ONLY(fprintf(stderr, "----------------------\n"));
  DEBUG_ONLY(support_vectors_.PrintDebug("support vectors"));
  DEBUG_ONLY(alpha_.PrintDebug("support vector weights"));
  DEBUG_ONLY(fprintf(stderr, "-- THRESHOLD: %f\n", thresh_));
}

template<typename TKernel>
int SVM<TKernel>::Classify(const Vector& datum) {
  double summation = 0;
  
  for (index_t i = 0; i < alpha_.length(); i++) {
    Vector support_vector;
    support_vectors_.MakeColumnVector(i, &support_vector);
    double term = alpha_[i] * kernel_.Eval(datum, support_vector);
    
    DEBUG_MSG(0, "alpha %f, term %f", alpha_[i], term);
    
    summation += term;
  }
  
  DEBUG_MSG(0, "summation=%f, thresh_=%f", summation, thresh_);
  
  return (summation - thresh_ > 0.0) ? 1 : 0;
}

#endif
