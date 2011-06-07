//****************************
//* Naive Kernel PCA in batch
//* Hua Ouyang 04/28/2011
//* Example: ./pole_pt -d heart_scale -t heart_scale -m bkpca --kernel gaussian --center 1 --sigma 3
//****************************
#ifndef OPT_BKPCA_H
#define OPT_BKPCA_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>
#include <armadillo>

#include "learner.h"
#include "kernel.h"

using namespace arma;

template <typename TKernel>
class BKPCA : public Learner {
 public:
  Mat<T_VAL> K_; // kernel matrix
  Col<T_VAL> eigval_; // eigenvalues
  Mat<T_VAL> eigvec_; // eigenvectors
  uvec eig_order_; // index of ascending eigenvalues
  Mat<T_VAL> test_f_; // features for test data; size: maxeig x n_TE
 private:
  TKernel kn_; // for random features
 public:
  BKPCA();
  void Learn();
  void Test();
 private:
  void MakeLog(T_IDX tid, const Col<T_VAL> &x, T_LBL y, double pred_val);
  void SaveLog();
};


template <typename TKernel>
BKPCA<TKernel>::BKPCA() {
  cout << "---Batch Kernel PCA---" << endl;
}

template <typename TKernel>
void BKPCA<TKernel>::Learn() {
  T_IDX n = TR_->EXs_.n_cols; // number of training samples
  // init kernel
  if (kernel_name_ == "gaussian") {
    kn_.sigma_ = sigma_;
  }
  K_.set_size(n, n);
  // for symmectic kernels
  if (kernel_name_ == "gaussian" || kernel_name_ == "linear" ) {
    for (T_IDX i=0; i<n; i++) {
      for (T_IDX j=0; j<=i; j++) {
        K_(i,j) = kn_.Eval(TR_->EXs_.col(i), TR_->EXs_.col(j)) / (double)n; // lower tri
        K_(j,i) = K_(i,j); // upper tri
      }
    }
  }
  else {
    for (T_IDX i=0; i<n; i++) {
      for (T_IDX j=0; j<n; j++) {
        K_(i,j) = kn_.Eval(TR_->EXs_.col(i), TR_->EXs_.col(j)) / (double)n;
      }
    }
  }
  cout << "Kernel matrix filled. Begin to eigendecompose...";
  // eigen decomposition
  eig_sym(eigval_, eigvec_, K_);
  cout << "done!"<< endl;
  // sort eigenvalues ascendingly
  eig_order_ = sort_index(eigval_, 0);
  //cout << eigval_ << endl;
  //cout << eig_order << endl;
  SaveLog();
}

template <typename TKernel>
void BKPCA<TKernel>::Test() {
  Col<T_VAL> kval;
  T_IDX n_tr = TR_->EXs_.n_cols;
  T_IDX n_te;
  if (fn_learn_ != fn_predict_) {
    n_te = TE_->EXs_.n_cols;
  }
  else {
    n_te = n_tr;
  }
  kval.set_size(n_tr);
  test_f_.set_size(maxeig_, n_te);
  for (T_IDX t=0; t<n_te; t++) {
    if (fn_learn_ != fn_predict_) {
      // TE_ != TR_
      for (T_IDX i=0; i<n_tr; i++) {
        kval(i) = kn_.Eval(TE_->EXs_.col(t), TR_->EXs_.col(i));
      }
    }
    else {
      // TE_ == TR_
      for (T_IDX i=0; i<n_tr; i++) {
        kval(i) = kn_.Eval(TR_->EXs_.col(t), TR_->EXs_.col(i));
      }
    }
    for (T_IDX g=0; g<maxeig_; g++) {
      // calculate projections
      test_f_(g, t) = dot(kval, eigvec_.col(g));
    }
  }
}

template <typename TKernel>
void BKPCA<TKernel>::MakeLog(T_IDX tid, const Col<T_VAL> &x, T_LBL y, double pred_val) {
  
}

template <typename TKernel>
void BKPCA<TKernel>::SaveLog() {
  
}


#endif
