//****************************
//* Naive Kernel PCA in batch
//* Hua Ouyang 04/28/2011
//* Example: ./pole_pt -d toy_train -t toy_predict -m bkpca --kernel gaussian --center 0 --sigma 0.2236 --maxeig 8
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
  Mat<T_VAL> K_test_; // kernel matrix
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
        K_(i,j) = kn_.Eval(TR_->EXs_.col(i), TR_->EXs_.col(j)); // lower tri
        K_(j,i) = K_(i,j); // upper tri
      }
    }
  }
  else {
    for (T_IDX i=0; i<n; i++) {
      for (T_IDX j=0; j<n; j++) {
        K_(i,j) = kn_.Eval(TR_->EXs_.col(i), TR_->EXs_.col(j));
      }
    }
  }
  // centering training data in feature space
  Mat<T_VAL> OnesM;
  OnesM.ones(n, n);
  OnesM = OnesM / n;
  K_ = K_ - OnesM*K_ - K_*OnesM + OnesM*K_*OnesM;
  cout << "Kernel matrix filled and centered. Begin to eigendecompose...";
  // eigen decomposition
  eig_sym(eigval_, eigvec_, K_);
  cout << "done!"<< endl;
  // sort eigenvalues descendingly
  eig_order_ = sort_index(eigval_, 1);
  cout << eigval_.n_rows << endl;
  //cout << eig_order << endl;
  SaveLog();
}

template <typename TKernel>
void BKPCA<TKernel>::Test() {
  T_IDX n_tr = TR_->EXs_.n_cols;
  if (maxeig_ > n_tr) {
    cout << "ERROR! Maximum number of eigenvalues larger than number of training data!" << endl;
    exit(1);
  }
  T_IDX n_te;
  if (fn_learn_ != fn_predict_) {
    n_te = TE_->EXs_.n_cols;
  }
  else {
    n_te = n_tr;
  }
  K_test_.set_size(n_te, n_tr);
  test_f_.set_size(n_te, maxeig_);
  if (fn_learn_ != fn_predict_) {
    // TE_ != TR_
    for (T_IDX t=0; t<n_te; t++) {
      for (T_IDX i=0; i<n_tr; i++) {
        K_test_(t,i) = kn_.Eval(TE_->EXs_.col(t), TR_->EXs_.col(i));
      }
    }
  }
  else {
    // TE_ == TR_
    for (T_IDX t=0; t<n_te; t++) {
      for (T_IDX i=0; i<n_tr; i++) {
        K_test_(t,i) = kn_.Eval(TR_->EXs_.col(t), TR_->EXs_.col(i));
      }
    }
  }
  // centering test data in feature space
  Mat<T_VAL> OnesMTr, OnesMTe;
  OnesMTr.ones(n_tr, n_tr);
  OnesMTe.ones(n_te, n_tr);
  OnesMTr = OnesMTr / n_tr;
  OnesMTe = OnesMTe / n_tr;
  K_test_ = K_test_ - OnesMTe*K_ - K_test_*OnesMTr + OnesMTe*K_*OnesMTr;

  for (T_IDX t=0; t<n_te; t++) {
    for (T_IDX g=0; g<maxeig_; g++) {
      test_f_(t,g) = dot(trans(K_test_.row(t)), eigvec_.col(eig_order_(g)));
    }
  }

  SaveLog();
}

template <typename TKernel>
void BKPCA<TKernel>::SaveLog() {
  string log_fn(TR_->fn_);
  log_fn += ".";
  log_fn += opt_name_;
  log_fn += ".log";
  test_f_.save(log_fn, raw_ascii);
}


#endif
