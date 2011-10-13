/**
 * @author Hua Ouyang
 *
 * @file svm.h
 *
 * This header file contains functions for performing SVM training and prediction
 * Supported SVM learner type:SVM_C, SVM_R, SVM_DE
 *
 * @see opt_smo.h
 */
#ifndef __MLPACK_METHODS_SVM_SVM_H
#define __MLPACK_METHODS_SVM_SVM_H

#include "opt_smo.h"

#include <mlpack/core.h>

#include <typeinfo>
#include <vector>
#include <err.h>

PARAM_MODULE("svm","Parameters for Support Vector Machines.");

PARAM(double, "sigma", "(for Gaussian kernel) sigma in the gaussian kernel\
    k(x1,x2)=exp(-(x1-x2)^2/(2sigma^2)), only required when using 'guassian' kernel"\
    ,"svm", 0.0, false);
PARAM(double, "c", "(for SVM_C) the weight (0~1) that controls compromise\
    between large margins and small margin violations. Default value: 10.0.",\
    "svm", 10.0, false);
PARAM(double, "c_p", "(for SVM_C) the weight (0~1) for the positive class\
    (y==1). Default value: c.", "svm", 3e8, false);
PARAM(double, "c_n", "(for SVM_C) the weight (0~1) for the negative class\
    (y==-1). Default value: c.", "svm", 3e8, false);
PARAM(double, "epsilon", "(for SVM_R) the epsilon in SVM regression of\
    epsilon-insensitive loss. Default value: 0.1.", "svm", 0.1, false);
PARAM(double, "wss", "Working set selection scheme.  1 for 1st order\
    expansion, 2 for 2nd order expansion. Default value: 1.", "svm", 1, false);
PARAM(double, "accuracy", "The minimum accuracy required to stop the optimization\
    algorithm. Default 1e-4.", "svm", 1e-4, false);

PARAM(size_t, "hinge", "Value for hinge loss. Default 1.", "svm", 1, false);
PARAM(size_t, "n_iter", "Maximum number of iterations. Default 100000000.",
    "svm", 100000000, false);

PARAM(std::string, "opt", "The optimization algorithm to use. A the moment\
    the only choice is SMO.", "svm", "smo", false); /* TODO: Update if more added */

PARAM_STRING_REQ("learner_name", "The name of the support vector learner,\
    values: 'svm_c' for classification, 'svm_r' for regression, 'svn_de'\
    for one class SVM", "svm");

PARAM_STRING_REQ("mode", "The mode of svm_main, values: 'train',\
    'train_test', 'test'.", "svm");

PARAM_STRING_REQ("kernel", "Kernel name, values: 'linear', 'gaussian'.", "svm");
PARAM_STRING("cv_data", "The file name for cross validation data, only\
    required  under 'cv' mode.", "svm", "");
PARAM_STRING("train_data", "The file name for training data, only required\
    under 'train' or 'train_test' mode.", "svm", "");
PARAM_STRING("test_data", "The file name for testing data, only required\
    under 'test' or 'train_test' mode.", "svm", "");

PARAM_INT("k_cv", "The number of folds for cross validation, only required\
    under 'cv' mode.", "svm", 0);

PARAM_FLAG("normalize", "Whether need to do data normalization before\
    training/testing, values: '0' for no normalize, '1' for normalize", "svm");
PARAM(bool, "shrink", "Whether we shrink every so many iterations or not.\
    Only used with SMO.", "svm", true, false);

#define ID_LINEAR 0
#define ID_GAUSSIAN 1

namespace mlpack {
namespace svm {

/**
 * Class for Linear Kernel
 */
class SVMLinearKernel {
 public:
  /* Init of kernel parameters */
  std::vector<double> kpara_; /* kernel parameters */

  SVMLinearKernel() { /*TODO: NULL->node */ }

  /* Kernel name */
  void GetName(std::string& kname) {
    kname = "linear";
  }

  /* Get an type ID for kernel */
  size_t GetTypeId() {
    return ID_LINEAR;
  }

  /* Kernel value evaluation */
  double Eval(const arma::vec& a, const arma::vec b, size_t n_cols) const {
    return dot(a.rows(0, n_cols), b.rows(0, n_cols));
  }

  /* Save kernel parameters to file */
  void SaveParam(std::ostream& f) { }
};

/**
 * Class for Gaussian RBF Kernel
 */
class SVMRBFKernel {
 public:
  /* Init of kernel parameters */
  std::vector<double> kpara_; /* kernel parameters */

  SVMRBFKernel() { /*TODO: NULL->node */
    kpara_.resize(2,0);
    kpara_[0] = mlpack::CLI::GetParam<double>("svm/sigma"); /*sigma */
    kpara_[1] = -1.0 / (2 * pow(kpara_[0], 2)); /*gamma */
  }

  /* Kernel name */
  void GetName(std::string &kname) {
    kname = "gaussian";
  }

  /* Get an type ID for kernel */
  size_t GetTypeId() {
    return ID_GAUSSIAN;
  }

  /* Kernel value evaluation */
  double Eval(const arma::vec& a, const arma::vec& b, size_t n_features) const {
    double distance_squared = arma::norm(a.subvec(0, n_features - 1) - b.subvec(0, n_features - 1), 2);
    return exp(kpara_[1] * distance_squared);
  }

  /* Save kernel parameters to file */
  void SaveParam(std::ostream& f) {
    f << "sigma " << kpara_[0] << std::endl;
    f << "gamma " << kpara_[1] << std::endl;
  }
};

/**
 * Class for SVM
 */
template<typename TKernel>
class SVM {
 private:
  /**
   * Type id of the SVM learner:
   *  0:SVM Classification (svm_c);
   *  1:SVM Regression (svm_r);
   *  2:One class SVM (svm_de);
   * Developers may add more learner types if necessary
   */
  size_t n_labels;
  size_t learner_typeid_;

  /* Optimization method: smo */
  std::string opt_method_;

  /* array of models for storage of the 2-class(binary) classifiers
   * Need to train num_classes_*(num_classes_-1)/2 binary models */
  struct SVM_MODELS {
    /* bias term in each binary model */
    double bias_;
    /* all coefficients (alpha*y) of the binary dataset, not necessarily those of SVs */
    std::vector<double> coef_;
  };

  std::vector<SVM_MODELS> models_;

  /* list of labels, double type, but may be converted to integers.
   * e.g. [0.0,1.0,2.0] for a 3-class dataset */
  std::vector<int> train_labels_list_;
  /* array of label indices, after grouping. e.g. [c1[0,5,6,7,10,13,17],c2[1,2,4,8,9],c3[...]]*/
  std::vector<size_t> train_labels_index_;
  /* counted number of label for each class. e.g. [7,5,8]*/
  std::vector<size_t> train_labels_count_;
  /* start positions of each classes in the training label list. e.g. [0,7,12] */
  std::vector<size_t> train_labels_startpos_;

  /* total set of support vectors and their coefficients */
  arma::mat sv_;
  arma::mat sv_coef_;
  std::vector<bool> trainset_sv_indicator_;

  /* total number of support vectors */
  size_t total_num_sv_;
  /* support vector list to store the indices (in the training set) of support vectors */
  std::vector<size_t> sv_index_;
  /* start positions of each class of support vectors, in the support vector list */
  std::vector<size_t> sv_list_startpos_;
  /* counted number of support vectors for each class */
  std::vector<size_t> sv_list_ct_;

  /* SVM parameters */
  struct PARAMETERS {
    TKernel kernel_;
    std::string kernelname_;
    size_t kerneltypeid_;
    size_t b_;
    double C_;
    /* for SVM_C of unbalanced data */
    double Cp_; /* C for y==1 */
    double Cn_; /* C for y==-1 */
    /* for SVM_R */
    double epsilon_;
    /* working set selection scheme, 1 for 1st order expansion; 2 for 2nd order expansion */
    double wss_;
    /* whether do L1-SVM (1) or L2-SVM (2) */
    size_t hinge_sqhinge_; /* L1-SVM uses hinge loss, L2-SVM uses squared hinge loss */
    /* accuracy for the optimization stopping creterion */
    double accuracy_;
    /* number of iterations */
    size_t n_iter_;
  } param_;

  /* number of data samples */
  size_t n_data_;
  /* number of classes in the training set */
  size_t num_classes_;
  /* number of binary models to be trained, i.e. num_classes_*(num_classes_-1)/2 */
  size_t num_models_;
  size_t num_features_;

 public:
  typedef TKernel Kernel;
  class SMO<Kernel>;

  void Init(size_t learner_typeid, arma::mat& dataset);
  void InitTrain(size_t learner_typeid, arma::mat& dataset);

  double Predict(size_t learner_typeid, const arma::vec& vector);
  void BatchPredict(size_t learner_typeid, arma::mat& testset, std::string predictedvalue_filename);
  void LoadModelBatchPredict(size_t learner_typeid, arma::mat& testset, std::string model_filename, std::string predictedvalue_filename);
  std::vector<int>& getLabels (arma::mat& data, std::vector<int>& labels);
  void reorderDataByLabels(arma::mat& data);
  arma::mat& buildDataByLabels (size_t i, size_t j, arma::mat& data, arma::mat& returnData);

 private:
  void SVM_C_Train_(size_t learner_typeid, arma::mat& dataset);
  void SVM_R_Train_(size_t learner_typeid, arma::mat& dataset);
  void SVM_DE_Train_(size_t learner_typeid, arma::mat& dataset);
  double SVM_C_Predict_(const arma::vec& vector);
  double SVM_R_Predict_(const arma::vec& vector);
  double SVM_DE_Predict_(const arma::vec& vector);

  void SaveModel_(size_t learner_typeid, std::string model_filename);
  void LoadModel_(size_t learner_typeid, std::string model_filename);
};

}; // namespace svm
}; // namespace mlpack

// Include implementation.
#include "svm_impl.h"

#endif
