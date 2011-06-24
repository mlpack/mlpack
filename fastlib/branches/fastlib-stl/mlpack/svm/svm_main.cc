/**
 * @author Hua Ouyang
 *
 * @file svm_main.cc
 *
 * This file contains main routines for performing  
 * 0. multiclass SVM classification (one-vs-one method is employed).
 * 1. SVM regression (epsilon-insensitive loss i.e. epsilon-SVR).
 * 2. one-class SVM (TODO)
 *
 * It provides four modes:
 * "cv": cross validation;
 * "train": model training
 * "train_test": training and then online batch testing; 
 * "test": offline batch testing.
 *
 * Please refer to README for detail description of usage and examples.
 *
 * @see svm.h
 * @see opt_smo.h
 */

#include "svm.h"
#include <fastlib/math/statistics.h>
#include <fastlib/fx/io.h>

#include <armadillo>
#include <fastlib/base/arma_compat.h>

using std::string;
using std::vector;


PARAM(double, "sigma", "(for Gaussian kernel) sigma in the gaussian kernel \
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

PARAM_STRING_REQ("learner_name", "The name of the support vector learner,\
 values: 'svm_c' for classification, 'svm_r' for regression, 'svn_de'\
 for one class SVM", "svm");

PARAM_STRING_REQ("mode", "The mode of svm_main, values: 'cv', 'train',\
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
PARAM_INT("wss", "Working set selection scheme.  1 for 1st order\
 expansion, 2 for 2nd order expansion. Default value: 1.", "svm", 1);

PARAM_FLAG("normalize", "Whether need to do data normalization before\
 training/testing, values: '0' for no normalize, '1' for normalize", "svm");

PROGRAM_INFO("SVM", "These are the implementations for Support Vector\
 Machines, including Multiclass classification, Regression, and One Class SVM");

using namespace mlpack;

/**
* Data Normalization
*
* @param: the dataset to be normalized
*/
void DoSvmNormalize(Dataset* dataset) {
  Matrix m;
  Vector sums;

  m.Init(dataset->n_features()-1, dataset->n_points());
  sums.Init(dataset->n_features() - 1);
  sums.SetZero();

  for (index_t i = 0; i < dataset->n_points(); i++) {
    Vector s;
    Vector d;
    for(index_t j = 0; j < dataset->n_features() - 1; j++)
      s[j] = dataset->matrix()(j, i);

    m.MakeColumnVector(i, &d);
    d.CopyValues(s);
    la::AddTo(s, &sums);
  }
  
  la::Scale(-1.0 / dataset->n_points(), &sums);
  for (index_t i = 0; i < dataset->n_points(); i++) {
    Vector d;
    m.MakeColumnVector(i, &d);
    la::AddTo(sums, &d);
  }
  
  Matrix cov;

  la::MulTransBInit(m, m, &cov);

  Vector d;
  Matrix u; // eigenvectors
  Matrix ui; // the inverse of eigenvectors

  PASSED(la::EigenvectorsInit(cov, &d, &u));
  la::TransposeInit(u, &ui);

  for (index_t i = 0; i < d.length(); i++) {
    d[i] = 1.0 / sqrt(d[i] / (dataset->n_points() - 1));
  }

  la::ScaleRows(d, &ui);

  Matrix cov_inv_half;
  la::MulInit(u, ui, &cov_inv_half);

  Matrix final;
  la::MulInit(cov_inv_half, m, &final);

  for (index_t i = 0; i < dataset->n_points(); i++) {
    Vector s;
    Vector d;
    for(index_t j = 0; j < dataset->n_features() - 1; j++)
      d[j] = dataset->matrix()(j, i);

    final.MakeColumnVector(i, &s);
    d.CopyValues(s);
  }

  if (IO::HasParam("svm/save")) {
    IO::GetParam<std::string>("kfold/save") = "1";
    dataset->WriteCsv("m_normalized.csv");
  }
}

/**
* Generate an artificial data set
*
* @param: the dataset to be generated
*/
void GenerateArtificialDataset(Dataset* dataset){
  Matrix m;
  index_t n = IO::GetParam<int>("svm/n") = 30;
  double offset = IO::GetParam<double>("svm/offset") = 0.0;
  double range = IO::GetParam<double>("svm/range") = 1.0;
  double slope = IO::GetParam<double>("svm/slope") = 1.0;
  double margin = IO::GetParam<double>("svm/margin") = 1.0;
  double var = IO::GetParam<double>("svm/var") = 1.0;
  double intercept = IO::GetParam<double>("svm/intercept") = 0.0;
    
  // 2 dimensional dataset, size n, 3 classes
  m.Init(3, n);
  for (index_t i = 0; i < n; i += 3) {
    double x;
    double y;
    
    x = (rand() * range / RAND_MAX) + offset;
    y = margin / 2 + (rand() * var / RAND_MAX);
    m.set(0, i, x);
    m.set(1, i, x*slope + y + intercept);
    m.set(2, i, 0); // labels
    
    x = (rand() * range / RAND_MAX) + offset;
    y = margin / 2 + (rand() * var / RAND_MAX);
    m.set(0, i+1, 10*x);
    m.set(1, i+1, x*slope + y + intercept);
    m.set(2, i+1, 1); // labels
    
    x = (rand() * range / RAND_MAX) + offset;
    y = margin / 2 + (rand() * var / RAND_MAX);
    m.set(0, i+2, 20*x);
    m.set(1, i+2, x*slope + y + intercept);
    m.set(2, i+2, 2); // labels
  }

  arma::mat tmp;
  arma_compat::matrixToArma(m, tmp);
  data::Save("artificialdata.csv", tmp); // TODO, for training, for testing
  // this is a bad way to do this
  dataset->CopyMatrix(tmp);
}

/**
* Load data set from data file. If data file not exists, generate an 
* artificial data set.
*
* @param: the dataset
* @param: name of the data file to be loaded
*/
index_t LoadData(Dataset* dataset, string datafilename){
  if (IO::HasParam(datafilename.c_str())) {
    // when a data file is specified, use it.
    if ( !PASSED(dataset->InitFromFile( IO::GetParam<std::string>(datafilename.c_str()).c_str() )) ) {
    fprintf(stderr, "Couldn't open the data file.\n");
    return 0;
    }
  } 
  else {
    fprintf(stderr, "No data file exist. Generating artificial dataset.\n");
    // otherwise, generate an artificial dataset and save it to "m.csv"
    GenerateArtificialDataset(dataset);
  }
  
  if (IO::HasParam("svm/normalize")) {
    fprintf(stderr, "Normalizing...\n");
    DoSvmNormalize(dataset);
  } else {
    fprintf(stderr, "Skipping normalization...\n");
  }
  return 1;
}

/**
* Multiclass SVM classification/ SVM regression - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[]) {
  IO::ParseCommandLine(argc, argv);
  srand(time(NULL));

  string mode = IO::GetParam<std::string>("svm/mode");
  string kernel = IO::GetParam<std::string>("svm/kernel");
  string learner_name = IO::GetParam<std::string>("svm/learner_name");
  index_t learner_typeid;
  
  if (learner_name == "svm_c") { // Support Vector Classfication
    learner_typeid = 0;
  }
  else if (learner_name == "svm_r") { // Support Vector Regression
    learner_typeid = 1;
  }
  else if (learner_name == "svm_de") { // One Class Support Vector Machine
    learner_typeid = 2;
  }
  else {
    fprintf(stderr, "Unknown support vector learner name! Program stops!\n");
    return 0;
  }
  
  // TODO: more kernels to be supported

  /* Cross Validation Mode, need cross validation data */
  if(mode == "cv") { 
    fprintf(stderr, "SVM Cross Validation... \n");
    
    /* Load cross validation data */
    Dataset cvset;
    if (LoadData(&cvset, "cv_data") == 0)
    return 1;
    
    if (kernel == "linear") {
      GeneralCrossValidator< SVM<SVMLinearKernel> > cross_validator; 
      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds, need k_cv>1 */
      cross_validator.Init(learner_typeid, IO::GetParam<int>("svm/k_cv"), &cvset, NULL, "svm");
      /* k_cv folds cross validation; (true): do training set permutation */
      cross_validator.Run(true);
      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
    else if (kernel == "gaussian") {
      GeneralCrossValidator< SVM<SVMRBFKernel> > cross_validator; 
      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds */
      cross_validator.Init(learner_typeid, IO::GetParam<int>("svm/k_cv"), &cvset, NULL, "svm");
      /* k_cv folds cross validation; (true): do training set permutation */
      cross_validator.Run(true);
      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
  }
  /* Training Mode, need training data | Training + Testing(online) Mode, need training data + testing data */
  else if (mode=="train" || mode=="train_test"){
    fprintf(stderr, "SVM Training... \n");

    /* Load training data */
    Dataset trainset;
    if (LoadData(&trainset, "train_data") == 0) // TODO:param_req
      return 1;
    
    /* Begin SVM Training | Training and Testing */
    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.InitTrain(learner_typeid, trainset);
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "SVM Predicting... \n");
	/* Load testing data */
	Dataset testset;
	if (LoadData(&testset, "test_data") == 0) // TODO:param_req
	  return 1;
	svm.BatchPredict(learner_typeid, testset, "predicted_values");
      }
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.InitTrain(learner_typeid, trainset);
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "SVM Predicting... \n");
	/* Load testing data */
	Dataset testset;
	if (LoadData(&testset, "test_data") == 0) // TODO:param_req
	  return 1;
	svm.BatchPredict(learner_typeid, testset, "predicted_values"); // TODO:param_req
      }
    }
  }
  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode=="test") {
    fprintf(stderr, "SVM Predicting... \n");

    /* Load testing data */
    Dataset testset;
    if (LoadData(&testset, "test_data") == 0) // TODO:param_req
      return 1;

    /* Begin Prediction */
    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.Init(learner_typeid, testset); 
      svm.LoadModelBatchPredict(learner_typeid, testset, "svm_model", "predicted_values"); // TODO:param_req
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.Init(learner_typeid, testset); 
      svm.LoadModelBatchPredict(learner_typeid, testset, "svm_model", "predicted_values"); // TODO:param_req
    }
  }
}

