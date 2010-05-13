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
 * @see smo.h
 */

#include "svm.h"
#include "fastlib/math/statistics.h"

const fx_entry_doc svm_main_entries_doc[] = {
  {"learner_name", FX_REQUIRED, FX_STR, NULL,
   "  The name of the support vecotr learner, values: \"svm_c\" for classification, \"svm_r\" for regression, \"svm_de\" for one class SVM\n"},
  {"mode", FX_REQUIRED, FX_STR, NULL,
   "  The mode of svm_main, values: \"cv\", \"train\", \"train_test\", \"test\".\n"},
  {"k_cv", FX_PARAM, FX_INT, NULL,
   "  The number of folds for cross validation, only required under \"cv\" mode.\n"},
  {"cv_data", FX_PARAM, FX_STR, NULL,
   "  The file name for cross validation data, only required under \"cv\" mode.\n"},
  {"train_data", FX_PARAM, FX_STR, NULL,
   "  The file name for training data, only required under \"train\" or \"train_test\" mode.\n"},
  {"test_data", FX_PARAM, FX_STR, NULL,
   "  The file name for testing data, only required under \"test\" or \"train_test\" mode.\n"},
  {"kernel", FX_REQUIRED, FX_STR, NULL,
   "  Kernel name, values:\"linear\", \"gaussian\".\n"},
  {"sigma", FX_PARAM, FX_DOUBLE, NULL,
   "  (for Gaussian kernel) sigma in the gaussian kernel k(x1,x2)=exp(-(x1-x2)^2/(2sigma^2)), only required when using \"guassian\" kernel\n"},
  {"c", FX_PARAM, FX_DOUBLE, NULL,
   "  (for SVM_C) the weight (0~1) that controls compromise between large margins and small margin violations. Default value: 10.0.\n"},
  {"c_p", FX_PARAM, FX_DOUBLE, NULL,
   "  (for SVM_C) the weight (0~1) for the positive class (y==1). Default value: c.\n"},
  {"c_n", FX_PARAM, FX_DOUBLE, NULL,
   "  (for SVM_C) the weight (0~1) for the negative class (y==-1). Default value: c.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "  (for SVM_R) the epsilon in SVM regression of epsilon-insensitive loss. Default value: 0.1.\n"},
  {"wss", FX_PARAM, FX_INT, NULL,
   "  Working set selection scheme. 1 for 1st order expansion; 2 for 2nd order expansion. Default value: 1.\n"},
  {"normalize", FX_PARAM, FX_BOOL, NULL,
   "  Whether need to do data normalization before training/testing, values: \"0\" for no normalize, \"1\" for normalize.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc svm_main_doc = {
  svm_main_entries_doc, NULL,
  "These are the implementations for Support Vector Machines (SVM), including Multiclass classification, Regression, and One Class SVM)\n"
};

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
    dataset->matrix().MakeColumnSubvector(i, 0, dataset->n_features()-1, &s);
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
    dataset->matrix().MakeColumnSubvector(i, 0, dataset->n_features()-1, &d);
    final.MakeColumnVector(i, &s);
    d.CopyValues(s);
  }

  if (fx_param_bool(NULL, "save", 0)) {
    fx_default_param(NULL, "kfold/save", "1");
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
  index_t n = fx_param_int(NULL, "n", 30);
  double offset = fx_param_double(NULL, "offset", 0.0);
  double range = fx_param_double(NULL, "range", 1.0);
  double slope = fx_param_double(NULL, "slope", 1.0);
  double margin = fx_param_double(NULL, "margin", 1.0);
  double var = fx_param_double(NULL, "var", 1.0);
  double intercept = fx_param_double(NULL, "intercept", 0.0);
    
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

  data::Save("artificialdata.csv", m); // TODO, for training, for testing
  dataset->OwnMatrix(&m);
}

/**
* Load data set from data file. If data file not exists, generate an 
* artificial data set.
*
* @param: the dataset
* @param: name of the data file to be loaded
*/
int LoadData(Dataset* dataset, String datafilename){
  if (fx_param_exists(NULL, datafilename)) {
    // when a data file is specified, use it.
    if ( !PASSED(dataset->InitFromFile( fx_param_str_req(NULL, datafilename) )) ) {
    fprintf(stderr, "Couldn't open the data file.\n");
    return 0;
    }
  } 
  else {
    fprintf(stderr, "No data file exist. Generating artificial dataset.\n");
    // otherwise, generate an artificial dataset and save it to "m.csv"
    GenerateArtificialDataset(dataset);
  }
  
  if (fx_param_bool(NULL, "normalize", 1)) {
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
  //fx_init(argc, argv, NULL);
  fx_module *root = fx_init(argc, argv, &svm_main_doc);
  srand(time(NULL));

  String mode = fx_param_str_req(NULL, "mode");
  String kernel = fx_param_str_req(NULL, "kernel");
  String learner_name = fx_param_str_req(root,"learner_name");
  int learner_typeid;
  
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
      cross_validator.Init(learner_typeid, fx_param_int_req(NULL,"k_cv"), &cvset, fx_root, "svm");
      /* k_cv folds cross validation; (true): do training set permutation */
      cross_validator.Run(true);
      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
    else if (kernel == "gaussian") {
      GeneralCrossValidator< SVM<SVMRBFKernel> > cross_validator; 
      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds */
      cross_validator.Init(learner_typeid, fx_param_int_req(NULL,"k_cv"), &cvset, fx_root, "svm");
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
    datanode *svm_module = fx_submodule(fx_root, "svm");

    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.InitTrain(learner_typeid, trainset, svm_module);
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
      svm.InitTrain(learner_typeid, trainset, svm_module);
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
    datanode *svm_module = fx_submodule(fx_root, "svm");

    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.Init(learner_typeid, testset, svm_module); 
      svm.LoadModelBatchPredict(learner_typeid, testset, "svm_model", "predicted_values"); // TODO:param_req
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.Init(learner_typeid, testset, svm_module); 
      svm.LoadModelBatchPredict(learner_typeid, testset, "svm_model", "predicted_values"); // TODO:param_req
    }
  }
  fx_done(NULL);
}

