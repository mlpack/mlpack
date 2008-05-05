/**
 * @author Hua Ouyang
 *
 * @file rvm_main.cc
 *
 * This file contains main routines for Relevance Vector Machine (RVM):
 * multiclass RVM classification and RVM regression
 *
 * It provides four modes:
 * "cv": cross validation;
 * "train": model training
 * "train_test": training and then online batch testing; 
 * "test": offline batch testing.
 *
 * @see rvm.h, sbl_est.h
 */

#include "rvm.h"

/**
* Data Normalization
*
* @param: the dataset to be normalized
*/
void DoRvmNormalize(Dataset* dataset) {
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
    fprintf(stderr, "Normalizing\n");
    DoRvmNormalize(dataset);
  } else {
    fprintf(stderr, "Skipping normalize\n");
  }
  return 1;
}

/**
* Multiclass RVM classification and RVM regression - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  srand(time(NULL));

  String mode = fx_param_str_req(NULL, "mode");
  String kernel = fx_param_str_req(NULL, "kernel");
  String learner_name = fx_param_str_req(NULL,"learner_name");
  int learner_typeid;
  
  if (learner_name == "rvm_c") { // Relevance Vector Classfication
    learner_typeid = 0;
  }
  else if (learner_name == "rvm_r") { // Relevance Vector Regression
    learner_typeid = 1;
  }
  else {
    fprintf(stderr, "Unknown relevance vector learner name! Program stops!\n");
    return 0;
  }

  /* Cross Validation Mode, need cross validation data */
  if(mode == "cv") { 
    fprintf(stderr, "RVM Cross Validation... \n");
    
    /* Load cross validation data */
    Dataset cvset;
    if (LoadData(&cvset, "cv_data") == 0)
    return 1;
    
    if (kernel == "linear") {
      GeneralCrossValidator< RVM<RVMLinearKernel> > cross_validator; 
      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds, need k_cv>1 */
      cross_validator.Init(learner_typeid, fx_param_int_req(NULL,"k_cv"), &cvset, fx_root, "rvm");
      /* k_cv folds cross validation; (true): do training set permutation */
      cross_validator.Run(true);
      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
    else if (kernel == "gaussian") {
      GeneralCrossValidator< RVM<RVMRBFKernel> > cross_validator; 
      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds */
      cross_validator.Init(learner_typeid, fx_param_int_req(NULL,"k_cv"), &cvset, fx_root, "rvm");
      /* k_cv folds cross validation; (true): do training set permutation */
      cross_validator.Run(true);
      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
  }
  /* Training Mode, need training data | Training + Testing(online) Mode, need training data + testing data */
  else if (mode=="train" || mode=="train_test"){
    fprintf(stderr, "RVM Training... \n");

    /* Load training data */
    Dataset trainset;
    if (LoadData(&trainset, "train_data") == 0) // TODO:param_req
      return 1;
    
    /* Begin RVM Training | Training and Testing */
    datanode *rvm_module = fx_submodule(fx_root, NULL, "rvm");

    if (kernel == "linear") {
      RVM<RVMLinearKernel> rvm;
      rvm.InitTrain(learner_typeid, trainset, trainset.n_labels(), rvm_module);
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "RVM Predicting... \n");
	/* Load testing data */
	Dataset testset;
	if (LoadData(&testset, "test_data") == 0) // TODO:param_req
	  return 1;
	rvm.BatchPredict(&testset, "predicted_values");
      }
    }
    else if (kernel == "gaussian") {
      RVM<RVMRBFKernel> rvm;
      rvm.InitTrain(learner_typeid, trainset, trainset.n_labels(), rvm_module);
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "RVM Predicting... \n");
	/* Load testing data */
	Dataset testset;
	if (LoadData(&testset, "test_data") == 0) // TODO:param_req
	  return 1;
	rvm.BatchPredict(&testset, "predicted_values"); // TODO:param_req
      }
    }
  }
  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode=="test") {
    fprintf(stderr, "RVM Predicting... \n");

    /* Load testing data */
    Dataset testset;
    if (LoadData(&testset, "test_data") == 0) // TODO:param_req
      return 1;

    /* Begin Classification */
    datanode *rvm_module = fx_submodule(fx_root, NULL, "rvm");

    if (kernel == "linear") {
      RVM<RVMLinearKernel> rvm;
      rvm.Init(learner_typeid, testset, testset.n_labels(), rvm_module); // TODO:n_labels() -> num_classes_
      rvm.LoadModelBatchPredict(&testset, "rvm_model", "predicted_values"); // TODO:param_req
    }
    else if (kernel == "gaussian") {
      RVM<RVMRBFKernel> rvm;
      rvm.Init(learner_typeid, testset, testset.n_labels(), rvm_module); // TODO:n_labels() -> num_classes_
      rvm.LoadModelBatchPredict(&testset, "rvm_model", "predicted_values"); // TODO:param_req
    }
  }
  fx_done();
}
