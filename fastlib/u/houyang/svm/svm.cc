#include "svm.h"

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

  //cov.PrintDebug("cov");
  la::EigenvectorsInit(cov, &d, &u);
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

  //dataset->matrix().PrintDebug("m");

  if (fx_param_bool(NULL, "save", 0)) {
    fx_default_param(NULL, "kfold/save", "1");
    dataset->WriteCsv("normalized.csv");
  }
}

void CreateArtificialDataset(Dataset* dataset){
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
    m.set(2, i, 0);
    
    x = (rand() * range / RAND_MAX) + offset;
    y = margin / 2 + (rand() * var / RAND_MAX);
    m.set(0, i+1, x);
    m.set(1, i+1, x*slope - y + intercept);
    m.set(2, i+1, 1);
    
    x = (rand() * range / RAND_MAX) + offset;
    y = margin / 2 + (rand() * var / RAND_MAX);
    m.set(0, i+2, x);
    m.set(1, i+2, 1e2*x*slope - y + intercept);
    m.set(2, i+2, 2);
  }
  data::Save("m.csv", m);
  dataset->OwnMatrix(&m);
}

int LoadData(Dataset* dataset, String datafilename){
if (fx_param_exists(NULL, datafilename)) {
      // if a data file is specified, use it.
      if (!SUCCEEDED(dataset->InitFromFile(fx_param_str_req(NULL, datafilename)))) {
	fprintf(stderr, "Couldn't open the data file.\n");
	return 0;
      }
    } else {
      // otherwise, create an artificial dataset and save it to "m.csv"
      CreateArtificialDataset(dataset);
    }
  
    if (fx_param_bool(NULL, "normalize", 1)) {
      fprintf(stderr, "Normalizing\n");
      DoSvmNormalize(dataset);
    } else {
      fprintf(stderr, "Skipping normalize\n");
    }
    return 1;
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  //srand(time(NULL));

  String mode = fx_param_str_req(NULL, "mode");
  String kernel = fx_param_str_req(NULL, "kernel");
  
  // TODO: more kernels to be supported

  // Cross Validation Mode, need cross validation data
  if(mode == "cv") { 
    fprintf(stderr, "SVM Cross Validation... \n");
    
    // Load cross validation data
    Dataset cvset;
    if (LoadData(&cvset, "cv_data") == 0)
    return 1;
    
    if (kernel == "linear") {
      SimpleCrossValidator< SVM<SVMLinearKernel> > cross_validator; 
      // Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds, need k_cv>1
      cross_validator.Init(&cvset,cvset.n_labels(),fx_param_int_req(NULL,"k_cv"), fx_root, "svm");
      // k_cv folds cross validation; (true): do training set permutation
      cross_validator.Run(true);
      cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
    else if (kernel == "gaussian") {
      SimpleCrossValidator< SVM<SVMRBFKernel> > cross_validator; 
      // Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds
      cross_validator.Init(&cvset,cvset.n_labels(),fx_param_int_req(NULL,"k_cv"), fx_root, "svm");
      // k_cv folds cross validation; (true): do training set permutation
      cross_validator.Run(true);
      cross_validator.confusion_matrix().PrintDebug("confusion matrix");
    }
  }
  // Training Mode, need training data | Training + Testing(online) Mode, need training data + testing data
  else if (mode=="train" || mode=="train_test"){
    fprintf(stderr, "SVM Training... \n");

    // Load training data
    Dataset trainset;
    if (LoadData(&trainset, "train_data") == 0) // TODO:param_req
      return 1;
    
    // Begin SVM Training | Training and Testing
    datanode *svm_module = fx_submodule(fx_root, NULL, "svm");

    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.InitTrain(trainset, trainset.n_labels(), svm_module);
      if (mode=="train_test"){ // training and testing, thus no need to load model from file
	fprintf(stderr, "SVM Classifying... \n");
	// Load testing data
	Dataset testset;
	if (LoadData(&testset, "test_data") == 0) // TODO:param_req
	  return 1;
	svm.BatchClassify(&testset, "test_labels");
      }
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.InitTrain(trainset, trainset.n_labels(), svm_module);
      if (mode=="train_test"){ // training and testing, thus no need to load model from file
	fprintf(stderr, "SVM Classifying... \n");
	// Load testing data
	Dataset testset;
	if (LoadData(&testset, "test_data") == 0) // TODO:param_req
	  return 1;
	svm.BatchClassify(&testset, "test_labels");
      }
    }
  }
  // Testing(offline) Mode, need loading model file and testing data
  else if (mode=="test") {
    fprintf(stderr, "SVM Classifying... \n");

    // Load testing data
    Dataset testset;
    if (LoadData(&testset, "test_data") == 0) // TODO:param_req
      return 1;

    // Begin Classification
    datanode *svm_module = fx_submodule(fx_root, NULL, "svm");

    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.Init(testset, testset.n_labels(), svm_module);
      svm.LoadModelBatchClassify(&testset, "svm_model", "test_labels"); // TODO:param_req
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.Init(testset, testset.n_labels(), svm_module);
      svm.LoadModelBatchClassify(&testset, "svm_model", "test_labels"); // TODO:param_req
    }
  }
  fx_done();
}

