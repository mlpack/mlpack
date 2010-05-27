/**
 * @author Hua Ouyang
 *
 * @file svm.h
 *
 * This head file contains functions for performing SVM training and prediction
 * Supported SVM learner type:SVM_C, SVM_R, SVM_DE
 *
 * @see opt_smo.h
 */

#ifndef U_SVM_SVM_H
#define U_SVM_SVM_H

#include "opt_smo.h"

#include "fastlib/fastlib.h"

#include <typeinfo>


#define ID_LINEAR 0
#define ID_GAUSSIAN 1

/**
* Class for Linear Kernel
*/
class SVMLinearKernel {
 public:
  /* Init of kernel parameters */
  ArrayList<double> kpara_; // kernel parameters
  void Init(datanode *node) { //TODO: NULL->node
    kpara_.Init(); 
  }
  /* Kernel name */
  void GetName(String* kname) {
    kname->Copy("linear");
  }
  /* Get an type ID for kernel */
  int GetTypeId() {
    return ID_LINEAR;
  }
  /* Kernel value evaluation */
  double Eval(const double* a, const double* b, index_t n_features) const {
    return la::Dot(n_features, a, b);
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp) {
  }
};

/**
* Class for Gaussian RBF Kernel
*/
class SVMRBFKernel {
 public:
  /* Init of kernel parameters */
  ArrayList<double> kpara_; // kernel parameters
  void Init(datanode *node) { //TODO: NULL->node
    kpara_.Init(2);
    kpara_[0] = fx_param_double_req(NULL, "sigma"); //sigma
    kpara_[1] = -1.0 / (2 * math::Sqr(kpara_[0])); //gamma
  }
  /* Kernel name */
  void GetName(String* kname) {
    kname->Copy("gaussian");
  }
  /* Get an type ID for kernel */
  int GetTypeId() {
    return ID_GAUSSIAN;
  }
  /* Kernel value evaluation */
  double Eval(const double *a, const double *b, index_t n_features) const {
    double distance_squared = la::DistanceSqEuclidean(n_features, a, b);
    return exp(kpara_[1] * distance_squared);
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp) {
    fprintf(fp, "sigma %g\n", kpara_[0]);
    fprintf(fp, "gamma %g\n", kpara_[1]);
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
  int learner_typeid_;
  // Optimization method: smo
  String opt_method_;
  /* array of models for storage of the 2-class(binary) classifiers 
     Need to train num_classes_*(num_classes_-1)/2 binary models */
  struct SVM_MODELS {
    /* bias term in each binary model */
    double bias_;
    /* all coefficients (alpha*y) of the binary dataset, not necessarily thoes of SVs */
    ArrayList<double> coef_;
  };
  ArrayList<SVM_MODELS> models_;

  /* list of labels, double type, but may be converted to integers.
     e.g. [0.0,1.0,2.0] for a 3-class dataset */
  ArrayList<double> train_labels_list_;
  /* array of label indices, after grouping. e.g. [c1[0,5,6,7,10,13,17],c2[1,2,4,8,9],c3[...]]*/
  ArrayList<index_t> train_labels_index_;
  /* counted number of label for each class. e.g. [7,5,8]*/
  ArrayList<index_t> train_labels_ct_;
  /* start positions of each classes in the training label list. e.g. [0,7,12] */
  ArrayList<index_t> train_labels_startpos_;
  
  /* total set of support vectors and their coefficients */
  Matrix sv_;
  Matrix sv_coef_;
  ArrayList<bool> trainset_sv_indicator_;

  /* total number of support vectors */
  index_t total_num_sv_;
  /* support vector list to store the indices (in the training set) of support vectors */
  ArrayList<index_t> sv_index_;
  /* start positions of each class of support vectors, in the support vector list */
  ArrayList<index_t> sv_list_startpos_;
  /* counted number of support vectors for each class */
  ArrayList<index_t> sv_list_ct_;

  /* SVM parameters */
  struct PARAMETERS {
    TKernel kernel_;
    String kernelname_;
    int kerneltypeid_;
    int b_;
    double C_;
    // for SVM_C of unbalanced data
    double Cp_; // C for y==1
    double Cn_; // C for y==-1
    // for SVM_R
    double epsilon_;
    // working set selection scheme, 1 for 1st order expansion; 2 for 2nd order expansion
    double wss_;
    // whether do L1-SVM (1) or L2-SVM (2)
    int hinge_sqhinge_; // L1-SVM uses hinge loss, L2-SVM uses squared hinge loss
    // accuracy for the optimization stopping creterion
    double accuracy_;
    // number of iterations
    index_t n_iter_;
  };
  PARAMETERS param_;
  
  /* number of data samples */
  index_t n_data_; 
  /* number of classes in the training set */
  int num_classes_;
  /* number of binary models to be trained, i.e. num_classes_*(num_classes_-1)/2 */
  int num_models_;
  int num_features_;

 public:
  typedef TKernel Kernel;
  class SMO<Kernel>;

  void Init(int learner_typeid, const Dataset& dataset, datanode *module);
  void InitTrain(int learner_typeid, const Dataset& dataset, datanode *module);

  double Predict(int learner_typeid, const Vector& vector);
  void BatchPredict(int learner_typeid, Dataset& testset, String predictedvalue_filename);
  void LoadModelBatchPredict(int learner_typeid, Dataset& testset, String model_filename, String predictedvalue_filename);

 private:
  void SVM_C_Train_(int learner_typeid, const Dataset& dataset, datanode *module);
  void SVM_R_Train_(int learner_typeid, const Dataset& dataset, datanode *module);
  void SVM_DE_Train_(int learner_typeid, const Dataset& dataset, datanode *module);
  double SVM_C_Predict_(const Vector& vector);
  double SVM_R_Predict_(const Vector& vector);
  double SVM_DE_Predict_(const Vector& vector);

  void SaveModel_(int learner_typeid, String model_filename);
  void LoadModel_(int learner_typeid, String model_filename);
};

/**
* SVM initialization
*
* @param: labeled training set or testing set
* @param: number of classes (different labels) in the data set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::Init(int learner_typeid, const Dataset& dataset, datanode *module){
  learner_typeid_ = learner_typeid;

  opt_method_ = fx_param_str(NULL, "opt", "smo"); // optimization method: default using SMO

  n_data_ = dataset.n_points();
  // # of features == # of row - 1, exclude the last row (for labels)
  num_features_ = dataset.n_features()-1;
  // # of classes of the training set
  num_classes_ = dataset.n_labels();

  train_labels_list_.Init();
  train_labels_index_.Init();
  train_labels_ct_.Init();
  train_labels_startpos_.Init();
  
  if (learner_typeid == 0) { /* for multiclass SVM classificatioin*/
    num_models_ = num_classes_ * (num_classes_-1) / 2;
    sv_list_startpos_.Init(num_classes_);
    sv_list_ct_.Init(num_classes_);
  }
  else { /* for other SVM learners */
    num_classes_ = 2; // dummy #, only meaningful in SaveModel and LoadModel

    num_models_ = 1;
    sv_list_startpos_.Init();
    sv_list_ct_.Init();
  }

  models_.Init();
  sv_index_.Init();
  total_num_sv_ = 0;

  /* bool indicators FOR THE TRAINING SET: is/isn't a support vector */
  /* Note: it has the same index as the training !!! */
  trainset_sv_indicator_.Init(n_data_);
  for (index_t i=0; i<n_data_; i++)
    trainset_sv_indicator_[i] = false;

  param_.kernel_.Init(fx_submodule(module, "kernel"));
  param_.kernel_.GetName(&param_.kernelname_);
  param_.kerneltypeid_ = param_.kernel_.GetTypeId();
  // working set selection scheme. default: 1st order expansion
  param_.wss_ = fx_param_int(NULL, "wss", 1);
  // whether do L1-SVM(1) or L2-SVM (2)
  param_.hinge_sqhinge_ = fx_param_int(NULL, "hinge", 1); // default do L1-SVM

  // accuracy for optimization
  param_.accuracy_ = fx_param_double(NULL, "accuracy", 1e-4);
  // number of iterations
  param_.n_iter_ = fx_param_int(NULL, "n_iter", 100000000);

  // tradeoff parameter for C-SV
  param_.C_ = fx_param_double(NULL, "c", 10.0);
  param_.Cp_ = fx_param_double(NULL, "c_p", param_.C_);
  param_.Cn_ = fx_param_double(NULL, "c_n", param_.C_);

  if (learner_typeid == 1) { // for SVM_R only
    // the "epsilon", default: 0.1
    param_.epsilon_ = fx_param_double(NULL, "epsilon", 0.1);
  }
  else if (learner_typeid == 2) { // SVM_DE
  }
}

/**
* Initialization(data dependent) and training for SVM learners
*
* @param: typeid of the learner
* @param: number of classes (different labels) in the training set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::InitTrain(int learner_typeid, const Dataset& dataset, datanode *module) {
  Init(learner_typeid, dataset, module);
  
  if (learner_typeid == 0) { // Multiclass SVM Clssification
    SVM_C_Train_(learner_typeid, dataset, module);
  }
  else if (learner_typeid == 1) { // SVM Regression
    SVM_R_Train_(learner_typeid, dataset, module);
  }
  else if (learner_typeid == 2) { // One Class SVM
    SVM_DE_Train_(learner_typeid, dataset, module);
  }
  
  /* Save models to file "svm_model" */
  SaveModel_(learner_typeid, "svm_model"); // TODO: param_req, and for CV mode
  // TODO: calculate training error
}


/**
* Training for Multiclass SVM Clssification, using One-vs-One method
*
* @param: type id of the learner
* @param: training set
* @param: number of classes of the training set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::SVM_C_Train_(int learner_typeid, const Dataset& dataset, datanode *module) {
  num_classes_ = dataset.n_labels();
  /* Group labels, split the training dataset for training bi-class SVM classifiers */
  dataset.GetLabels(train_labels_list_, train_labels_index_, train_labels_ct_, train_labels_startpos_);

  /* Train num_classes*(num_classes-1)/2 binary class(labels:-1, 1) models */
  index_t ct = 0;
  index_t i, j;
  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      models_.PushBack();
      /* Construct dataset consists of two classes i and j (reassign labels 1 and -1) */
      // TODO: avoid these ugly and time-consuming memory allocation
      Dataset dataset_bi;
      dataset_bi.InitBlank();
      dataset_bi.info().Init();
      dataset_bi.matrix().Init(num_features_+1, train_labels_ct_[i]+train_labels_ct_[j]);
      ArrayList<index_t> dataset_bi_index;
      dataset_bi_index.Init(train_labels_ct_[i]+train_labels_ct_[j]);
      for (index_t m = 0; m < train_labels_ct_[i]; m++) {
	Vector source, dest;
	dataset_bi.matrix().MakeColumnVector(m, &dest);
	dataset.matrix().MakeColumnVector(train_labels_index_[train_labels_startpos_[i]+m], &source);
	dest.CopyValues(source);
	/* last row for labels 1 */
	dataset_bi.matrix().set(num_features_, m, 1);
	dataset_bi_index[m] = train_labels_index_[train_labels_startpos_[i]+m];
      }
      for (index_t n = 0; n < train_labels_ct_[j]; n++) {
	Vector source, dest;
	dataset_bi.matrix().MakeColumnVector(n+train_labels_ct_[i], &dest);
	dataset.matrix().MakeColumnVector(train_labels_index_[train_labels_startpos_[j]+n], &source);
	dest.CopyValues(source);
	/* last row for labels -1 */
	dataset_bi.matrix().set(num_features_, n+train_labels_ct_[i], -1);
	dataset_bi_index[n+train_labels_ct_[i]] = train_labels_index_[train_labels_startpos_[j]+n];
      }

      if (opt_method_== "smo") {
	/* Initialize SMO parameters */
	ArrayList<double> param_feed_db;
	param_feed_db.Init();
	param_feed_db.PushBack() = param_.Cp_;
	param_feed_db.PushBack() = param_.Cn_;
	param_feed_db.PushBack() = param_.hinge_sqhinge_;
	param_feed_db.PushBack() = param_.wss_;
	param_feed_db.PushBack() = param_.n_iter_;
	param_feed_db.PushBack() = param_.accuracy_;
	SMO<Kernel> smo;
	smo.InitPara(learner_typeid, param_feed_db);

	/* Initialize kernel */
	smo.kernel().Init(fx_submodule(module, "kernel"));

	/* 2-classes SVM training using SMO */
	fx_timer_start(NULL, "train_smo");
	smo.Train(learner_typeid, &dataset_bi);
	fx_timer_stop(NULL, "train_smo");

	/* Get the trained bi-class model */
	models_[ct].coef_.Init(); // alpha*y
	models_[ct].bias_ = smo.Bias(); // bias
	smo.GetSV(dataset_bi_index, models_[ct].coef_, trainset_sv_indicator_); // get support vectors
      }
      else {
	fprintf(stderr, "ERROR!!! Unknown optimization method!\n");
      }

      ct++;
    }
  }
  
  /* Get total set of SVs from all the binary models */
  index_t k;
  sv_list_startpos_[0] = 0;

  for (i = 0; i < num_classes_; i++) {
    ct = 0;
    for (j = 0; j < train_labels_ct_[i]; j++) {
      if (trainset_sv_indicator_[ train_labels_index_[train_labels_startpos_[i]+j] ]) {
	sv_index_.PushBack() = train_labels_index_[train_labels_startpos_[i]+j];
	total_num_sv_++;
	ct++;
      }
    }
    sv_list_ct_[i] = ct;
    if (i >= 1)
      sv_list_startpos_[i] = sv_list_startpos_[i-1] + sv_list_ct_[i-1];    
  }
  sv_.Init(num_features_, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    Vector source, dest;
    sv_.MakeColumnVector(i, &dest);
    /* last row of dataset is for labels */
    dataset.matrix().MakeColumnSubvector(sv_index_[i], 0, num_features_, &source); 
    dest.CopyValues(source);
  }
  /* Get the matrix sv_coef_ which stores the coefficients of all sets of SVs */
  /* i.e. models_[x].coef_ -> sv_coef_ */
  index_t ct_model = 0;
  index_t p;
  sv_coef_.Init(num_classes_-1, total_num_sv_);
  sv_coef_.SetZero();
  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      p = sv_list_startpos_[i];
      for (k = 0; k < train_labels_ct_[i]; k++) {
	if (trainset_sv_indicator_[ train_labels_index_[train_labels_startpos_[i]+k] ]) {
	  sv_coef_.set(j-1, p++, models_[ct_model].coef_[k]);
	}
      }
      p = sv_list_startpos_[j];
      for (k = 0; k < train_labels_ct_[j]; k++) {
	if (trainset_sv_indicator_[ train_labels_index_[train_labels_startpos_[j]+k] ]) {
	  sv_coef_.set(i, p++, models_[ct_model].coef_[train_labels_ct_[i] + k]);
	}
      }
      ct_model++;
    }
  }
}

/**
* Training for SVM Regression
*
* @param: type id of the learner
* @param: training set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::SVM_R_Train_(int learner_typeid, const Dataset& dataset, datanode *module) {
  index_t i;
  ArrayList<index_t> dataset_index;
  dataset_index.Init(n_data_);
  for (i=0; i<n_data_; i++)
    dataset_index[i] = i;

  models_.PushBack();

  if (opt_method_== "smo") {
    /* Initialize SMO parameters */
    ArrayList<double> param_feed_db;
    param_feed_db.Init();
    param_feed_db.PushBack() = param_.C_;
    param_feed_db.PushBack() = param_.epsilon_;
    param_feed_db.PushBack() = param_.wss_;
    param_feed_db.PushBack() = param_.n_iter_;
    param_feed_db.PushBack() = param_.accuracy_;
    SMO<Kernel> smo;
    smo.InitPara(learner_typeid, param_feed_db);
    
    /* Initialize kernel */
    smo.kernel().Init(fx_submodule(module, "kernel"));

    /* SVM_R Training using SMO*/
    smo.Train(learner_typeid, &dataset);
    
    /* Get the trained model */
    models_[0].bias_ = smo.Bias(); // bias
    models_[0].coef_.Init(); // alpha*y
    smo.GetSV(dataset_index, models_[0].coef_, trainset_sv_indicator_); // get support vectors
  }
  else {
    fprintf(stderr, "ERROR!!! Unknown optimization method!");
  }

  /* Get index list of support vectors */
  for (i = 0; i < n_data_; i++) {
    if (trainset_sv_indicator_[i]) {
      sv_index_.PushBack() = i;
      total_num_sv_++;
    }
  }

  /* Get support vecotors and coefficients */
  sv_.Init(num_features_, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    Vector source, dest;
    sv_.MakeColumnVector(i, &dest);
    /* last row of dataset is for labels */
    dataset.matrix().MakeColumnSubvector(sv_index_[i], 0, num_features_, &source); 
    dest.CopyValues(source);
  }
  sv_coef_.Init(1, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    sv_coef_.set(0, i, models_[0].coef_[i]);
  }
  
}

/**
* Training for One Class SVM
*
* @param: type id of the learner
* @param: training set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::SVM_DE_Train_(int learner_typeid, const Dataset& dataset, datanode *module) {
  // TODO
}


/**
* SVM prediction for one testing vector
*
* @param: type id of the learner
* @param: testing vector
*
* @return: predited value
*/
template<typename TKernel>
double SVM<TKernel>::Predict(int learner_typeid, const Vector& datum) {
  double predicted_value = INFINITY;
  if (learner_typeid == 0) { // Multiclass SVM Clssification
    predicted_value = SVM_C_Predict_(datum);
  }
  else if (learner_typeid == 1) { // SVM Regression
    predicted_value = SVM_R_Predict_(datum);
  }
  else if (learner_typeid == 2) { // One class SVM
    predicted_value = SVM_DE_Predict_(datum);
  }
  return predicted_value;
}

/**
* Multiclass SVM classification for one testing vector
*
* @param: testing vector
*
* @return: a label (double-type-integer, e.g. 1.0, 2.0, 3.0)
*/
template<typename TKernel>
double SVM<TKernel>::SVM_C_Predict_(const Vector& datum) {
  index_t i, j, k;
  ArrayList<double> keval;
  keval.Init(total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    keval[i] = param_.kernel_.Eval(datum.ptr(), sv_.GetColumnPtr(i), num_features_);
  }
  ArrayList<double> values;
  values.Init(num_models_);
  index_t ct = 0;
  double sum = 0.0;
  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      if (opt_method_== "smo") {
	sum = 0.0;
	for(k = 0; k < sv_list_ct_[i]; k++) {
	  sum += sv_coef_.get(j-1, sv_list_startpos_[i]+k) * keval[sv_list_startpos_[i]+k];
	}
	for(k = 0; k < sv_list_ct_[j]; k++) {
	  sum += sv_coef_.get(i, sv_list_startpos_[j]+k) * keval[sv_list_startpos_[j]+k];
	}
	sum += models_[ct].bias_;
      }
      values[ct] = sum;
      ct++;
    }
  }

  ArrayList<index_t> vote;
  vote.Init(num_classes_);
  for (i = 0; i < num_classes_; i++) {
    vote[i] = 0;
  }
  ct = 0;
  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      if(values[ct] > 0.0) { // label 1 in bi-classifiers (for i=...)
	vote[i] = vote[i] + 1;
      }
      else {  // label -1 in bi-classifiers (for j=...)
	vote[j] = vote[j] + 1;
      }
      ct++;
    }
  }
  index_t vote_max_idx = 0;
  for (i = 1; i < num_classes_; i++) {
    if (vote[i] >= vote[vote_max_idx]) {
      vote_max_idx = i;
    }
  }
  return train_labels_list_[vote_max_idx];
}

/**
* SVM Regression Prediction for one testing vector
*
* @param: testing vector
*
* @return: predicted regression value
*/
template<typename TKernel>
double SVM<TKernel>::SVM_R_Predict_(const Vector& datum) {
  index_t i;
  double sum = 0.0;
  if (opt_method_== "smo") {
    for (i = 0; i < total_num_sv_; i++) {
      sum += sv_coef_.get(0, i) * param_.kernel_.Eval(datum.ptr(), sv_.GetColumnPtr(i), num_features_);
    }
  }
  sum += models_[0].bias_;
  return sum;
}

/**
* One class SVM Prediction for one testing vector
*
* @param: testing vector
*
* @return: estimated value
*/
template<typename TKernel>
double SVM<TKernel>::SVM_DE_Predict_(const Vector& datum) {
  // TODO
  return 0.0;
}



/**
* Batch classification for multiple testing vectors. No need to load model file, 
* since models are already in RAM.
*
* Note: for test set, if no true test labels provided, just put some dummy labels 
* (e.g. all -1) in the last row of testset
*
* @param: type id of the learner
* @param: testing set
* @param: file name of the testing data
*/
template<typename TKernel>
void SVM<TKernel>::BatchPredict(int learner_typeid, Dataset& testset, String predictedvalue_filename) {
  FILE *fp = fopen(predictedvalue_filename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Cannot save predicted values to file!");
    return;
  }
  index_t err_ct = 0;
  num_features_ = testset.n_features()-1;
  for (index_t i = 0; i < testset.n_points(); i++) {
    Vector testvec;
    testset.matrix().MakeColumnSubvector(i, 0, num_features_, &testvec);
    double predictedvalue = Predict(learner_typeid, testvec);
    if (predictedvalue != testset.matrix().get(num_features_, i))
      err_ct++;
    /* save predicted values to file*/
    fprintf(fp, "%f\n", predictedvalue);
  }
  fclose(fp);
  /* calculate testing error */
  printf( "\n*** %d out of %d misclassified ***\n", err_ct, testset.n_points() );
  printf( "*** Testing error is %f, accuracy is %f. ***\n", double(err_ct)/double(testset.n_points()), 1- double(err_ct)/double(testset.n_points()) );
  //fprintf( stderr, "*** Results are save in \"%s\" ***\n\n", predictedvalue_filename.c_str());
}

/**
* Load models from a file, and perform offline batch classification for multiple testing vectors
*
* @param: type id of the learner
* @param: testing set
* @param: name of the model file
* @param: name of the file to store classified labels
*/
template<typename TKernel>
void SVM<TKernel>::LoadModelBatchPredict(int learner_typeid, Dataset& testset, String model_filename, String predictedvalue_filename) {
  LoadModel_(learner_typeid, model_filename);
  BatchPredict(learner_typeid, testset, predictedvalue_filename);
}


/**
* Save SVM model to a text file
*
* @param: type id of the learner
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void SVM<TKernel>::SaveModel_(int learner_typeid, String model_filename) {
  FILE *fp = fopen(model_filename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Cannot save trained model to file!");
    return;
  }
  index_t i, j;

  if (learner_typeid == 0) { // for SVM_C
    fprintf(fp, "svm_type SVM_C\n");
    fprintf(fp, "total_num_sv %d\n", total_num_sv_);
    fprintf(fp, "num_classes %d\n", num_classes_);
    // save labels
    fprintf(fp, "labels ");
    for (i = 0; i < num_classes_; i++) 
      fprintf(fp, "%f ", train_labels_list_[i]);
    fprintf(fp, "\n");
    // save support vector info
    fprintf(fp, "sv_list_startpos ");
    for (i =0; i < num_classes_; i++)
      fprintf(fp, "%d ", sv_list_startpos_[i]);
    fprintf(fp, "\n");
    fprintf(fp, "sv_list_ct ");
    for (i =0; i < num_classes_; i++)
      fprintf(fp, "%d ", sv_list_ct_[i]);
    fprintf(fp, "\n");
  }
  else if (learner_typeid == 1) { // for SVM_R
    fprintf(fp, "svm_type SVM_R\n");
    fprintf(fp, "total_num_sv %d\n", total_num_sv_);
    fprintf(fp, "sv_index ");
    for (i = 0; i < total_num_sv_; i++)
      fprintf(fp, "%d ", sv_index_[i]);
    fprintf(fp, "\n");
  }
  else if (learner_typeid == 2) { // for SVM_DE
    fprintf(fp, "svm_type SVM_DE\n");
    fprintf(fp, "total_num_sv %d\n", total_num_sv_);
    fprintf(fp, "sv_index ");
    for (i = 0; i < total_num_sv_; i++)
      fprintf(fp, "%d ", sv_index_[i]);
    fprintf(fp, "\n");
  }

  // save kernel parameters
  fprintf(fp, "kernel_name %s\n", param_.kernelname_.c_str());
  fprintf(fp, "kernel_typeid %d\n", param_.kerneltypeid_);
  param_.kernel_.SaveParam(fp);

  // save models: bias, coefficients and support vectors
  fprintf(fp, "bias ");
  for (i = 0; i < num_models_; i++)
    fprintf(fp, "%.16g ", models_[i].bias_);
  fprintf(fp, "\n");
  
  fprintf(fp, "SV_coefs\n");
  for (i = 0; i < total_num_sv_; i++) {
    for (j = 0; j < num_classes_-1; j++) {
      fprintf(fp, "%.16g ", sv_coef_.get(j,i));
    }
    fprintf(fp, "\n");
  }

  fprintf(fp, "SVs\n");
  for (i = 0; i < total_num_sv_; i++) {
    for (j = 0; j < num_features_; j++) { // n_rows-1
      fprintf(fp, "%.8g ", sv_.get(j,i));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/**
* Load SVM model file
*
* @param: type id of the learner
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void SVM<TKernel>::LoadModel_(int learner_typeid, String model_filename) {
  if (learner_typeid == 0) {// SVM_C
    train_labels_list_.Renew();
    train_labels_list_.Init(num_classes_); // get labels list from the model file
  }

  /* load model file */
  FILE *fp = fopen(model_filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open SVM model file!");
    return;
  }
  char cmd[80]; 
  int i, j; int temp_d; double temp_f;
  for (i = 0; i < num_models_; i++) {
	models_.PushBack();
	models_[i].coef_.Init();
  }
  while (1) {
    fscanf(fp,"%80s",cmd);
    if(strcmp(cmd,"svm_type")==0) {
      fscanf(fp,"%80s", cmd);
      if (strcmp(cmd,"SVM_C")==0)
	learner_typeid_ = 0;
      else if (strcmp(cmd,"SVM_R")==0)
	learner_typeid_ = 1;
      else if (strcmp(cmd,"SVM_DE")==0)
	learner_typeid_ = 2;
    }
    else if (strcmp(cmd, "total_num_sv")==0) {
      fscanf(fp,"%d",&total_num_sv_);
    }
    // for SVM_C
    else if (strcmp(cmd, "num_classes")==0) {
      fscanf(fp,"%d",&num_classes_);
    }
    else if (strcmp(cmd, "labels")==0) {
      for (i=0; i<num_classes_; i++) {
	fscanf(fp,"%lf",&temp_f);
	train_labels_list_[i] = temp_f;
      }
    }
    else if (strcmp(cmd, "sv_list_startpos")==0) {
      for ( i= 0; i < num_classes_; i++) {
	fscanf(fp,"%d",&temp_d);
	sv_list_startpos_[i]= temp_d;
      }
    }
    else if (strcmp(cmd, "sv_list_ct")==0) {
      for ( i= 0; i < num_classes_; i++) {
	fscanf(fp,"%d",&temp_d); 
	sv_list_ct_[i]= temp_d;
      }
    }
    // for SVM_R
    else if (strcmp(cmd, "sv_index")==0) {
      for ( i= 0; i < total_num_sv_; i++) {
	fscanf(fp,"%d",&temp_d); 
	sv_index_.PushBack() = temp_d;
      }
    }
    // load kernel info
    else if (strcmp(cmd, "kernel_name")==0) {
      fscanf(fp,"%80s",param_.kernelname_.c_str());
    }
    else if (strcmp(cmd, "kernel_typeid")==0) {
      fscanf(fp,"%d",&param_.kerneltypeid_);
    }
    else if (strcmp(cmd, "sigma")==0) {
      fscanf(fp,"%lf",&param_.kernel_.kpara_[0]); /* for gaussian kernels only */
    }
    else if (strcmp(cmd, "gamma")==0) {
      fscanf(fp,"%lf",&param_.kernel_.kpara_[1]); /* for gaussian kernels only */
    }
    // load bias
    else if (strcmp(cmd, "bias")==0) {
      for ( i= 0; i < num_models_; i++) {
	fscanf(fp,"%lf",&temp_f); 
	models_[i].bias_= temp_f;
      }
      break;
    }
  }

  // load coefficients and support vectors
  sv_coef_.Init(num_classes_-1, total_num_sv_);
  sv_coef_.SetZero();
  sv_.Init(num_features_, total_num_sv_);
  while (1) {
    fscanf(fp,"%80s",cmd);
    if (strcmp(cmd, "SV_coefs")==0) {
      for (i = 0; i < total_num_sv_; i++) {
	for (j = 0; j < num_classes_-1; j++) {
	  fscanf(fp,"%lf",&temp_f);
	  sv_coef_.set(j, i, temp_f);
	}
      }
    }
    else if (strcmp(cmd, "SVs")==0) {
      for (i = 0; i < total_num_sv_; i++) {
	for (j = 0; j < num_features_; j++) {
	  fscanf(fp,"%lf",&temp_f);
	  sv_.set(j, i, temp_f);
	}
      }
      break;
    }
  }
  fclose(fp);
}

#endif
