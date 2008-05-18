/**
 * @author Hua Ouyang
 *
 * @file rvm.h
 *
 * This head file contains functions for Relevance Vector Machine (RVM):
 * multiclass RVM classification and RVM regression
 *
 * @see sbl_rvm.h
 *
 */

#ifndef U_RVM_RVM_H
#define U_RVM_RVM_H

#include "sbl_est.h"

#include "fastlib/fastlib.h"

#include <typeinfo>

#define ID_LINEAR 0
#define ID_GAUSSIAN 1

/**
* Class for Linear Kernel
*/
class RVMLinearKernel {
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
  double Eval(const Vector& a, const Vector& b) const {
    return la::Dot(a, b);
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp) {
  }
};

/**
* Class for Gaussian RBF Kernel
*/
class RVMRBFKernel {
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
  double Eval(const Vector& a, const Vector& b) const {
    double distance_squared = la::DistanceSqEuclidean(a, b);
    return exp(kpara_[1] * distance_squared);
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp) {
    fprintf(fp, "sigma %g\n", kpara_[0]);
    fprintf(fp, "gamma %g\n", kpara_[1]);
  }
};

/**
* Class for RVM
*/
template<typename TKernel>
class RVM {
 private:
  /** 
   * Type id of the learner: 
   *  0:RVM Classification (rvc);
   *  1:RVM Regression (rvr)
   * Developers may add more learner types if necessary
   */
  int learner_typeid_;
  
  
  /* total set of relevance vectors */
  Matrix rv_;
  /* all coefficients of the binary dataset, not necessarily thoes of RVs */
  ArrayList<double> weights_;  


  /* total number of relevance vectors */
  index_t total_num_rv_;
  /* relevance vector list to store the indices (in the training set) of relevance vectors */
  ArrayList<index_t> rv_index_;

  /* RVM parameters, same for every binary model */
  struct RVM_PARAMETERS {
    TKernel kernel_;
    String kernelname_;
    int kerneltypeid_;
    double initalpha_;
    double beta_;
    index_t max_iter_;
  };
  RVM_PARAMETERS param_;

  Vector alpha_v_;
  
  /* number of data points in the training set */
  int num_data_;
  int num_features_;

  // FOR RVM CLASSIFICATION ONLY
  /* number of classes */
  int num_classes_;
  /* list of labels, double type, but may be converted to integers.
     e.g. [0.0,1.0,2.0] for a 3-class dataset */
  ArrayList<double> train_labels_list_;
  /* array of label indices, after grouping. e.g. [c1[0,5,6,7,10,13,17],c2[1,2,4,8,9],c3[...]]*/
  ArrayList<index_t> train_labels_index_;
  /* counted number of label for each class. e.g. [7,5,8]*/
  ArrayList<index_t> train_labels_ct_;
  /* start positions of each classes in the training label list. e.g. [0,7,12] */
  ArrayList<index_t> train_labels_startpos_;

 public:
  typedef TKernel Kernel;

  void Init(int learner_typeid, const Dataset& dataset, datanode *module);
  void InitTrain(int learner_typeid, const Dataset& dataset, datanode *module);
  double Predict(int learner_typeid, const Vector& vector);
  void BatchPredict(int learner_typeid, Dataset& testset, String testlabelfilename);
  void LoadModelBatchPredict(int learner_typeid, Dataset& testset, String modelfilename, String testlabelfilename);
 private:
  void SaveModel_(int learner_typeid, String modelfilename);
  void LoadModel_(int learner_typeid, String modelfilename);

};

/**
* RVM initialization
*
* @param: labeled training set or testing set
* @param: number of classes (different labels) in the data set
* @param: module name
*/
template<typename TKernel>
void RVM<TKernel>::Init(int learner_typeid, const Dataset& dataset, datanode *module){
  learner_typeid_ = learner_typeid;

  total_num_rv_ = 0;

  weights_.Init();
  rv_index_.Init();

  param_.kernel_.Init(fx_submodule(module, "kernel"));
  param_.kernel_.GetName(&param_.kernelname_);
  param_.kerneltypeid_ = param_.kernel_.GetTypeId();

  // # of data samples
  num_data_ = dataset.n_points();
  /* # of features == # of rows in data matrix, since last row in dataset is for discrete labels or continuous values */
  num_features_ = dataset.n_features()-1;
  // # of classes of the training set
  num_classes_ = dataset.n_labels();

  // initialize alpha_, same for rvc and rvr
  param_.initalpha_ =  1.0/math::Sqr(num_data_);
  
  // initilialize beta_ accroding to different learner types
  if (learner_typeid_ == 0 ) { // RVM Classification
    param_.beta_ = 0;
  }
  else if (learner_typeid_ == 1) { // RVM Regression
    Vector values; // the vector that stores the values for data points
    values.Init(num_data_);
    for(index_t i=0; i<num_data_; i++)
      values[i] = dataset.get(dataset.n_features()-1, i);
    param_.beta_ = 1 / pow(math::Std(values)/10.0, 2);
  }
  else {
    fprintf(stderr, "Unknown learner name of RVM! Program stops!\n");
    return;
  }

  // init maximum number of iterations
  param_.max_iter_ = fx_param_int(NULL, "max_iter", 500); // defult number of iterations: 500
  
  // init the Alpha Vector form the inital alpha value
  alpha_v_.Init(num_data_ + 1); // +1:consider bias
  alpha_v_.SetAll(param_.initalpha_);

  if (learner_typeid == 0) {
    /* Group labels, split the training dataset for training RVM classifier */
    dataset.GetLabels(train_labels_list_, train_labels_index_, train_labels_ct_, train_labels_startpos_);
  }
  else {
    train_labels_list_.Init();
    train_labels_index_.Init();
    train_labels_ct_.Init();
    train_labels_startpos_.Init();
  }
}

/**
* Initialization(data dependent) and training for Multiclass RVM Classifier
*
* @param: labeled training set
* @param: number of classes (different labels) in the training set
* @param: module name
*/
template<typename TKernel>
void RVM<TKernel>::InitTrain(int learner_typeid, const Dataset& dataset, datanode *module) {
  index_t i;
  // Initialize parameters
  Init(learner_typeid, dataset, module);

  /* bool indicators FOR THE TRAINING SET: is/isn't a relevance vector */
  /* Note: it has the same index as the training !!! */
  ArrayList<bool> trainset_rv_indicator;
  trainset_rv_indicator.Init(dataset.n_points());
  for (index_t i=0; i<dataset.n_points(); i++)
    trainset_rv_indicator[i] = false;

  SBL_EST<Kernel> sbl_est;
  // Initialize parameters alpha_v, beta_, max_itr_ for sbl_est
  //sbl_est.Init(param_.beta_, param_.max_iter_);
  
  // Initialize kernels for sbl_est
  sbl_est.kernel().Init(fx_submodule(module, "kernel"));

  /* Training for Relevance Vector Classification and Regression */
  sbl_est.Train(learner_typeid_, &dataset, alpha_v_, param_.beta_, param_.max_iter_, rv_index_, weights_);
  total_num_rv_ = rv_index_.size();

  fprintf(stderr, "total_num_rv:%d\n", total_num_rv_);

  for (i = 0; i < total_num_rv_; i++) {
    trainset_rv_indicator[rv_index_[i]] = true;
  }

  // get relevance vectors
  rv_.Init(num_features_, total_num_rv_);
  for (i = 0; i < total_num_rv_; i++) {
    Vector source, dest;
    rv_.MakeColumnVector(i, &dest);
    /* last row of dataset is for values/labels */
    dataset.matrix().MakeColumnSubvector(rv_index_[i], 0, num_features_, &source); 
    dest.CopyValues(source);
  }

  /* Save models to file "rvm_model" */
  SaveModel_(learner_typeid, "rvm_model"); // TODO: param_req, and for CV mode
  // TODO: calculate training error
}


/**
* RVM Prediction for one testing vector
*
* @param: testing vector
*
* @return: a real value for regression; a label (double-type-integer, e.g. 1.0, 2.0, 3.0) for classification
*/
template<typename TKernel>
double RVM<TKernel>::Predict(int learner_typeid, const Vector& datum) {
  index_t i, j;
  double value_predict = 0.0;

  // for multiclass classification, output: predicted class label
  if (learner_typeid == 0) {
    double k_eval;
    Vector values_clsf;
    values_clsf.Init(num_classes_);
    Vector relevance_vector_i;
    index_t i_rv = 0;
    for (i=0; i < num_classes_; i++) {
      for (j=0; j < train_labels_ct_[i]; j++ ) {
	if ( alpha_v_[ train_labels_startpos_[i]+j ] != 0 ) {	  
	  rv_.MakeColumnVector(i_rv, &relevance_vector_i);
	  k_eval = param_.kernel_.Eval(datum, relevance_vector_i);
	  values_clsf[i] =  values_clsf[i] + weights_[i] * k_eval;
	  i_rv++;
	}
      }
    }
    // classification, assign the class lable with the largest predicted value
    double values_clsf_pre = values_clsf[0];
    for (i=1; i<num_classes_; i++) {
      if (values_clsf[i] > values_clsf_pre) {
	value_predict = double(i);
	values_clsf_pre = values_clsf[i];
      }
    }
  }
  // for regression, output: predicted regression value
  else if (learner_typeid == 1) {
    ArrayList<double> k_evals;
    k_evals.Init(total_num_rv_);
    
    for (i=0; i<total_num_rv_; i++) {
      Vector relevance_vector_i;
      rv_.MakeColumnVector(i, &relevance_vector_i);
      k_evals[i] = param_.kernel_.Eval(datum, relevance_vector_i);
    }

    for (i = 0; i < total_num_rv_; i++) {
      value_predict += weights_[i] * k_evals[i]; // Dot product
    }
  }
  
  return value_predict;
}

/**
* Online batch classification for multiple testing vectors. No need to load model file, 
* since models are already in RAM.
*
* Note: for test set, if no true test labels provided, just put some dummy labels 
* (e.g. all -1) in the last row of testset
*
* @param: testing set
* @param: file name of the testing data
*/
template<typename TKernel>
void RVM<TKernel>::BatchPredict(int learner_typeid, Dataset& testset, String testresultfilename) {
  FILE *fp = fopen(testresultfilename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Cannot save test results to file!");
    return;
  }
  //index_t err_ct = 0;
  num_features_ = testset.n_features()-1;                
  for (index_t i = 0; i < testset.n_points(); i++) {
    Vector testvec;
    testset.matrix().MakeColumnSubvector(i, 0, num_features_, &testvec);
    double testresult = Predict(learner_typeid, testvec);
    //if (testlabel != testset.matrix().get(num_features_, i))
    //  err_ct++;
    /* save classified labels to file*/
    fprintf(fp, "%f\n", testresult);
  }
  fclose(fp);
  /* calculate testing error */
  //fprintf( stderr, "\n*** %d out of %d misclassified ***\n", err_ct, testset.n_points() );
  //fprintf( stderr, "*** Testing error is %f ***\n", double(err_ct)/double(testset.n_points()) );
  fprintf( stderr, "*** Results are save in \"%s\" ***\n\n", testresultfilename.c_str());
}

/**
* Load models from a file, and perform offline batch classification for multiple testing vectors
*
* @param: testing set
* @param: name of the model file
* @param: name of the file to store classified results (labels, or doubles)
*/
template<typename TKernel>
void RVM<TKernel>::LoadModelBatchPredict(int learner_typeid, Dataset& testset, String modelfilename, String testresultfilename) {
  LoadModel_(learner_typeid, modelfilename);
  BatchPredict(learner_typeid, testset, testresultfilename);
}



/**
* Save multiclass RVM model to a text file
*
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void RVM<TKernel>::SaveModel_(int learner_typeid, String modelfilename) {
  FILE *fp = fopen(modelfilename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Cannot save trained model to file!");
    return;
  }
  index_t i, j;

  if (learner_typeid_ == 0) {
    fprintf(fp, "rvm_type rvm_c\n"); // RVM classification
    fprintf(fp, "num_classes %d\n", num_classes_); // number of classes
    fprintf(fp, "labels ");
    for (i = 0; i < num_classes_; i++) 
      fprintf(fp, "%d ", int(train_labels_list_[i]));
    fprintf(fp, "\n");
  }
  else if (learner_typeid_ == 1) {
    fprintf(fp, "rvm_type rvm_r\n"); // RVM regression
  }
  fprintf(fp, "kernel_name %s\n", param_.kernelname_.c_str());
  fprintf(fp, "kernel_typeid %d\n", param_.kerneltypeid_);
  /* save kernel parameters */
  param_.kernel_.SaveParam(fp);
  fprintf(fp, "total_num_rv %d\n", total_num_rv_);
  /* save model: relevance vector indices, in the training set */
  fprintf(fp, "rv_index ");
  for (i = 0; i < total_num_rv_; i++)
    fprintf(fp, "%d ", rv_index_[i]);
  fprintf(fp, "\n");
  /* save model: coefficients and relevance vectors */
  fprintf(fp, "RV_weights\n");
  for (i = 0; i < total_num_rv_; i++) {
    fprintf(fp, "%f ", weights_[i]);
    fprintf(fp, "\n");
  }
  fprintf(fp, "RVs\n");
  for (i = 0; i < total_num_rv_; i++) {
    for (j = 0; j < num_features_; j++) { // n_rows-1
      fprintf(fp, "%f ", rv_.get(j,i));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}
/**
* Load RVM model file
*
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void RVM<TKernel>::LoadModel_(int learner_typeid, String modelfilename) {
  /* Init */
  train_labels_list_.Renew();
  train_labels_list_.Init(num_classes_);


  /* load model file */
  FILE *fp = fopen(modelfilename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open RVM model file!");
    return;
  }
  char cmd[80]; 
  index_t i, j;
  int temp_d; double temp_f;

  while (1) {
    fscanf(fp,"%80s",cmd);
    if(strcmp(cmd,"rvm_type")==0) {
      fscanf(fp,"%80s",cmd);
      if (strcmp(cmd,"rvm_c")==0) { // RVM classification
	learner_typeid_ = 0;
	fprintf(stderr, "RVM_C\n");
	// load number of classes
	if (strcmp(cmd, "num_classes")==0) {
	  fscanf(fp,"%d",&num_classes_);
	}
	// load labels
	if (strcmp(cmd, "labels")==0) {
	  for (i=0; i<num_classes_; i++) {
	    fscanf(fp,"%d",&temp_d);
	    train_labels_list_[i] = temp_d;
	  }
	}
      }
      else if (strcmp(cmd,"rvm_r")==0) { // RVM regression
	learner_typeid_ = 1;
	fprintf(stderr, "RVM_R\n");
      }
    }
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
    else if (strcmp(cmd, "total_num_rv")==0) {
      fscanf(fp,"%d",&total_num_rv_);
      break;
    }
  }
  weights_.Renew();
  weights_.Init(total_num_rv_);

  rv_.Init(num_features_, total_num_rv_);

  rv_index_.Renew();
  rv_index_.Init(total_num_rv_);

  while (1) {
    fscanf(fp,"%80s",cmd);
    // load the index of RVs
    if (strcmp(cmd, "rv_index")==0) {
      for ( i= 0; i < total_num_rv_; i++) {
	fscanf(fp,"%d",&temp_d);
	rv_index_[i]= temp_d;
      }
    }
    // load weights
    else if (strcmp(cmd, "RV_weights")==0) {
      for (i = 0; i < total_num_rv_; i++) {
	fscanf(fp,"%lf",&temp_f);
	weights_[i] = temp_f;
	fprintf(stderr, "%lf\n", weights_[i]);
      }
    }
    // load RVs
    else if (strcmp(cmd, "RVs")==0) {
      for (i = 0; i < total_num_rv_; i++) {
	for (j = 0; j < num_features_; j++) {
	  fscanf(fp,"%lf",&temp_f);
	  rv_.set(j, i, temp_f);
	}
      }
      break;
    }
  }
  fclose(fp);
}

#endif
