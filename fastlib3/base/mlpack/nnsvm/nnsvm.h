/**
 * @file nnsvm.h
 *
 * This head file contains functions for performing NNSVM training.
 * NNSMO algorithm is employed. 
 *
 * @see nnsmo.h
 */

#ifndef U_NNSVM_NNSVM_H
#define U_NNSVM_NNSVM_H

#include "nnsmo.h"

#include "fastlib/fastlib.h"

#include <typeinfo>

#define ID_LINEAR 0
#define ID_GAUSSIAN 1

/**
* Class for Linear Kernel
*/
struct SVMLinearKernel {
  void Init(datanode *node) {}
  void Copy(const SVMLinearKernel& other) {}
  /* Kernel value evaluation */
  double Eval(const Vector& a, const Vector& b) const {
    return la::Dot(a, b);
  }
  /* Kernel name */
  void GetName(String* kname) {
    kname->Copy("linear");
  }
  /* Get an type ID for kernel */
  int GetTypeId() {
    return ID_LINEAR;
  }  
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp) {
  }
};

/**
* Class for Gaussian RBF Kernel
*/
class SVMRBFKernel {
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
* Class for NNSVM
*/
template<typename TKernel>
class NNSVM {
 public:
  typedef TKernel Kernel;
  
 private:
  struct NNSVM_MODELS {
    double thresh_; //negation of the intercept
    Vector sv_coef_; // the alpha vector
    Vector w_; // the weight vector 
    index_t num_sv_; // number of support vectors
  };
  NNSVM_MODELS model_;

  struct NNSVM_PARAMETERS {
    TKernel kernel_; 
    String kernelname_; 
    int kerneltypeid_;
    double c_; 
    int b_;
	double eps_; //tolerance
	int max_iter_; // maximum iterations
  };
  NNSVM_PARAMETERS param_; // same for every binary model
 
  Matrix support_vectors_;
  int num_features_;
  
 public:
  void Init(const Dataset& dataset, int n_classes, datanode *module);
  void InitTrain(const Dataset& dataset, int n_classes, datanode *module);
  void SaveModel(String modelfilename);
  void LoadModel(Dataset* testset, String modelfilename);
  int Classify(const Vector& vector);
  void BatchClassify(Dataset* testset, String testlabelfilename);
  void LoadModelBatchClassify(Dataset* testset, String modelfilename, String testlabelfilename);
};

/**
* NNSVM initialization
*
* @param: labeled training set
* @param: number of classes (different labels) in the data set
* @param: module name
*/
template<typename TKernel>
void NNSVM<TKernel>::Init(const Dataset& dataset, int n_classes, datanode *module){
  param_.kernel_.Init(fx_submodule(module, "kernel", "kernel"));
  param_.kernel_.GetName(&param_.kernelname_);
  param_.kerneltypeid_ = param_.kernel_.GetTypeId();
  // c; default:10
  param_.c_ = fx_param_double(NULL, "c", 10.0); 
  // budget parameter, contorls # of support vectors; default: # of data samples
  param_.b_ = fx_param_int(module, "b", dataset.n_points()); 
  // tolerance: eps, default: 1.0e-6
  param_.eps_ = fx_param_double(NULL, "eps", 1.0e-6); 
  //max iterations: max_iter, default: 1000
  param_.max_iter_ = fx_param_int(NULL, "max_iter", 1000); 
  fprintf(stderr, "c=%f, eps=%g, max_iter=%d \n", param_.c_, param_.eps_, param_.max_iter_);
}

/**
* Initialization(data dependent) and training for NNSVM Classifier
*
* @param: labeled training set
* @param: number of classes (different labels) in the training set
* @param: module name
*/
template<typename TKernel>
void NNSVM<TKernel>::InitTrain(
    const Dataset& dataset, int n_classes, datanode *module) {
  Init(dataset, n_classes, module);
  /* # of features = # of rows in data matrix - 1, as last row is for labels*/
  num_features_ = dataset.n_features() - 1;
  DEBUG_ASSERT_MSG(n_classes == 2, "SVM is only a binary classifier");
  fx_set_param(module, "kernel_type", typeid(TKernel).name());
  
  /* Initialize parameters c_, budget_, eps_, max_iter_, VTA_, alpha_, error_, thresh_ */
  NNSMO<Kernel> nnsmo;
  nnsmo.Init(&dataset, param_.c_, param_.b_, param_.eps_, param_.max_iter_);
  nnsmo.kernel().Copy(param_.kernel_);

  /* 2-classes NNSVM training using NNSMO */
  fx_timer_start(NULL, "nnsvm_train");
  nnsmo.Train();  
  fx_timer_stop(NULL, "nnsvm_train");
  
  /* Get the trained bi-class model */
  nnsmo.GetNNSVM(&support_vectors_, &(model_.sv_coef_), &(model_.w_));
  DEBUG_ASSERT(model_.sv_coef_.length() != 0);  
  model_.num_sv_ = support_vectors_.n_cols();
  model_.thresh_ = nnsmo.threshold();
  DEBUG_ONLY(fprintf(stderr, "THRESHOLD: %f\n", model_.thresh_));

  /* Save models to file "nnsvm_model" */
  SaveModel("nnsvm_model"); // TODO: param_req
}

/**
* Save the NNSVM model to a text file
*
* @param: name of the model file
*/
template<typename TKernel>
void NNSVM<TKernel>::SaveModel(String modelfilename) {
  FILE *fp = fopen(modelfilename,"w");
  if (fp==NULL){
    fprintf(stderr, "Cannot save trained model to file!");
    return;
  }
  
  fprintf(fp, "svm_type svm_c\n"); // TODO: svm-mu, svm-regression...
  fprintf(fp, "kernel_name %s\n", param_.kernelname_.c_str());
  fprintf(fp, "kernel_typeid %d\n", param_.kerneltypeid_);
  // save kernel parameters
  param_.kernel_.SaveParam(fp);
  fprintf(fp, "total_num_sv %d\n", model_.num_sv_);
  fprintf(fp, "threshold %g\n", model_.thresh_);
  fprintf(fp, "weights");
  index_t len = model_.w_.length();
  for(index_t s=0; s<len; s++)
  {
	fprintf(fp, " %f", model_.w_[s]);
  }
  fprintf(fp, "\nsvs\n");
  for(index_t i=0; i<model_.num_sv_; i++)
  {
     fprintf(fp, "%f ", model_.sv_coef_[i]);
     for(index_t s=0; s < num_features_; s++)
     {
        fprintf(fp, "%f ", support_vectors_.get(s, i) );
     }
     fprintf(fp, "\n");
  }  
  fclose(fp);
}



/**
* Load NNSVM model file
*
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void NNSVM<TKernel>::LoadModel(Dataset* testset, String modelfilename) {
  /* Init */
  //fprintf(stderr, "modelfilename= %s\n", modelfilename.c_str()); 
  num_features_ = testset->n_features() - 1;

  model_.w_.Init(num_features_);
  /* load model file */
  FILE *fp = fopen(modelfilename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open NNSVM model file!");
    return;
  }
  char cmd[80]; 
  int i, j; double temp_f;
  while (1) {
    fscanf(fp,"%80s",cmd);
    if(strcmp(cmd,"svm_type")==0) {
      fscanf(fp,"%80s",cmd);
      if(strcmp(cmd,"svm_c")==0) {
	fprintf(stderr, "SVM_C\n");
      }
    }
    else if (strcmp(cmd, "kernel_name")==0) {
      fscanf(fp,"%80s",param_.kernelname_.c_str());
    }
    else if (strcmp(cmd, "kernel_typeid")==0) {
      fscanf(fp,"%d",&param_.kerneltypeid_);
    }
    else if (strcmp(cmd, "total_num_sv")==0) {
      fscanf(fp,"%d",&model_.num_sv_);
    }
    else if (strcmp(cmd, "threshold")==0) {
      fscanf(fp,"%lf",&model_.thresh_); 
    }
    else if (strcmp(cmd, "weights")==0) {
      for (index_t s= 0; s < num_features_; s++) {
	fscanf(fp,"%lf",&temp_f); 
	model_.w_[s] = temp_f;
      }
      break;
    }
  }
  support_vectors_.Init(num_features_, model_.num_sv_);
  model_.sv_coef_.Init(model_.num_sv_);
  while (1) {
    fscanf(fp,"%80s",cmd);
    if (strcmp(cmd, "svs")==0) {
      for (i = 0; i < model_.num_sv_; i++) {
        fscanf(fp,"%lf",&temp_f);
        model_.sv_coef_[i] = temp_f;
	for (j = 0; j < num_features_; j++) {
	  fscanf(fp,"%lf",&temp_f);
	  support_vectors_.set(j, i, temp_f);
	}
      }
      break;
    }
  }
  fclose(fp);
}

/**
* NNSVM classification for one testing vector
*
* @param: testing vector
*
* @return: a label (integer)
*/

template<typename TKernel>
int NNSVM<TKernel>::Classify(const Vector& datum) {
  
  double summation = la::Dot(model_.w_, datum);
  
  VERBOSE_MSG(0, "summation=%f, thresh_=%f", summation, model_.thresh_);
  
  return (summation - model_.thresh_ > 0.0) ? 1 : 0;
  
  return 0;
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
void NNSVM<TKernel>::BatchClassify(Dataset* testset, String testlablefilename) {
  FILE *fp = fopen(testlablefilename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Cannot save test labels to file!");
    return;
  }
  num_features_ = testset->n_features()-1;
  for (index_t i = 0; i < testset->n_points(); i++) {
    Vector testvec;
    testset->matrix().MakeColumnSubvector(i, 0, num_features_, &testvec);
    int testlabel = Classify(testvec);
    fprintf(fp, "%d\n", testlabel);
  }
  fclose(fp);
}

/**
* Load models from a file, and perform offline batch classification for multiple testing vectors
*
* @param: testing set
* @param: name of the model file
* @param: name of the file to store classified labels
*/
template<typename TKernel>
void NNSVM<TKernel>::LoadModelBatchClassify(Dataset* testset, String modelfilename, String testlabelfilename) {
  LoadModel(testset, modelfilename);
  BatchClassify(testset, testlabelfilename);
}

#endif

