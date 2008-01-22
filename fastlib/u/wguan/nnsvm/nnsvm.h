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
 private:
  double sigma_;
  double gamma_;
  
 public:
  /* Init of kernel parameters */
  void Init(datanode *node) {
    sigma_ = fx_param_double_req(node, "sigma");
    gamma_ = -1.0 / (2 * math::Sqr(sigma_));
  }  
  void Copy(const SVMRBFKernel& other) {
    sigma_ = other.sigma_;
    gamma_ = other.gamma_;
  }  
  /* Kernel value evaluation */
  double Eval(const Vector& a, const Vector& b) const {
    double distance_squared = la::DistanceSqEuclidean(a, b);
    return exp(gamma_ * distance_squared);
  }  
  double sigma() const {
    return sigma_;
  }
  /* Kernel name */
  void GetName(String* kname) {
    kname->Copy("gaussian");
  }
  /* Get an type ID for kernel */
  int GetTypeId() {
    return ID_GAUSSIAN;
  }
  /* Save kernel parameters to file */
  void SaveParam(FILE* fp) {
    fprintf(fp, "sigma %g\n", sigma_);
    fprintf(fp, "gamma %g\n", gamma_);
  }
};

/**
* Class for NNSVM
*/
template<typename TKernel>
class SVM {
 public:
  typedef TKernel Kernel;
  
 private:
  struct SVM_MODELS {
    double thresh_; //negation of the intercept
    Vector alpha_; // the alpha vector
    Vector w_; // the weight vector 
    index_t num_sv_; // number of support vectors
  };
  SVM_MODELS model_;
 
  Matrix support_vectors_;

  struct SVM_PARAMETERS {
    TKernel kernel_; 
    String kernelname_; 
    int kerneltypeid_;
    double c_; 
    int b_;
	double eps_; //tolerance
	int max_iter_; // maximum iterations
  };
  SVM_PARAMETERS param_; // same for every binary model
  
 public:
  void Init(const Dataset& dataset, int n_classes, datanode *module);
  void InitTrain(const Dataset& dataset, int n_classes, datanode *module);
  void SaveModel(String modelfilename);
  int Classify(const Vector& vector);
};

/**
* NNSVM initialization
*
* @param: labeled training set
* @param: number of classes (different labels) in the data set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::Init(const Dataset& dataset, int n_classes, datanode *module){
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
void SVM<TKernel>::InitTrain(
    const Dataset& dataset, int n_classes, datanode *module) {
  Init(dataset, n_classes, module);
  DEBUG_ASSERT_MSG(n_classes == 2, "SVM is only a binary classifier");
  fx_set_param(module, "kernel_type", typeid(TKernel).name());
  
  /* Initialize parameters c_, budget_, eps_, max_iter_, VTA_, alpha_, error_, thresh_ */
  SMO<Kernel> smo;
  smo.Init(&dataset, param_.c_, param_.b_, param_.eps_, param_.max_iter_);
  smo.kernel().Copy(param_.kernel_);

  /* 2-classes NNSVM training using NNSMO */
	fx_timer_start(NULL, "nnsvm_train");
  smo.Train();  
    fx_timer_stop(NULL, "nnsvm_train");
  
  /* Get the trained bi-class model */
  smo.GetSVM(&support_vectors_, &(model_.alpha_), &(model_.w_));
  DEBUG_ASSERT(model_.alpha_.length() != 0);
  model_.num_sv_ =  support_vectors_.n_cols();
  model_.thresh_ = smo.threshold();
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
void SVM<TKernel>::SaveModel(String modelfilename) {
  FILE *fp = fopen(modelfilename,"w");
  if (fp==NULL)
    fprintf(stderr, "Cannot save trained model to file!");
  
  fprintf(fp, "svm_type svm_c\n"); // TODO: svm-mu, svm-regression...
  fprintf(fp, "kernel_name %s\n", param_.kernelname_.c_str());
  fprintf(fp, "kernel_typeid %d\n", param_.kerneltypeid_);
  // save kernel parameters
  param_.kernel_.SaveParam(fp);
  fprintf(fp, "total_num_sv %d\n", model_.num_sv_);
  fprintf(fp, "threshold %g\n", model_.thresh_);
  fprintf(fp, "weight");
  index_t len = model_.w_.length();
  for(index_t s=0; s<len; s++)
  {
	fprintf(fp, " %f", model_.w_[s]);
  }
  fprintf(fp, "\nalpha\n");
  len = model_.alpha_.length();
  for(index_t i=0; i<len; i++)
  {
		fprintf(fp, "%g\n", model_.alpha_[i]);
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
int SVM<TKernel>::Classify(const Vector& datum) {
  
  double summation = la::Dot(model_.w_, datum);
  
  DEBUG_MSG(0, "summation=%f, thresh_=%f", summation, model_.thresh_);
  
  return (summation - model_.thresh_ > 0.0) ? 1 : 0;
  
  return 0;
}

#endif
