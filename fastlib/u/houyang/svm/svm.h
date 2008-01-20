/**
 * @file svm.h
 *
 * This head file contains functions for performing multiclass SVM 
 * classification. One-vs-One method is employed. 
 *
 * @see smo.h
 */

#ifndef U_SVM_SVM_H
#define U_SVM_SVM_H

#include "smo.h"

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
* Class for SVM
*/
template<typename TKernel>
class SVM {
 private:
  /* array of models for storage of the 2-class(binary) classifiers 
     Need to train num_classes_*(num_classes_-1)/2 binary models */
  struct SVM_MODELS {
    /* bias term in each binary model */
    double thresh_;
    /* all coefficients of the binary dataset, not necessarily thoes of SVs */
    ArrayList<double> bi_coef_;
  };
  ArrayList<SVM_MODELS> models_;
  /* list of labels, need to be integers. e.g. [0,1,2] for a 3-class dataset */
  ArrayList<int> train_labels_list_;
  
  /* total set of support vectors and their coefficients */
  Matrix sv_;
  Matrix sv_coef_;

  /* total number of support vectors */
  index_t total_num_sv_;
  /* array of indicators: is/isn't a support vector */
  ArrayList<bool> sv_indicator_;
  /* support vector list to store the indices (in the training set) of support vectors */
  ArrayList<index_t> sv_index_;
  /* start positions of each class of support vectors, in the support vector list */
  ArrayList<index_t> sv_list_startpos_;
  /* counted number of support vectors for each class */
  ArrayList<index_t> sv_list_ct_;

  /* SVM parameters, same for every binary model */
  struct SVM_PARAMETERS {
    TKernel kernel_;
    String kernelname_;
    int kerneltypeid_;
    double c_;
    int b_;
  };
  SVM_PARAMETERS param_;
  
  /* number of classes in the training set */
  int num_classes_;
  /* number of binary models to be trained, i.e. num_classes_*(num_classes_-1)/2 */
  int num_models_;
  int num_features_;

 public:
  typedef TKernel Kernel;

  void Init(const Dataset& dataset, int n_classes, datanode *module);
  void InitTrain(const Dataset& dataset, int n_classes, datanode *module);
  void SaveModel(String modelfilename);
  void LoadModel(Dataset* testset, String modelfilename);
  int Classify(const Vector& vector);
  void BatchClassify(Dataset* testset, String testlabelfilename);
  void LoadModelBatchClassify(Dataset* testset, String modelfilename, String testlabelfilename);
};

/**
* SVM initialization
*
* @param: labeled training set or testing set
* @param: number of classes (different labels) in the data set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::Init(const Dataset& dataset, int n_classes, datanode *module){
  models_.Init();
  sv_index_.Init();
  total_num_sv_ = 0;
  sv_indicator_.Init(dataset.n_points());
  for (index_t i=0; i<dataset.n_points(); i++)
    sv_indicator_[i] = false;

  param_.kernel_.Init(fx_submodule(module, "kernel", "kernel"));
  param_.kernel_.GetName(&param_.kernelname_);
  param_.kerneltypeid_ = param_.kernel_.GetTypeId();

  param_.c_ =  fx_param_double_req(NULL, "c");
  /* budget parameter, contorls # of support vectors; default: # of data samples (use all) */
  param_.b_ = fx_param_int(module, "b", dataset.n_points());  // TODO: param_req

  num_classes_ = n_classes;
  num_models_ = num_classes_ * (num_classes_-1) / 2;
  num_features_ = 0;
  sv_list_startpos_.Init(n_classes);
  sv_list_ct_.Init(n_classes);
}

/**
* Initialization(data dependent) and training for Multiclass SVM Classifier
* Use One-vs-One, or called All-vs-All method
*
* @param: labeled training set
* @param: number of classes (different labels) in the training set
* @param: module name
*/
template<typename TKernel>
void SVM<TKernel>::InitTrain(const Dataset& dataset, int n_classes, datanode *module) {
  Init(dataset, n_classes, module);
  /* # of features == # of rows in data matrix, since last row in dataset is for labels */
  num_features_ = dataset.n_features()-1;

  /* Group labels, split the training dataset for training bi-class SVM classifiers */
  /* array of label indices, after grouping. e.g. [c1[0,5,6,7,10,13,17],c2[1,2,4,8,9],c3[...]]*/
  ArrayList<index_t> train_labels_index;
  /* counted number of label for each class. e.g. [7,5,8]*/
  ArrayList<index_t> train_labels_ct;
  /* start positions of each classes in the training label list. e.g. [0,7,12] */
  ArrayList<index_t> train_labels_startpos;
  dataset.GetLabels(train_labels_list_, train_labels_index, train_labels_ct, train_labels_startpos);

  /* Train n_classes*(n_classes-1)/2 binary class(labels: 0, 1) models using SMO */
  index_t ct = 0;
  index_t i; index_t j;
  for (i = 0; i < n_classes; i++) {
    for (j = i+1; j < n_classes; j++) {
      models_.AddBack();

      SMO<Kernel> smo;
      /* Initialize parameters c_, budget_, alpha_, error_, thresh_ */
      smo.Init(n_classes, param_.c_, param_.b_);
      smo.kernel().Init(fx_submodule(module, "kernel", "kernel"));

      /* Construct dataset consists of two classes i and j (reassign labels 0 and 1) */
      Dataset dataset_bi;
      dataset_bi.InitBlank();
      dataset_bi.info().Init();
      dataset_bi.matrix().Init(num_features_+1, train_labels_ct[i]+train_labels_ct[j]);
      ArrayList<index_t> dataset_bi_index;
      dataset_bi_index.Init(train_labels_ct[i]+train_labels_ct[j]);
      for (index_t m = 0; m < train_labels_ct[i]; m++) {
	Vector source, dest;
	dataset_bi.matrix().MakeColumnVector(m, &dest);
	dataset.matrix().MakeColumnVector(train_labels_index[train_labels_startpos[i]+m], &source);
	dest.CopyValues(source);
	/* last row for labels 0 */
	dataset_bi.matrix().set(num_features_, m, 0);
	dataset_bi_index[m] = train_labels_index[train_labels_startpos[i]+m];
      }
      for (index_t n = 0; n < train_labels_ct[j]; n++) {
	Vector source, dest;
	dataset_bi.matrix().MakeColumnVector(n+train_labels_ct[i], &dest);
	dataset.matrix().MakeColumnVector(train_labels_index[train_labels_startpos[j]+n], &source);
	dest.CopyValues(source);
	/* last row for labels 1 */
	dataset_bi.matrix().set(num_features_, n+train_labels_ct[i], 1);
	dataset_bi_index[n+train_labels_ct[i]] = train_labels_index[train_labels_startpos[j]+n];
      }

      /* 2-classes SVM-SMO training using SMO */
      smo.Train(&dataset_bi);

      /* Get the trained bi-class model */
      models_[ct].thresh_ = smo.threshold();
      models_[ct].bi_coef_.Init();
      smo.GetSVM(dataset_bi_index, models_[ct].bi_coef_, sv_indicator_);

      ct++;
    }
  }
  
  /* Get total set of SVs from all the binary models */
  index_t k;
  sv_list_startpos_[0] = 0;
  total_num_sv_ = 0;
  for (i = 0; i < n_classes; i++) {
    ct = 0;
    for (j = 0; j < train_labels_ct[i]; j++) {
      if (sv_indicator_[train_labels_startpos[i] + j]) {
	*sv_index_.AddBack() = train_labels_index[train_labels_startpos[i] + j];
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
  /* Get coefficients for the total set of SVs, i.e. models_[x].bi_coef_ -> sv_coef_ */
  index_t p;
  index_t ct_model = 0;
  sv_coef_.Init(n_classes-1, total_num_sv_);
  sv_coef_.SetZero();
  for (i = 0; i < n_classes; i++) {
    for (j = i+1; j < n_classes; j++) {
      p = sv_list_startpos_[i];
      for (k = 0; k < train_labels_ct[i]; k++) {
	if (sv_indicator_[train_labels_startpos[i]+k]) {
	  sv_coef_.set(j-1, p++, models_[ct_model].bi_coef_[k]);
	}
      }
      p = sv_list_startpos_[j];
      for (k = 0; k < train_labels_ct[j]; k++) {
	if (sv_indicator_[train_labels_startpos[j]+k]) {
	  sv_coef_.set(i, p++, models_[ct_model].bi_coef_[k]);
	}
      }
      ct_model++;
    }
  }
  /* Save models to file "svm_model" */
  SaveModel("svm_model"); // TODO: param_req, and for CV mode
}

/**
* Save multiclass SVM model to a text file
*
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void SVM<TKernel>::SaveModel(String modelfilename) {
  FILE *fp = fopen(modelfilename,"w");
  if (fp==NULL) {
    fprintf(stderr, "Cannot save trained model to file!");
    return;
  }
  index_t i, j;

  fprintf(fp, "svm_type svm_c\n"); // TODO: svm-mu, svm-regression...
  fprintf(fp, "num_classes %d\n", num_classes_); // TODO: only for svm_c
  fprintf(fp, "kernel_name %s\n", param_.kernelname_.c_str());
  fprintf(fp, "kernel_typeid %d\n", param_.kerneltypeid_);
  /* save kernel parameters */
  param_.kernel_.SaveParam(fp);
  fprintf(fp, "total_num_sv %d\n", total_num_sv_);
  fprintf(fp, "labels ");
  for (i = 0; i < num_classes_; i++) 
    fprintf(fp, "%d ", train_labels_list_[i]);
  fprintf(fp, "\n");
  /* save models */
  fprintf(fp, "thresholds ");
  for (i = 0; i < num_models_; i++)
    fprintf(fp, "%f ", models_[i].thresh_);
  fprintf(fp, "\n");
  fprintf(fp, "sv_list_startpos ");
  for (i =0; i < num_classes_; i++)
    fprintf(fp, "%d ", sv_list_startpos_[i]);
  fprintf(fp, "\n");
  fprintf(fp, "sv_list_ct ");
  for (i =0; i < num_classes_; i++)
    fprintf(fp, "%d ", sv_list_ct_[i]);
  fprintf(fp, "\n");
  /* save coefficients and support vectors */
  fprintf(fp, "SV_coefs\n");
  for (i = 0; i < total_num_sv_; i++) {
    for (j = 0; j < num_classes_-1; j++) {
      fprintf(fp, "%f ", sv_coef_.get(j,i));
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "SVs\n");
  for (i = 0; i < total_num_sv_; i++) {
    for (j = 0; j < num_features_; j++) { // n_rows-1
      fprintf(fp, "%f ", sv_.get(j,i));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}
/**
* Load SVM model file
*
* @param: name of the model file
*/
// TODO: use XML
template<typename TKernel>
void SVM<TKernel>::LoadModel(Dataset* testset, String modelfilename) {
  /* Init */
  train_labels_list_.Init(num_classes_);
  num_features_ = testset->n_features() - 1;

  /* load model file */
  FILE *fp = fopen(modelfilename, "r");
  if (fp==NULL) {
    fprintf(stderr, "Cannot open SVM model file!");
    return;
  }
  char cmd[80]; 
  int i, j; int temp_d; double temp_f;
  for (i = 0; i < num_models_; i++) {
	models_.AddBack();
	models_[i].bi_coef_.Init();
  }
  while (1) {
    fscanf(fp,"%80s",cmd);
    if(strcmp(cmd,"svm_type")==0) {
      fscanf(fp,"%80s",cmd);
      if(strcmp(cmd,"svm_c")==0) {
	fprintf(stderr, "SVM_C\n");
      }
    }
    else if (strcmp(cmd, "num_classes")==0) {
      fscanf(fp,"%d",&num_classes_);
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
    else if (strcmp(cmd, "total_num_sv")==0) {
      fscanf(fp,"%d",&total_num_sv_);
    }
    else if (strcmp(cmd, "labels")==0) {
      for (i=0; i<num_classes_; i++) {
	fscanf(fp,"%d",&temp_d);
	train_labels_list_[i] = temp_d;
      }
    }
    else if (strcmp(cmd, "thresholds")==0) {
      for ( i= 0; i < num_models_; i++) {
	fscanf(fp,"%lf",&temp_f); 
	models_[i].thresh_= temp_f;
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
      break;
    }
  }
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

/**
* Multiclass SVM classification for one testing vector
*
* @param: testing vector
*
* @return: a label (integer)
*/
template<typename TKernel>
int SVM<TKernel>::Classify(const Vector& datum) {
  index_t i, j, k;
  ArrayList<double> keval;
  keval.Init(total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    Vector support_vector_i;
    sv_.MakeColumnVector(i, &support_vector_i);
    keval[i] = param_.kernel_.Eval(datum, support_vector_i);
  }
  ArrayList<double> values;
  values.Init(num_models_);
  index_t ct = 0;
  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      double sum = 0;
      for(k = 0; k < sv_list_ct_[i]; k++) {
	sum += sv_coef_.get(j-1, sv_list_startpos_[i]+k) * keval[sv_list_startpos_[i]+k];
      }
      for(k = 0; k < sv_list_ct_[j]; k++) {
	sum += sv_coef_.get(i, sv_list_startpos_[j]+k) * keval[sv_list_startpos_[j]+k];
      }
      sum -= models_[ct].thresh_;
      values[ct] = sum;
      //fprintf(stderr, "%f\n", values[ct]);
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
      if(values[ct] > 0.0) {
	++vote[i];
      }
      else {
	++vote[j];
      }
      ct++;
    }
  }
  index_t vote_max_idx = 0;
  for (i = 1; i < num_classes_; i++) {
    if (vote[i] > vote[vote_max_idx]) {
      vote_max_idx = i;
    }
  }
  return train_labels_list_[vote_max_idx];
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
void SVM<TKernel>::BatchClassify(Dataset* testset, String testlablefilename) {
  FILE *fp = fopen(testlablefilename,"w");
  if (fp==NULL)
    fprintf(stderr, "Cannot save test labels to file!");
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
// 
template<typename TKernel>
void SVM<TKernel>::LoadModelBatchClassify(Dataset* testset, String modelfilename, String testlabelfilename) {
  LoadModel(testset, modelfilename);
  BatchClassify(testset, testlabelfilename);
}

#endif
