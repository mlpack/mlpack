/**
 * * @author Hua Ouyang
 * *
 * * @file svm.h
 * *
 * * This head file contains functions for performing SVM training and prediction
 * * Supported SVM learner type:SVM_C, SVM_R, SVM_DE
 * *
 * * @see opt_smo.h
 * */

#ifndef U_SVM_SVM_H
#define U_SVM_SVM_H

#define REACHED_HERE std::cerr << "we reached " << __LINE__ << " in " << __FILE__ << "\n";

#include "opt_smo.h"

#include <fastlib/fastlib.h>

#include <string>
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

/**
 * * Class for Linear Kernel
 * */
class SVMLinearKernel
{
  public:
    /* Init of kernel parameters */
    std::vector<double> kpara_; /* kernel parameters */
    SVMLinearKernel() { /*TODO: NULL->node */
      }
    /* Kernel name */
    void GetName(std::string& kname)
    {
      kname = "linear";
      }
    /* Get an type ID for kernel */
    size_t GetTypeId()
    {
      return ID_LINEAR;
      }
    /* Kernel value evaluation */
    double Eval(const arma::vec& a, const arma::vec b, size_t n_cols) const
    {
      return dot(a.rows(0,n_cols),b.rows(0,n_cols));
      }
    /* Save kernel parameters to file */
    void SaveParam(std::ostream& f)
    {
      }
    };

/**
 * * Class for Gaussian RBF Kernel
 * */
class SVMRBFKernel
{
  public:
    /* Init of kernel parameters */
    std::vector<double> kpara_; /* kernel parameters */
    SVMRBFKernel() { /*TODO: NULL->node */
      kpara_.resize(2,0);
      kpara_[0] = mlpack::IO::GetParam<double>("svm/sigma"); /*sigma */
      kpara_[1] = -1.0 / (2 * math::Sqr(kpara_[0])); /*gamma */
      }
    /* Kernel name */
    void GetName(std::string &kname)
    {
      kname = "gaussian";
      }
    /* Get an type ID for kernel */
    size_t GetTypeId()
    {
      return ID_GAUSSIAN;
      }
    /* Kernel value evaluation */
    double Eval(const arma::vec& a, const arma::vec& b, size_t n_features) const
    {
      double distance_squared = arma::norm(a.subvec(0,n_features-1)-b.subvec(0,n_features-1),2);
      return exp(kpara_[1] * distance_squared);
      }
    /* Save kernel parameters to file */
    void SaveParam(std::ostream& f)
    {
      f << "sigma " << kpara_[0] << std::endl;
      f << "gamma " << kpara_[1] << std::endl;
      }
    };

/**
 * * Class for SVM
 * */
template<typename TKernel>
class SVM
{

  private:
    /**
     * * Type id of the SVM learner:
     * *  0:SVM Classification (svm_c);
     * *  1:SVM Regression (svm_r);
     * *  2:One class SVM (svm_de);
     * * Developers may add more learner types if necessary
     * */
    size_t n_labels;
    size_t learner_typeid_;
    /* Optimization method: smo */
    std::string opt_method_;
    /* array of models for storage of the 2-class(binary) classifiers
     * Need to train num_classes_*(num_classes_-1)/2 binary models */
    struct SVM_MODELS
    {
      /* bias term in each binary model */
      double bias_;
      /* all coefficients (alpha*y) of the binary dataset, not necessarily thoes of SVs */
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
    struct PARAMETERS
    {
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

/**
 * * SVM initialization
 * *
 * * @param: labeled training set or testing set
 * * @param: number of classes (different labels) in the data set
 * * @param: module name
 * */
template<typename TKernel>
void SVM<TKernel>::Init(size_t learner_typeid, arma::mat& dataset){
  learner_typeid_ = learner_typeid;

  opt_method_ = mlpack::IO::GetParam<std::string>("svm/opt"); /*Default "smo" optimization method: default using SMO */

  n_data_ = dataset.n_cols;
  /* # of features == # of row - 1, exclude the last row (for labels) */
  num_features_ = dataset.n_rows-1;
  /* # of classes of the training set */
  num_classes_ = this->getLabels(dataset, train_labels_list_).size();

  if (learner_typeid == 0) { /* for multiclass SVM classificatioin*/
    num_models_ = num_classes_ * (num_classes_-1) / 2;
    sv_list_startpos_.reserve(num_classes_);
    sv_list_ct_.reserve(num_classes_);
    }
  else { /* for other SVM learners */
    num_classes_ = 2; /* dummy #, only meaningful in SaveModel and LoadModel */

    num_models_ = 1;
    }

  total_num_sv_ = 0;

  /* bool indicators FOR THE TRAINING SET: is/isn't a support vector */
  /* Note: it has the same index as the training !!! */
  trainset_sv_indicator_.resize(n_data_,false);

  param_.kernel_.GetName(param_.kernelname_);
  param_.kerneltypeid_ = param_.kernel_.GetTypeId();
  /* working set selection scheme. default: 1st order expansion */
  param_.wss_ = mlpack::IO::GetParam<double>("svm/wss");
  /* whether do L1-SVM(1) or L2-SVM (2) */
  param_.hinge_sqhinge_ = mlpack::IO::GetParam<size_t>("svm/hinge");/*Default value 1 */
  param_.hinge_sqhinge_ = 1;

  /* accuracy for optimization */
  param_.accuracy_ = mlpack::IO::GetParam<double>("svm/accuracy");/*Default value 1e-4 */
  /* number of iterations */
  param_.n_iter_ = mlpack::IO::GetParam<size_t>("svm/n_iter");/*Default value 100000000 */

  /* tradeoff parameter for C-SV */
  param_.C_ = mlpack::IO::GetParam<double>("svm/c");
  param_.Cp_ = mlpack::IO::GetParam<double>("svm/c_p");
  param_.Cn_ = mlpack::IO::GetParam<double>("svm/c_n");

  if (learner_typeid == 1) { /* for SVM_R only */
    /* the "epsilon", default: 0.1 */
    param_.epsilon_ = mlpack::IO::GetParam<double>("svm/epsilon");
  }
  else if (learner_typeid == 2) { /* SVM_DE */
  }
}

/**
 * * Initialization(data dependent) and training for SVM learners
 *
 * *
 * * @param: typeid of the learner
 * * @param: number of classes (different labels) in the training set
 * * @param: module name
 * */
  template<typename TKernel>
void SVM<TKernel>::InitTrain(size_t learner_typeid, arma::mat& dataset)
{
  Init(learner_typeid, dataset);

  if (learner_typeid == 0) { /* Multiclass SVM Clssification */
    SVM_C_Train_(learner_typeid, dataset);
  }
  else if (learner_typeid == 1) { /* SVM Regression */
    SVM_R_Train_(learner_typeid, dataset);
  }
  else if (learner_typeid == 2) { /* One Class SVM */
    SVM_DE_Train_(learner_typeid, dataset);
  }

  /* Save models to file "svm_model" */
  SaveModel_(learner_typeid, "svm_model"); /* TODO: param_req, and for CV mode */
  /* TODO: calculate training error */
}


/**
 * * Training for Multiclass SVM Clssification, using One-vs-One method
 * *
 * * @param: type id of the learner
 * * @param: training set
 * * @param: number of classes of the training set
 * * @param: module name
 * */
template<typename TKernel>
void SVM<TKernel>::SVM_C_Train_(size_t learner_typeid, arma::mat& dataset)
{
  num_classes_ = train_labels_list_.size();
  /* Group labels, split the training dataset for training bi-class SVM classifiers */
  this->reorderDataByLabels(dataset);

  /* Train num_classes*(num_classes-1)/2 binary class(labels:-1, 1) models */
  size_t ct = 0;
  size_t i, j;
  /* Reserve space for everything in one go. */
  {
    SVM_MODELS n;
    models_.resize( num_classes_*(num_classes_-1)/2, n);
  }

  for (i = 0; i < num_classes_; i++)
  {
    for (j = i+1; j < num_classes_; j++)
    {
      /* Construct dataset consists of two classes i and j (reassign labels 1 and -1) */
      /* TODO: avoid these ugly and time-consuming memory allocation */
      arma::mat dataset_bi;
      this->buildDataByLabels (i, j, dataset, dataset_bi);
      std::vector<size_t> dataset_bi_index;
      for (size_t indexI = 0; indexI < train_labels_count_[i]; ++indexI)
      {
        dataset_bi_index.push_back (train_labels_startpos_[i] + indexI);
      }
      for (size_t indexJ = 0; indexJ < train_labels_count_[j]; ++indexJ)
      {
        dataset_bi_index.push_back (train_labels_startpos_[j] + indexJ);
      }
      /*      dataset_bi.matrix().set_size(num_features_+1, train_labels_count_[i]+train_labels_count_[j]); */
      /*      std::vector<size_t> dataset_bi_index; */
      /*      dataset_bi_index.reserve(train_labels_count_[i]+train_labels_count_[j]); */
      /*      for (size_t m = 0; m < train_labels_count_[i]; m++) { */
      /*dataset_bi.matrix().col(m) = dataset.matrix().col(train_labels_index_[train_labels_startpos_[i] + m]); */
      /* last row for labels 1 */
        /*dataset_bi.matrix()(num_features_, m) = 1; */
        /*dataset_bi_index[m] = train_labels_index_[train_labels_startpos_[i]+m]; */
        /*      } */
        /*      for (size_t n = 0; n < train_labels_count_[j]; n++) { */
        /*dataset_bi.matrix().col(n + train_labels_count_[i]) = dataset.matrix().col(train_labels_index_[train_labels_startpos_[j] + n]); */
        /* last row for labels -1 */
        /*dataset_bi.matrix()(num_features_, n+train_labels_count_[i]) = -1; */
        /*dataset_bi_index[n+train_labels_count_[i]] = train_labels_index_[train_labels_startpos_[j]+n]; */
        /*      } */

    if (opt_method_== "smo")
    {
      /* Initialize SMO parameters */
      SMO<Kernel> smo;
      smo.InitPara(learner_typeid, param_.Cp_, param_.Cn_, param_.hinge_sqhinge_, param_.wss_, param_.n_iter_, param_.accuracy_);

      /* Initialize kernel */
      /* 2-classes SVM training using SMO */
      mlpack::IO::StartTimer("svm/train_smo");
      smo.Train(learner_typeid, &dataset_bi);
      mlpack::IO::StopTimer("svm/train_smo");

      /* Get the trained bi-class model */
      models_[ct].bias_ = smo.Bias(); /* bias */
      /* TODO: we can make this much more efficient */
      smo.GetSV(dataset_bi_index, models_[ct].coef_, trainset_sv_indicator_); /* get support vectors */
    }
    else
    {
      std::cerr << "ERROR!!! Unknown optimization method!\n";
    }

      ct++;
    }
  }

  /* Get total set of SVs from all the binary models */
  size_t k;
  sv_list_startpos_[0] = 0;

  for (i = 0; i < num_classes_; i++)
  {
    ct = 0;
    for (j = 0; j < train_labels_count_[i]; j++)
    {
      if (trainset_sv_indicator_[train_labels_startpos_[i]+j])
      {
        sv_index_.push_back(train_labels_startpos_[i]+j);
        total_num_sv_++;
        ct++;
      }
    }
    sv_list_ct_[i] = ct;
    if (i >= 1)
      sv_list_startpos_[i] = sv_list_startpos_[i-1] + sv_list_ct_[i-1];
  }
  sv_.set_size(num_features_, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++)
  {
    /* last row of dataset is for labels */
    for(size_t j = 0; j < num_features_; j++)
      sv_(j,i) = dataset(j,sv_index_[i]);
  }
  /* Get the matrix sv_coef_ which stores the coefficients of all sets of SVs */
  /* i.e. models_[x].coef_ -> sv_coef_ */
  size_t ct_model = 0;
  size_t p;
  sv_coef_.set_size(num_classes_-1, total_num_sv_);
  sv_coef_.zeros();
  for (i = 0; i < num_classes_; i++)
  {
    for (j = i+1; j < num_classes_; j++)
    {
      p = sv_list_startpos_[i];
      for (k = 0; k < train_labels_count_[i]; k++)
      {
        if (trainset_sv_indicator_[train_labels_startpos_[i]+k])
        {
          sv_coef_(j-1, p++)= models_[ct_model].coef_[k];
        }
      }
      p = sv_list_startpos_[j];
      for (k = 0; k < train_labels_count_[j]; k++)
      {
        if (trainset_sv_indicator_[train_labels_startpos_[j]+k])
        {
          sv_coef_(i, p++) = models_[ct_model].coef_[train_labels_count_[i] + k];
        }
      }
      ct_model++;
    }
  }
}

/**
 * * Training for SVM Regression
 * *
 * * @param: type id of the learner
 * * @param: training set
 * * @param: module name
 * */
template<typename TKernel>
void SVM<TKernel>::SVM_R_Train_(size_t learner_typeid, arma::mat& dataset)
{
  size_t i;
  std::vector<size_t> dataset_index;
  dataset_index.reserve(n_data_);
  models_.reserve(models_.size() + n_data_);
  for (i=0; i<n_data_; i++)
    dataset_index[i] = i;

  models_.push_back(*new SVM_MODELS);

  if (opt_method_== "smo")
  {
    /* Initialize SMO parameters */
    SMO<Kernel> smo;
    smo.InitPara(learner_typeid, param_.Cp_, param_.epsilon_, param_.hinge_sqhinge_, param_.wss_, param_.n_iter_, param_.accuracy_);

    /* Initialize kernel */
    /* SVM_R Training using SMO*/
    smo.Train(learner_typeid, &dataset);

    /* Get the trained model */
    models_[0].bias_ = smo.Bias(); /* bias */
    smo.GetSV(dataset_index, models_[0].coef_, trainset_sv_indicator_); /* get support vectors */
  }
  else
  {
    std::cerr << "ERROR!!! Unknown optimization method!\n";
  }

  /* Get index list of support vectors */
  for (i = 0; i < n_data_; i++)
  {
    if (trainset_sv_indicator_[i])
    {
      sv_index_.push_back(i);
      total_num_sv_++;
    }
  }

  /* Get support vecotors and coefficients */
  sv_.set_size(num_features_, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++)
  {
    /* last row of dataset is for labels */
    for(size_t j = 0; j < num_features_; j++)
      sv_(j,i)= dataset(j,sv_index_[i]);
  }
  sv_coef_.set_size(1, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++)
  {
    sv_coef_(0, i) = models_[0].coef_[i];
  }

}

/**
 * * Training for One Class SVM
 * *
 * * @param: type id of the learner
 * * @param: training set
 * * @param: module name
 * */
  template<typename TKernel>
void SVM<TKernel>::SVM_DE_Train_(size_t learner_typeid, arma::mat& dataset)
{
  /* TODO */
}


/**
 * * SVM prediction for one testing vector
 * *
 * * @param: type id of the learner
 * * @param: testing vector
 * *
 * * @return: predited value
 * */
  template<typename TKernel>
double SVM<TKernel>::Predict(size_t learner_typeid, const arma::vec& datum)
{
  double predicted_value = INFINITY;
  if (learner_typeid == 0) { /* Multiclass SVM Clssification */
    predicted_value = SVM_C_Predict_(datum);
  }
  else if (learner_typeid == 1) { /* SVM Regression */
    predicted_value = SVM_R_Predict_(datum);
  }
  else if (learner_typeid == 2) { /* One class SVM */
    predicted_value = SVM_DE_Predict_(datum);
  }
  return predicted_value;
}

/**
 * * Multiclass SVM classification for one testing vector
 * *
 * * @param: testing vector
 * *
 * * @return: a label (double-type-integer, e.g. 1.0, 2.0, 3.0)
 * */
  template<typename TKernel>
double SVM<TKernel>::SVM_C_Predict_(const arma::vec& datum)
{
  size_t i, j, k;
  std::vector<double> keval;
  keval.resize(total_num_sv_,0);
  for (i = 0; i < total_num_sv_; i++)
  {
    keval[i] = param_.kernel_.Eval(datum, sv_.col(i), num_features_);
  }
  std::vector<double> values;
  values.resize(num_models_,0);
  size_t ct = 0;
  double sum = 0.0;
  for (i = 0; i < num_classes_; i++)
  {
    for (j = i+1; j < num_classes_; j++)
    {
      if (opt_method_== "smo")
      {
        sum = 0.0;
        for(k = 0; k < sv_list_ct_[i]; k++)
        {
          sum += sv_coef_(j-1, sv_list_startpos_[i]+k) * keval[sv_list_startpos_[i]+k];
        }
        for(k = 0; k < sv_list_ct_[j]; k++)
        {
          sum += sv_coef_(i, sv_list_startpos_[j]+k) * keval[sv_list_startpos_[j]+k];
        }
        sum += models_[ct].bias_;
      }
      values[ct] = sum;
      ct++;
    }
  }

  std::vector<size_t> vote;
  vote.resize(num_classes_,0);
  ct = 0;
  for (i = 0; i < num_classes_; i++)
  {
    for (j = i+1; j < num_classes_; j++)
    {
      if(values[ct] > 0.0) { /* label 1 in bi-classifiers (for i=...) */
        vote[i] = vote[i] + 1;
      }
      else {  /* label -1 in bi-classifiers (for j=...) */
        vote[j] = vote[j] + 1;
      }
      ct++;
    }
  }
  size_t vote_max_idx = 0;
  for (i = 1; i < num_classes_; i++)
  {
    if (vote[i] >= vote[vote_max_idx])
    {
      vote_max_idx = i;
    }
  }
  return train_labels_list_[vote_max_idx];
}

/**
 * * SVM Regression Prediction for one testing vector
 * *
 * * @param: testing vector
 * *
 * * @return: predicted regression value
 * */
  template<typename TKernel>
double SVM<TKernel>::SVM_R_Predict_(const arma::vec& datum)
{
  size_t i;
  double sum = 0.0;
  if (opt_method_== "smo")
  {
    for (i = 0; i < total_num_sv_; i++)
    {
      sum += sv_coef_(0, i) * param_.kernel_.Eval(datum, sv_.col(i), num_features_);
    }
  }
  sum += models_[0].bias_;
  return sum;
}

/**
 * * One class SVM Prediction for one testing vector
 * *
 * * @param: testing vector
 * *
 * * @return: estimated value
 * */
  template<typename TKernel>
double SVM<TKernel>::SVM_DE_Predict_(const arma::vec& datum)
{
  /* TODO */
  return 0.0;
}



/**
 * * Batch classification for multiple testing vectors. No need to load model file,
 * * since models are already in RAM.
 * *
 * * Note: for test set, if no true test labels provided, just put some dummy labels
 * * (e.g. all -1) in the last row of testset
 * *
 * * @param: type id of the learner
 * * @param: testing set
 * * @param: file name of the testing data
 * */
  template<typename TKernel>
void SVM<TKernel>::BatchPredict(size_t learner_typeid, arma::mat& testset, std::string predictedvalue_filename)
{
  FILE *fp = fopen(predictedvalue_filename.c_str(), "w");
  if (fp == NULL)
  {
    std::cerr << "Cannot save predicted values to file!\n";
    return;
  }
  size_t err_ct = 0;
  num_features_ = testset.n_rows-1;
  for (size_t i = 0; i < testset.n_rows; i++)
  {
    arma::vec testvec = testset.row(i);

    double predictedvalue = Predict(learner_typeid, testvec);
    if (predictedvalue != testset(i, num_features_))
      err_ct++;
    /* save predicted values to file*/
    fprintf(fp, "%f\n", predictedvalue);
  }
  fclose(fp);
  /* calculate testing error */
  printf( "\n*** %zu out of %zu misclassified ***\n", err_ct, (size_t) (testset.n_rows) );
  printf( "*** Testing error is %f, accuracy is %f. ***\n", double(err_ct)/double(testset.n_rows), 1- double(err_ct)/double(testset.n_rows) );
  /*fprintf( stderr, "*** Results are save in \"%s\" ***\n\n", predictedvalue_filename.c_str()); */
}

/**
 * * Load models from a file, and perform offline batch classification for multiple testing vectors
 * *
 * * @param: type id of the learner
 * * @param: testing set
 * * @param: name of the model file
 * * @param: name of the file to store classified labels
 * */
  template<typename TKernel>
void SVM<TKernel>::LoadModelBatchPredict(size_t learner_typeid, arma::mat& testset, std::string model_filename, std::string predictedvalue_filename)
{
  LoadModel_(learner_typeid, model_filename);
  BatchPredict(learner_typeid, testset, predictedvalue_filename);
}


/**
 * * Save SVM model to a text file
 * *
 * * @param: type id of the learner
 * * @param: name of the model file
 * */
  template<typename TKernel>
void SVM<TKernel>::SaveModel_(size_t learner_typeid, std::string model_filename)
{
  std::ofstream f(model_filename.c_str());

  if (!f.is_open())
    mlpack::IO::Fatal << "Cannot save trained SVM model to file " <<
      model_filename << "!" << std::endl;

  size_t i, j;

  if (learner_typeid == 0) { /* for SVM_C */
    f << "svm_type SVM_C" << std::endl;
    f << "total_num_sv " << total_num_sv_ << std::endl;
    f << "num_classes " << num_classes_ << std::endl;

    /* Save labels. */
    f << "labels ";
    for (i = 0; i < num_classes_; i++)
      f << train_labels_list_[i] << " ";
    f << std::endl;

    /* Save support vector info. */
    f << "sv_list_startpos ";
    for (i = 0; i < num_classes_; i++)
      f << sv_list_startpos_[i] << " ";
    f << std::endl;

    f << "sv_list_ct ";
    for (i = 0; i < num_classes_; i++)
      f << sv_list_ct_[i] << " ";
    f << std::endl;
  }
  else if (learner_typeid == 1) { /* for SVM_R */
    f << "svm_type SVM_R" << std::endl;
    f << "total_num_sv " << total_num_sv_ << std::endl;
    f << "sv_index ";
    for (i = 0; i < total_num_sv_; i++)
      f << sv_index_[i] << " ";
    f << std::endl;
  }
  else if (learner_typeid == 2) { /* for SVM_DE */
    f << "svm_type SVM_DE" << std::endl;
    f << "total_num_sv " << total_num_sv_ << std::endl;
    f << "sv_index ";
    for (i = 0; i < total_num_sv_; i++)
      f << sv_index_[i] << " ";
    f << std::endl;
  }

  /* save kernel parameters */
  f << "kernel_name " << param_.kernelname_ << std::endl;
  f << "kernel_typeid " << param_.kerneltypeid_ << std::endl;
  param_.kernel_.SaveParam(f);

  /* save models: bias, coefficients and support vectors */
  f << "bias ";
  for (i = 0; i < num_models_; i++)
    f << models_[i].bias_ << " ";
  f << std::endl;

  f << "SV_coefs" << std::endl;
  for (i = 0; i < total_num_sv_; i++)
  {
    for (j = 0; j < num_classes_-1; j++)
    {
      f << sv_coef_(j, i) << " ";
    }
    f << std::endl;
  }

  f << "SVs" << std::endl;
  for (i = 0; i < total_num_sv_; i++)
  {
    for (j = 0; j < num_features_; j++) { /* n_rows-1 */
      f << sv_(j, i) << " ";
    }
    f << std::endl;
  }

  f.close();
}

/**
 * * Load SVM model file
 * *
 * * @param: type id of the learner
 * * @param: name of the model file
 * */
/* TODO: use XML */
  template<typename TKernel>
void SVM<TKernel>::LoadModel_(size_t learner_typeid, std::string model_filename)
{
  if (learner_typeid == 0) {/* SVM_C */
    train_labels_list_.clear();
    train_labels_list_.reserve(num_classes_); /* get labels list from the model file */
  }

  /* load model file */
  FILE *fp = fopen(model_filename.c_str(), "r");
  if (fp == NULL)
  {
    std::cerr << "Cannot open SVM model file!\n";
    return;
  }
  char cmd[80];
  size_t i, j; size_t temp_d; double temp_f;
  models_.reserve(num_models_+models_.size());
  for (i = 0; i < num_models_; i++)
  {
    models_.push_back(*new SVM_MODELS);
  }
  while (1)
  {
    fscanf(fp,"%80s",cmd);
    if(strcmp(cmd,"svm_type")==0)
    {
      fscanf(fp,"%80s", cmd);
      if (strcmp(cmd,"SVM_C")==0)
        learner_typeid_ = 0;
      else if (strcmp(cmd,"SVM_R")==0)
        learner_typeid_ = 1;
      else if (strcmp(cmd,"SVM_DE")==0)
        learner_typeid_ = 2;
    }
    else if (strcmp(cmd, "total_num_sv")==0)
    {
      fscanf(fp,"%zu",&total_num_sv_);
    }
    /* for SVM_C */
    else if (strcmp(cmd, "num_classes")==0)
    {
      fscanf(fp,"%zu",&num_classes_);
    }
    else if (strcmp(cmd, "labels")==0)
    {
      for (i=0; i<num_classes_; i++)
      {
        fscanf(fp,"%lf",&temp_f);
        train_labels_list_[i] = temp_f;
      }
    }
    else if (strcmp(cmd, "sv_list_startpos")==0)
    {
      for ( i= 0; i < num_classes_; i++)
      {
        fscanf(fp,"%zu",&temp_d);
        sv_list_startpos_[i]= temp_d;
      }
    }
    else if (strcmp(cmd, "sv_list_ct")==0)
    {
      for ( i= 0; i < num_classes_; i++)
      {
        fscanf(fp,"%zu",&temp_d);
        sv_list_ct_[i]= temp_d;
      }
    }
    /* for SVM_R */
    else if (strcmp(cmd, "sv_index")==0)
    {
      for ( i= 0; i < total_num_sv_; i++)
      {
        fscanf(fp,"%zu",&temp_d);
        sv_index_.push_back(temp_d);
      }
    }
    /* load kernel info */
    else if (strcmp(cmd, "kernel_name")==0)
    {
      /* Switch to iostream file input instead? =/ */
      char in[81];
      fscanf(fp,"%80s",in);
      param_.kernelname_ = in;
    }
    else if (strcmp(cmd, "kernel_typeid")==0)
    {
      fscanf(fp,"%zu",&param_.kerneltypeid_);
    }
    else if (strcmp(cmd, "sigma")==0)
    {
      fscanf(fp,"%lf",&param_.kernel_.kpara_[0]); /* for gaussian kernels only */
    }
    else if (strcmp(cmd, "gamma")==0)
    {
      fscanf(fp,"%lf",&param_.kernel_.kpara_[1]); /* for gaussian kernels only */
    }
    /* load bias */
    else if (strcmp(cmd, "bias")==0)
    {
      for ( i= 0; i < num_models_; i++)
      {
        fscanf(fp,"%lf",&temp_f);
        models_[i].bias_= temp_f;
      }
      break;
    }
  }

  /* load coefficients and support vectors */
  sv_coef_.set_size(num_classes_-1, total_num_sv_);
  sv_coef_.zeros();
  sv_.set_size(num_features_, total_num_sv_);
  while (1)
  {
    fscanf(fp,"%80s",cmd);
    if (strcmp(cmd, "SV_coefs")==0)
    {
      for (i = 0; i < total_num_sv_; i++)
      {
        for (j = 0; j < num_classes_-1; j++)
        {
          fscanf(fp,"%lf",&temp_f);
          sv_coef_(j, i) = temp_f;
        }
      }
    }
    else if (strcmp(cmd, "SVs")==0)
    {
      for (i = 0; i < total_num_sv_; i++)
      {
        for (j = 0; j < num_features_; j++)
        {
          fscanf(fp,"%lf",&temp_f);
          sv_(j, i) = temp_f;
        }
      }
      break;
    }
  }
  fclose(fp);
}

template<typename TKernel>
std::vector<int>& SVM<TKernel>::getLabels (arma::mat& data, std::vector<int>& labels)
{
  /* we'll assume that the last column of the data contains labels;
   * * another useful idea would be to state explicitly in the file the number of labels */
  size_t n_rows = data.n_rows;
  std::set<int> labelSet;
  for (size_t n = 0; n < data.n_cols; n++)
  {
    labelSet.insert ((int) (data(n_rows - 1, n)));
  }
  for (std::set<int>::iterator it = labelSet.begin(); it != labelSet.end(); ++it)
  {
    labels.push_back(*it);
  }
  std::sort (labels.begin(), labels.end());
  return labels;
}

template<typename TKernel>
void SVM<TKernel>::reorderDataByLabels(arma::mat& data)
{
  for (std::vector<int>::iterator it = train_labels_list_.begin(); it != train_labels_list_.end(); ++it)
  {
    train_labels_count_.push_back (0);
    train_labels_startpos_.push_back (0);
  }
  size_t currentIndex = 0;
  size_t searchIndex = 0;
  size_t columnCount = data.n_cols;
  size_t rowCount = data.n_rows;
  for (size_t iterator = 0; iterator < train_labels_list_.size(); ++iterator)
  {
    int currentLabel = train_labels_list_[iterator];
    bool foundLabel = false;
    for (searchIndex = currentIndex; searchIndex < columnCount; ++searchIndex)
    {
      if (currentLabel == (int) (data(rowCount - 1, searchIndex)))
      {
        if (!foundLabel)
        {
          train_labels_startpos_[iterator] = currentIndex;
          foundLabel = true;
        }
        /* swap the columns */
        if (searchIndex != currentIndex)
        {
          data.swap_cols (currentIndex, searchIndex);
        }
        ++(train_labels_count_[iterator]);
        ++currentIndex;
      }
    }
  }
}

template<typename TKernel>
arma::mat& SVM<TKernel>::buildDataByLabels (size_t i, size_t j, arma::mat& data, arma::mat& returnData)
{
  size_t countForI = train_labels_count_[i];
  size_t countForJ = train_labels_count_[j];
  size_t startPosForI = train_labels_startpos_[i];
  size_t startPosForJ = train_labels_startpos_[j];

  returnData = arma::mat (data.n_rows, countForI + countForJ);

  size_t columnCounter = 0;
  for (size_t countI = startPosForI; countI < startPosForI + countForI; countI++)
  {
    for (size_t rowCounter = 0; rowCounter < data.n_rows - 1; rowCounter++)
    {
      returnData(rowCounter, columnCounter) = data(rowCounter, countI);
    }
    returnData(data.n_rows - 1, columnCounter) = 1.0;
    columnCounter++;
  }
  for (size_t countJ = startPosForJ; countJ < startPosForJ + countForJ; countJ++)
  {
    for (size_t rowCounter = 0; rowCounter < data.n_rows - 1; rowCounter++)
    {
      returnData(rowCounter, columnCounter) = data(rowCounter, countJ);
    }
    returnData(data.n_rows - 1, columnCounter) = -1.0;
    columnCounter++;
  }
  return returnData;
}
#endif
