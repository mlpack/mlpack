/**
 * @author Hua Ouyang
 *
 * @file svm_impl.h
 *
 * This header file contains the implementation of the SVM class.
 * Supported SVM learner type:SVM_C, SVM_R, SVM_DE
 *
 * @see opt_smo.h
 */
#ifndef __MLPACK_METHODS_SVM_SVM_IMPL_H
#define __MLPACK_METHODS_SVM_SVM_IMPL_H

// In case it has not already been included.
#include <err.h>
#include <errno.h>
#include "svm.h"

/* TODO TODO TODO: this is temporary to hide a bunch of warnings; we shouldn't directly use fscanf in this file (or really anywhere else in MLPack) */

#define FSCANF_CHECK(...) if (fscanf (__VA_ARGS__) == EOF) errx (1, "You've reached a terrible end...");

namespace mlpack {
namespace svm {

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

  opt_method_ = CLI::GetParam<std::string>("svm/opt"); /*Default "smo" optimization method: default using SMO */

  n_data_ = dataset.n_cols;
  /* # of features == # of row - 1, exclude the last row (for labels) */
  num_features_ = dataset.n_rows - 1;
  /* # of classes of the training set */
  num_classes_ = this->getLabels(dataset, train_labels_list_).size();

  if (learner_typeid == 0) { /* for multiclass SVM classificatioin*/
    num_models_ = num_classes_ * (num_classes_ - 1) / 2;
    sv_list_startpos_.reserve(num_classes_);
    sv_list_ct_.reserve(num_classes_);
  } else { /* for other SVM learners */
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
  param_.wss_ = CLI::GetParam<double>("svm/wss");
  /* whether do L1-SVM(1) or L2-SVM (2) */
  param_.hinge_sqhinge_ = CLI::GetParam<size_t>("svm/hinge"); /* Default value 1 */
  param_.hinge_sqhinge_ = 1;

  /* accuracy for optimization */
  param_.accuracy_ = CLI::GetParam<double>("svm/accuracy"); /* Default value 1e-4 */
  /* number of iterations */
  param_.n_iter_ = CLI::GetParam<size_t>("svm/n_iter"); /* Default value 100000000 */

  /* tradeoff parameter for C-SV */
  param_.C_ = CLI::GetParam<double>("svm/c");
  param_.Cp_ = CLI::GetParam<double>("svm/c_p");
  param_.Cn_ = CLI::GetParam<double>("svm/c_n");

  if (learner_typeid == 1) { /* for SVM_R only */
    /* the "epsilon", default: 0.1 */
    param_.epsilon_ = CLI::GetParam<double>("svm/epsilon");
  } else if (learner_typeid == 2) { /* SVM_DE */ }
}

/**
 * Initialization(data dependent) and training for SVM learners
 *
 * @param: typeid of the learner
 * @param: number of classes (different labels) in the training set
 * @param: module name
 */
template<typename TKernel>
void SVM<TKernel>::InitTrain(size_t learner_typeid, arma::mat& dataset) {
  Init(learner_typeid, dataset);

  if (learner_typeid == 0) /* Multiclass SVM Clssification */
    SVM_C_Train_(learner_typeid, dataset);
  else if (learner_typeid == 1) /* SVM Regression */
    SVM_R_Train_(learner_typeid, dataset);
  else if (learner_typeid == 2) /* One Class SVM */
    SVM_DE_Train_(learner_typeid, dataset);

  /* Save models to file "svm_model" */
  SaveModel_(learner_typeid, "svm_model"); /* TODO: param_req, and for CV mode */
  /* TODO: calculate training error */
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
void SVM<TKernel>::SVM_C_Train_(size_t learner_typeid, arma::mat& dataset) {
  num_classes_ = train_labels_list_.size();
  /* Group labels, split the training dataset for training bi-class SVM classifiers */
  this->reorderDataByLabels(dataset);

  /* Train num_classes*(num_classes-1)/2 binary class(labels:-1, 1) models */
  size_t ct = 0;
  size_t i, j;

  /* Reserve space for everything in one go. */
  models_.resize( num_classes_*(num_classes_-1)/2);

  for (i = 0; i < num_classes_; i++) {
    for (j = i + 1; j < num_classes_; j++) {
      /* Construct dataset consists of two classes i and j (reassign labels 1 and -1) */
      /* TODO: avoid these ugly and time-consuming memory allocation */
      arma::mat dataset_bi;
      this->buildDataByLabels (i, j, dataset, dataset_bi);
      std::vector<size_t> dataset_bi_index;

      for (size_t indexI = 0; indexI < train_labels_count_[i]; ++indexI)
        dataset_bi_index.push_back(train_labels_startpos_[i] + indexI);

      for (size_t indexJ = 0; indexJ < train_labels_count_[j]; ++indexJ)
        dataset_bi_index.push_back(train_labels_startpos_[j] + indexJ);

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

      if (opt_method_== "smo") {
        /* Initialize SMO parameters */
        SMO<Kernel> smo;
        smo.InitPara(learner_typeid, param_.Cp_, param_.Cn_, param_.hinge_sqhinge_, param_.wss_, param_.n_iter_, param_.accuracy_);

        /* Initialize kernel */
        /* 2-classes SVM training using SMO */
        Timers::StartTimer("svm/train_smo");
        smo.Train(learner_typeid, &dataset_bi);
        Timers::StopTimer("svm/train_smo");

        /* Get the trained bi-class model */
        models_[ct].bias_ = smo.Bias(); /* bias */
        /* TODO: we can make this much more efficient */
        smo.GetSV(dataset_bi_index, models_[ct].coef_, trainset_sv_indicator_); /* get support vectors */
      } else
        Log::Fatal << "--svm/opt: Unknown optimization method." << std::endl;

      ct++;
    }
  }

  /* Get total set of SVs from all the binary models */
  size_t k;
  sv_list_startpos_[0] = 0;

  for (i = 0; i < num_classes_; i++) {
    ct = 0;
    for (j = 0; j < train_labels_count_[i]; j++) {
      if (trainset_sv_indicator_[train_labels_startpos_[i]+j]) {
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
  for (i = 0; i < total_num_sv_; i++) {
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

  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      p = sv_list_startpos_[i];
      for (k = 0; k < train_labels_count_[i]; k++) {
        if (trainset_sv_indicator_[train_labels_startpos_[i] + k]) {
          sv_coef_(j - 1, p++) = models_[ct_model].coef_[k];
        }
      }

      p = sv_list_startpos_[j];
      for (k = 0; k < train_labels_count_[j]; k++) {
        if (trainset_sv_indicator_[train_labels_startpos_[j] + k]) {
          sv_coef_(i, p++) = models_[ct_model].coef_[train_labels_count_[i] + k];
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
void SVM<TKernel>::SVM_R_Train_(size_t learner_typeid, arma::mat& dataset) {
  size_t i;
  std::vector<size_t> dataset_index;
  dataset_index.reserve(n_data_);
  models_.reserve(models_.size() + n_data_);
  for (i = 0; i < n_data_; i++)
    dataset_index[i] = i;

  models_.push_back(*new SVM_MODELS);

  if (opt_method_ == "smo") {
    /* Initialize SMO parameters */
    SMO<Kernel> smo;
    smo.InitPara(learner_typeid, param_.Cp_, param_.epsilon_, param_.hinge_sqhinge_, param_.wss_, param_.n_iter_, param_.accuracy_);

    /* Initialize kernel */
    /* SVM_R Training using SMO*/
    smo.Train(learner_typeid, &dataset);

    /* Get the trained model */
    models_[0].bias_ = smo.Bias(); /* bias */
    smo.GetSV(dataset_index, models_[0].coef_, trainset_sv_indicator_); /* get support vectors */
  } else
    Log::Fatal << "--svm/opt: Unknown optimization method." << std::endl;

  /* Get index list of support vectors */
  for (i = 0; i < n_data_; i++) {
    if (trainset_sv_indicator_[i]) {
      sv_index_.push_back(i);
      total_num_sv_++;
    }
  }

  /* Get support vecotors and coefficients */
  sv_.set_size(num_features_, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    /* last row of dataset is for labels */
    for(size_t j = 0; j < num_features_; j++)
      sv_(j,i)= dataset(j,sv_index_[i]);
  }

  sv_coef_.set_size(1, total_num_sv_);
  for (i = 0; i < total_num_sv_; i++) {
    sv_coef_(0, i) = models_[0].coef_[i];
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
void SVM<TKernel>::SVM_DE_Train_(size_t learner_typeid, arma::mat& dataset)
{
  /* TODO */
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
double SVM<TKernel>::Predict(size_t learner_typeid, const arma::vec& datum) {
  double predicted_value = INFINITY;
  if (learner_typeid == 0) /* Multiclass SVM Clssification */
    predicted_value = SVM_C_Predict_(datum);
  else if (learner_typeid == 1) /* SVM Regression */
    predicted_value = SVM_R_Predict_(datum);
  else if (learner_typeid == 2) /* One class SVM */
    predicted_value = SVM_DE_Predict_(datum);

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
double SVM<TKernel>::SVM_C_Predict_(const arma::vec& datum) {
  size_t i, j, k;
  std::vector<double> keval;
  keval.resize(total_num_sv_,0);
  for (i = 0; i < total_num_sv_; i++) {
    keval[i] = param_.kernel_.Eval(datum, sv_.col(i), num_features_);
  }

  std::vector<double> values;
  values.resize(num_models_,0);
  size_t ct = 0;
  double sum = 0.0;
  for (i = 0; i < num_classes_; i++) {
    for (j = i + 1; j < num_classes_; j++) {
      if (opt_method_== "smo") {
        sum = 0.0;
        for (k = 0; k < sv_list_ct_[i]; k++) {
          sum += sv_coef_(j-1, sv_list_startpos_[i]+k) * keval[sv_list_startpos_[i]+k];
        }
        for (k = 0; k < sv_list_ct_[j]; k++) {
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
  for (i = 0; i < num_classes_; i++) {
    for (j = i+1; j < num_classes_; j++) {
      if(values[ct] > 0.0) { /* label 1 in bi-classifiers (for i=...) */
        vote[i] = vote[i] + 1;
      } else {  /* label -1 in bi-classifiers (for j=...) */
        vote[j] = vote[j] + 1;
      }
      ct++;
    }
  }

  size_t vote_max_idx = 0;
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
double SVM<TKernel>::SVM_R_Predict_(const arma::vec& datum) {
  size_t i;
  double sum = 0.0;
  if (opt_method_== "smo") {
    for (i = 0; i < total_num_sv_; i++) {
      sum += sv_coef_(0, i) * param_.kernel_.Eval(datum, sv_.col(i), num_features_);
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
double SVM<TKernel>::SVM_DE_Predict_(const arma::vec& datum) {
  /* TODO */
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
void SVM<TKernel>::BatchPredict(size_t learner_typeid, arma::mat& testset, std::string predictedvalue_filename) {
  FILE *fp = fopen(predictedvalue_filename.c_str(), "w");
  if (fp == NULL) {
    Log::Warn << "Failed to open '" << predictedvalue_filename
        << "' for writing." << std::endl;
  }
  size_t err_ct = 0;
  num_features_ = testset.n_rows - 1;
  for (size_t i = 0; i < testset.n_rows; i++) {
    arma::vec testvec = testset.row(i);

    double predictedvalue = Predict(learner_typeid, testvec);
    if (predictedvalue != testset(i, num_features_))
      err_ct++;
    /* save predicted values to file*/
    fprintf(fp, "%f\n", predictedvalue);
  }
  fclose(fp);

  /* calculate testing error */
  Log::Info << err_ct << " out of " << (size_t) testset.n_rows
      << " misclassified." << std::endl;
  Log::Info << "Testing error is "
      << (double(err_ct) / double(testset.n_rows)) * 100.0
      << "%; accuracy is "
      << (1 - double(err_ct) / double(testset.n_rows)) * 100.0
      << "%." << std::endl;
  Log::Info << "Results saved in " << predictedvalue_filename << "."
      << std::endl;
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
void SVM<TKernel>::LoadModelBatchPredict(size_t learner_typeid, arma::mat& testset, std::string model_filename, std::string predictedvalue_filename) {
  LoadModel_(learner_typeid, model_filename);
  BatchPredict(learner_typeid, testset, predictedvalue_filename);
}

/**
 * Save SVM model to a text file
 *
 * @param: type id of the learner
 * @param: name of the model file
 */
template<typename TKernel>
void SVM<TKernel>::SaveModel_(size_t learner_typeid, std::string model_filename) {
  std::ofstream f(model_filename.c_str());

  if (!f.is_open())
    Log::Fatal << "Cannot save trained SVM model to file " << model_filename
        << "." << std::endl;

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
  } else if (learner_typeid == 1) { /* for SVM_R */
    f << "svm_type SVM_R" << std::endl;
    f << "total_num_sv " << total_num_sv_ << std::endl;
    f << "sv_index ";
    for (i = 0; i < total_num_sv_; i++)
      f << sv_index_[i] << " ";
    f << std::endl;
  } else if (learner_typeid == 2) { /* for SVM_DE */
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
  for (i = 0; i < total_num_sv_; i++) {
    for (j = 0; j < num_classes_-1; j++) {
      f << sv_coef_(j, i) << " ";
    }

    f << std::endl;
  }

  f << "SVs" << std::endl;
  for (i = 0; i < total_num_sv_; i++) {
    for (j = 0; j < num_features_; j++) { /* n_rows - 1 */
      f << sv_(j, i) << " ";
    }
    f << std::endl;
  }

  f.close();
}

/**
 * Load SVM model file
 *
 * @param: type id of the learner
 * @param: name of the model file
 */
/* TODO: use XML */
template<typename TKernel>
void SVM<TKernel>::LoadModel_(size_t learner_typeid, std::string model_filename) {
  if (learner_typeid == 0) { /* SVM_C */
    train_labels_list_.clear();
    train_labels_list_.reserve(num_classes_); /* get labels list from the model file */
  }

  /* load model file */
  FILE *fp = fopen(model_filename.c_str(), "r");
  if (fp == NULL) {
    Log::Fatal << "Cannot open SVM model file '" << model_filename << "'."
        << std::endl;
  }

  char cmd[80];
  size_t i, j;
  size_t temp_d;
  double temp_f;
  models_.reserve(num_models_+models_.size());

  for (i = 0; i < num_models_; i++) {
    models_.push_back(*new SVM_MODELS);
  }

  while (1) {
    FSCANF_CHECK(fp,"%80s",cmd);
    if(strcmp(cmd,"svm_type") == 0) {
      FSCANF_CHECK(fp,"%80s", cmd);

      if (strcmp(cmd,"SVM_C") == 0)
        learner_typeid_ = 0;
      else if (strcmp(cmd,"SVM_R") == 0)
        learner_typeid_ = 1;
      else if (strcmp(cmd,"SVM_DE") == 0)
        learner_typeid_ = 2;

    } else if (strcmp(cmd, "total_num_sv") == 0) {
      FSCANF_CHECK(fp,"%zu",&total_num_sv_);
    } else if (strcmp(cmd, "num_classes") == 0) { /* for SVM_C */
      FSCANF_CHECK(fp,"%zu",&num_classes_);
    } else if (strcmp(cmd, "labels") == 0) {
      for (i = 0; i<num_classes_; i++) {
        FSCANF_CHECK(fp,"%lf",&temp_f);
        train_labels_list_[i] = temp_f;
      }
    } else if (strcmp(cmd, "sv_list_startpos") == 0) {
      for (i= 0; i < num_classes_; i++) {
        FSCANF_CHECK(fp,"%zu",&temp_d);
        sv_list_startpos_[i]= temp_d;
      }
    } else if (strcmp(cmd, "sv_list_ct") == 0) {
      for (i = 0; i < num_classes_; i++) {
        FSCANF_CHECK(fp,"%zu",&temp_d);
        sv_list_ct_[i]= temp_d;
      }
    } else if (strcmp(cmd, "sv_index") == 0) { /* for SVM_R */
      for (i = 0; i < total_num_sv_; i++) {
        FSCANF_CHECK(fp,"%zu",&temp_d);
        sv_index_.push_back(temp_d);
      }
    } else if (strcmp(cmd, "kernel_name") == 0) { /* load kernel info */
      /* Switch to iostream file input instead? =/ */
      char in[81];
      FSCANF_CHECK(fp,"%80s",in);
      param_.kernelname_ = in;
    } else if (strcmp(cmd, "kernel_typeid") == 0) {
      FSCANF_CHECK(fp,"%zu",&param_.kerneltypeid_);
    } else if (strcmp(cmd, "sigma") == 0) {
      FSCANF_CHECK(fp,"%lf",&param_.kernel_.kpara_[0]); /* for gaussian kernels only */
    } else if (strcmp(cmd, "gamma") == 0) {
      FSCANF_CHECK(fp,"%lf",&param_.kernel_.kpara_[1]); /* for gaussian kernels only */
    } else if (strcmp(cmd, "bias") == 0) { /* load bias */
      for (i = 0; i < num_models_; i++) {
        FSCANF_CHECK(fp,"%lf",&temp_f);
        models_[i].bias_= temp_f;
      }
      break;
    }
  }

  /* load coefficients and support vectors */
  sv_coef_.set_size(num_classes_ - 1, total_num_sv_);
  sv_coef_.zeros();
  sv_.set_size(num_features_, total_num_sv_);
  while (1) {
    FSCANF_CHECK(fp,"%80s",cmd);
    if (strcmp(cmd, "SV_coefs") == 0) {
      for (i = 0; i < total_num_sv_; i++) {
        for (j = 0; j < num_classes_-1; j++) {
          FSCANF_CHECK(fp,"%lf",&temp_f);
          sv_coef_(j, i) = temp_f;
        }
      }
    } else if (strcmp(cmd, "SVs") == 0) {
      for (i = 0; i < total_num_sv_; i++) {
        for (j = 0; j < num_features_; j++) {
          FSCANF_CHECK(fp,"%lf",&temp_f);
          sv_(j, i) = temp_f;
        }
      }
      break;
    }
  }

  fclose(fp);
}

template<typename TKernel>
std::vector<int>& SVM<TKernel>::getLabels (arma::mat& data, std::vector<int>& labels) {
  /* we'll assume that the last column of the data contains labels;
   * another useful idea would be to state explicitly in the file the number of labels */
  size_t n_rows = data.n_rows;
  std::set<int> labelSet;
  for (size_t n = 0; n < data.n_cols; n++) {
    labelSet.insert((int) (data(n_rows - 1, n)));
  }

  for (std::set<int>::iterator it = labelSet.begin(); it != labelSet.end(); ++it) {
    labels.push_back(*it);
  }

  std::sort(labels.begin(), labels.end());
  return labels;
}

template<typename TKernel>
void SVM<TKernel>::reorderDataByLabels(arma::mat& data) {
  for (std::vector<int>::iterator it = train_labels_list_.begin(); it != train_labels_list_.end(); ++it) {
    train_labels_count_.push_back(0);
    train_labels_startpos_.push_back(0);
  }

  size_t currentIndex = 0;
  size_t searchIndex = 0;
  size_t columnCount = data.n_cols;
  size_t rowCount = data.n_rows;

  for (size_t iterator = 0; iterator < train_labels_list_.size(); ++iterator) {
    int currentLabel = train_labels_list_[iterator];
    bool foundLabel = false;

    for (searchIndex = currentIndex; searchIndex < columnCount; ++searchIndex) {
      if (currentLabel == (int) (data(rowCount - 1, searchIndex))) {
        if (!foundLabel) {
          train_labels_startpos_[iterator] = currentIndex;
          foundLabel = true;
        }

        /* swap the columns */
        if (searchIndex != currentIndex) {
          data.swap_cols (currentIndex, searchIndex);
        }

        ++(train_labels_count_[iterator]);
        ++currentIndex;
      }
    }
  }
}

template<typename TKernel>
arma::mat& SVM<TKernel>::buildDataByLabels (size_t i, size_t j, arma::mat& data, arma::mat& returnData) {
  size_t countForI = train_labels_count_[i];
  size_t countForJ = train_labels_count_[j];
  size_t startPosForI = train_labels_startpos_[i];
  size_t startPosForJ = train_labels_startpos_[j];

  returnData = arma::mat (data.n_rows, countForI + countForJ);

  size_t columnCounter = 0;
  for (size_t countI = startPosForI; countI < startPosForI + countForI; countI++) {
    for (size_t rowCounter = 0; rowCounter < data.n_rows - 1; rowCounter++) {
      returnData(rowCounter, columnCounter) = data(rowCounter, countI);
    }

    returnData(data.n_rows - 1, columnCounter) = 1.0;
    columnCounter++;
  }

  for (size_t countJ = startPosForJ; countJ < startPosForJ + countForJ; countJ++) {
    for (size_t rowCounter = 0; rowCounter < data.n_rows - 1; rowCounter++) {
      returnData(rowCounter, columnCounter) = data(rowCounter, countJ);
    }

    returnData(data.n_rows - 1, columnCounter) = -1.0;
    columnCounter++;
  }

  return returnData;
}

}; // namespace svm
}; // namespace mlpack

#endif
