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
 * @see opt_smo.h
 */
#include <mlpack/core.h>
#include "svm.h"

using std::string;
using std::vector;

PROGRAM_INFO("SVM", "These are the implementations for Support Vector\
 Machines, including Multiclass classification, Regression, and One Class SVM", "svm");

using namespace mlpack;
using namespace mlpack::svm;

/**
* Multiclass SVM classification/ SVM regression - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);
  srand(time(NULL));

  string mode = CLI::GetParam<std::string>("svm/mode");
  string kernel = CLI::GetParam<std::string>("svm/kernel");
  string learner_name = CLI::GetParam<std::string>("svm/learner_name");
  size_t learner_typeid = 0;

  if (learner_name == "svm_c") { // Support Vector Classfication
    learner_typeid = 0;
  } else if (learner_name == "svm_r") { // Support Vector Regression
    learner_typeid = 1;
  } else if (learner_name == "svm_de") { // One Class Support Vector Machine
    learner_typeid = 2;
  } else {
    Log::Fatal << "--svm/learner_name: Unknown learner name (valid: 'svm_c',"
        " 'svm_r', 'svm_de')." << std::endl;
  }

  // TODO: more kernels to be supported

//  /* Cross Validation Mode, need cross validation data */
//  if(mode == "cv") {
//    fprintf(stderr, "SVM Cross Validation... \n");
//
//    /* Load cross validation data */
//    Dataset cvset;
//    if (LoadData(&cvset, "cv_data") == 0)
//    return 1;
//
//    if (kernel == "linear") {
//      GeneralCrossValidator< SVM<SVMLinearKernel> > cross_validator;
//      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds, need k_cv>1 */
//      cross_validator.Init(learner_typeid, CLI::GetParam<int>("svm/k_cv"), &cvset, NULL, "svm");
//      /* k_cv folds cross validation; (true): do training set permutation */
//      cross_validator.Run(true);
//      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
//    }
//    else if (kernel == "gaussian") {
//      GeneralCrossValidator< SVM<SVMRBFKernel> > cross_validator;
//      /* Initialize n_folds_, confusion_matrix_; k_cv: number of cross-validation folds */
//      cross_validator.Init(learner_typeid, CLI::GetParam<int>("svm/k_cv"), &cvset, NULL, "svm");
//      /* k_cv folds cross validation; (true): do training set permutation */
//      cross_validator.Run(true);
//      //cross_validator.confusion_matrix().PrintDebug("confusion matrix");
//    }
//  }
  /* Training Mode, need training data | Training + Testing(online) Mode, need training data + testing data */

  if (mode == "train" || mode == "train_test") {
    arma::mat dataSet;
    std::string trainFile = CLI::GetParam<std::string>("svm/train_data");
    Log::Info << "Training SVM..." << std::endl;

    /* Load training data */
    if (dataSet.load(trainFile.c_str()) == false)
      return 1;

    /* Begin SVM Training | Training and Testing */
    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.InitTrain(learner_typeid, dataSet);
      /* training and testing, thus no need to load model from file */
      if (mode == "train_test") {
        Log::Info << "Making predictions with SVM model..." << std::endl;

        /* Load testing data */
        arma::mat dataSet;
        std::string testFile = CLI::GetParam<std::string>("svm/test_data");
        if (dataSet.load(testFile.c_str()) == false)
          return 1;

        svm.BatchPredict(learner_typeid, dataSet, "predicted_values");
      }
    } else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.InitTrain(learner_typeid, dataSet);
      /* training and testing, thus no need to load model from file */
      if (mode == "train_test") {
        Log::Info << "Making predictions with SVM model..." << std::endl;

        /* Load testing data */
        arma::mat dataSet;
        std::string testFile = CLI::GetParam<std::string>("svm/test_data");
        if (dataSet.load(testFile.c_str()) == false)
          return 1;

        svm.BatchPredict(learner_typeid, dataSet, "predicted_values"); // TODO:param_req
      }
    }
  }

  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode == "test") {
    Log::Info << "Making predictions with SVM model..." << std::endl;

    /* Load testing data */
    arma::mat dataSet;
    std::string testFile = CLI::GetParam<std::string>("svm/test_data");
    if (dataSet.load(testFile.c_str()) == false)
      return 1;

    /* Begin Prediction */
    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.Init(learner_typeid, dataSet);
      svm.LoadModelBatchPredict(learner_typeid, dataSet, "svm_model", "predicted_values"); // TODO:param_req
    } else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.Init(learner_typeid, dataSet);
      svm.LoadModelBatchPredict(learner_typeid, dataSet, "svm_model", "predicted_values"); // TODO:param_req
    }
  }

  return 0;
}
