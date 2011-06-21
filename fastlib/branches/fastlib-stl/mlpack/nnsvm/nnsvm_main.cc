/**
 * @file nnsvm.cc
 *
 * This file contains main routines for performing
 * NNSVM (Non-Negativity constrained SVM) training
 * NNSMO algorithm is employed.
 *
 * It currently support "train", "train-test", "test" mode with "linear" kernel
 * Example:
 *  nnsvm_main --mode=train --train_data=toy1.csv --kernel=linear --c=10.0 --eps=0.000001 --max_iter=1000
 *  nnsvm_main --mode=train_test --train_data=toy1.csv --test_data=toy2.csv --kernel=linear --c=10.0 --max_iter=1000
 *  nnsvm_main --mode=test --train_data=toy2.csv --kernel=linear
 *
 * @see nnsvm.h
 * @see nnsmo.h
 */

#include <iostream>
#include "nnsvm.h"
#include <fastlib/fx/io.h>

PARAM_STRING_REQ("mode", "Undocumented", "nnsvm");
PARAM_STRING_REQ("kernel", "Undocumented", "nnsvm");
PARAM_STRING_REQ("train_data", "Undocumented", "nnsvm");




using namespace mlpack;
using namespace mlpack::nnsvm;

/**
* NNSVM training - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[])
{
  IO::ParseCommandLine(argc, argv);
  std::string mode = IO::GetParam<std::string>("nnsvm/mode");
  std::string kernel = IO::GetParam<std::string>("nnsvm/kernel");

  /* Training Mode, need training data */
  if (mode == "train" || mode == "train_test")
  {
    IO::Debug << "Non-Negativity Constrained SVM Training... " << std::endl;

    std::string trainFile = IO::GetParam<std::string>("nnsvm/train_data");
    // Load training data
    arma::mat dataSet;
    if (data::Load(trainFile.c_str(), dataSet) == SUCCESS_FAIL) // TODO:param_req
    {
      /* TODO: eventually, we need better exception handling */
      IO::Debug << "Could not open " << trainFile << " for reading" << std::endl;
      return 1;
    }

    // Begin NNSVM Training
    if (kernel == "linear")
    {
      NNSVM<SVMLinearKernel> nnsvm;

      nnsvm.InitTrain(dataSet, 2,
          IO::GetParam<double>("nnsvm/c"),
          IO::GetParam<double>("nnsvm/b"),
          IO::GetParam<double>("nnsvm/eps"),
          IO::GetParam<double>("nnsvm/max_iter"));

      IO::StartTimer("nnsvm/nnsvm_train");
      IO::Debug << "nnsvm_train_time" << IO::GetParam<timeval>("nnsvm/nnsvm_train").tv_usec / 1e6 << std::endl;
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test")
      {
        IO::Debug << "Non-Negativity SVM Classifying... " << std::endl;
        /* Load testing data */
        std::string testFile = IO::GetParam<std::string>("nnsvm/test_data");
        arma::mat testset;
        if (data::Load(testFile.c_str(), testset) == SUCCESS_FAIL) // TODO:param_req
        {
          /* TODO: eventually, we need better exception handling */
          IO::Debug << "Could not open " << testFile << " for reading" <<
            std::endl;
          return 1;
        }
        nnsvm.BatchClassify(testset, "testlabels");
      }
    }
  }
  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode == "test")
  {
    IO::Debug << "Non-Negativity Constrained SVM Classifying... " << std::endl;

    /* Load testing data */
    arma::mat testset;

    /* Begin Classification */
    if (kernel == "linear")
    {
      NNSVM<SVMLinearKernel> nnsvm;
      nnsvm.Init(testset, 2);
      nnsvm.LoadModelBatchClassify(testset, "nnsvm_model", "testlabels"); // TODO:param_req
    }
  }
}
