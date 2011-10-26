/**
 * @file nnsvm.cc
 *
 * This file contains main routines for performing
 * NNSVM (Non-Negativity constrained SVM) training
 * NNSMO algorithm is employed.
 *
 * It currently support "train", "train-test", "test" mode with "linear" kernel
 * Example:
 *  nnsvm --mode=train --train_data=toy1.csv --kernel=linear --c=10.0 --eps=0.000001 --max_iter=1000
 *  nnsvm --mode=train_test --train_data=toy1.csv --test_data=toy2.csv --kernel=linear --c=10.0 --max_iter=1000
 *  nnsvm --mode=test --train_data=toy2.csv --kernel=linear
 *
 * @see nnsvm.h
 * @see nnsmo.h
 */
#include <iostream>
#include "nnsvm.hpp"
#include <mlpack/core.h>
#include <mlpack/core/kernels/linear_kernel.hpp>

PARAM_STRING_REQ("mode", "operating mode: train, train_test, or test", "nnsvm");
PARAM_STRING_REQ("kernel", "kernel type: linear (currently supported)", "nnsvm");
PARAM_STRING_REQ("train_data", "name of the file containing the training data", "nnsvm");

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::nnsvm;

/**
* NNSVM training - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);
  std::string mode = CLI::GetParam<std::string>("nnsvm/mode");
  std::string kernel = CLI::GetParam<std::string>("nnsvm/kernel");

  /* Training Mode, need training data */
  if (mode == "train" || mode == "train_test")
  {
    Log::Debug << "Non-Negativity Constrained SVM Training... " << std::endl;

    std::string trainFile = CLI::GetParam<std::string>("nnsvm/train_data");
    // Load training data
    arma::mat dataSet;
    if (dataSet.load(trainFile.c_str()) == false) // TODO:param_req
    {
      /* TODO: eventually, we need better exception handling */
      Log::Debug << "Could not open " << trainFile << " for reading" << std::endl;
      return 1;
    }

    // Begin NNSVM Training
    if (kernel == "linear")
    {
      NNSVM<LinearKernel> nnsvm;

      nnsvm.InitTrain(dataSet, 2,
          (int) CLI::GetParam<double>("nnsvm/c"),
          (int) CLI::GetParam<double>("nnsvm/b"),
          CLI::GetParam<double>("nnsvm/eps"),
          CLI::GetParam<int>("nnsvm/max_iter"));

      CLI::StartTimer("nnsvm/nnsvm_train");
      Log::Debug << "nnsvm_train_time" << CLI::GetParam<timeval>("nnsvm/nnsvm_train").tv_usec / 1e6 << std::endl;
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test")
      {
        Log::Debug << "Non-Negativity SVM Classifying... " << std::endl;
        /* Load testing data */
        std::string testFile = CLI::GetParam<std::string>("nnsvm/test_data");
        arma::mat testset;
        if (testset.load(testFile.c_str()) == false)// TODO:param_req
        {
          /* TODO: eventually, we need better exception handling */
          Log::Debug << "Could not open " << testFile << " for reading" <<
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
    Log::Debug << "Non-Negativity Constrained SVM Classifying... " << std::endl;

    /* Load testing data */
    arma::mat testset;

    /* Begin Classification */
    if (kernel == "linear")
    {
      NNSVM<LinearKernel> nnsvm;
      nnsvm.Init(testset, 2);
      nnsvm.LoadModelBatchClassify(testset, "nnsvm_model", "testlabels"); // TODO:param_req
    }
  }
  return 0;
}
