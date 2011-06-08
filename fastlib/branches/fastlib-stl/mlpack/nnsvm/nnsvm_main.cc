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

/**
* NNSVM training - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[])
{
  fx_init(argc, argv, NULL);

  std::string mode = fx_param_str_req(NULL, "mode");
  std::string kernel = fx_param_str_req(NULL, "kernel");

  /* Training Mode, need training data */
  if (mode == "train" || mode == "train_test")
  {
    std::cerr << "Non-Negativity Constrained SVM Training... \n";

    std::string trainFile = fx_param_str_req(NULL, "train_data");
    // Load training data
    arma::mat dataSet;
    if (data::Load(trainFile.c_str(), dataSet) == SUCCESS_FAIL) // TODO:param_req
    {
      /* TODO: eventually, we need better exception handling */
      std::cerr << "Could not open " << trainFile << " for reading\n";
      return 1;
    }

    // Begin NNSVM Training
    datanode *nnsvm_module = fx_submodule(fx_root, "nnsvm");

    if (kernel == "linear")
    {
      NNSVM<SVMLinearKernel> nnsvm;

      nnsvm.InitTrain(dataSet, 2, nnsvm_module);

      fx_timer_start(NULL, "nnsvm_train");
      fx_timer *nnsvm_train_time = fx_get_timer(NULL, "nnsvm_train");
      std::cerr << "nnsvm_train_time" << nnsvm_train_time->total.micros / 1e6 << "\n";
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test")
      {
        std::cerr << "Non-Negativity SVM Classifying... \n";
        /* Load testing data */
        std::string testFile = fx_param_str_req(NULL, "test_data");
        arma::mat testset;
        if (data::Load(testFile.c_str(), testset) == SUCCESS_FAIL) // TODO:param_req
        {
          /* TODO: eventually, we need better exception handling */
          std::cerr << "Could not open " << testFile << " for reading\n";
          return 1;
        }
        nnsvm.BatchClassify(testset, "testlabels");
      }
    }
  }
  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode == "test")
  {
    std::cerr << "Non-Negativity Constrained SVM Classifying... \n";

    /* Load testing data */
    arma::mat testset;

    /* Begin Classification */
    datanode *nnsvm_module = fx_submodule(fx_root, "nnsvm");

    if (kernel == "linear")
    {
      NNSVM<SVMLinearKernel> nnsvm;
      nnsvm.Init(testset, 2, nnsvm_module);
      nnsvm.LoadModelBatchClassify(testset, "nnsvm_model", "testlabels"); // TODO:param_req
    }
  }
  fx_done(NULL);
}
