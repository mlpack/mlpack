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

#include "nnsvm.h"

/**
* Load data set from data file. 
*
* @param: the dataset
* @param: name of the data file to be loaded
*/
int LoadData(Dataset& dataset, std::string datafilename){
  if (fx_param_exists(NULL, datafilename.c_str())) {
    // if a data file is specified, use it.
    if (!PASSED(dataset.InitFromFile(fx_param_str_req(NULL, datafilename.c_str())))) {
      fprintf(stderr, "Couldn't open the data file.\n");
      return 0;
    }
  }
  return 1;
}

/**
* NNSVM training - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[]) {  

  fx_init(argc, argv, NULL);

  std::string mode = fx_param_str_req(NULL, "mode");
  std::string kernel = fx_param_str_req(NULL, "kernel");  

  /* Training Mode, need training data */  
  if (mode == "train" || mode == "train_test") {
    fprintf(stderr, "Non-Negativity Constrained SVM Training... \n");

    // Load training data
    Dataset trainset;
    if (LoadData(trainset, "train_data") == 0) // TODO:param_req
      return 1;
    
    // Begin NNSVM Training 
    datanode *nnsvm_module = fx_submodule(fx_root, "nnsvm");
    
    if (kernel == "linear") {
      NNSVM<SVMLinearKernel> nnsvm;
      nnsvm.InitTrain(trainset, 2, nnsvm_module);
  	
      fx_timer_start(NULL, "nnsvm_train");
      fx_timer *nnsvm_train_time = fx_get_timer(NULL, "nnsvm_train");
      fprintf(stderr, "nnsvm_train_time %g \n", nnsvm_train_time->total.micros / 1e6);
      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "Non-Negativity SVM Classifying... \n");
	/* Load testing data */
	Dataset testset;
	if (LoadData(testset, "test_data") == 0) // TODO:param_req
	  return 1;
	nnsvm.BatchClassify(testset, "testlabels");
      }
    }
  } 
  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode == "test") {
    fprintf(stderr, "Non-Negativity Constrained SVM Classifying... \n");

    /* Load testing data */
    Dataset testset;
    if (LoadData(testset, "test_data") == 0) // TODO:param_req
      return 1;

    /* Begin Classification */
    datanode *nnsvm_module = fx_submodule(fx_root, "nnsvm");

    if (kernel == "linear") {
      NNSVM<SVMLinearKernel> nnsvm;
      nnsvm.Init(testset, 2, nnsvm_module);  
      nnsvm.LoadModelBatchClassify(testset, "nnsvm_model", "testlabels"); // TODO:param_req
    }
  }

  fx_done(NULL);
}
