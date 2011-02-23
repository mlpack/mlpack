/**
 * @author Hua Ouyang
 *
 * @file regmin.cc
 *
 * This file contains main routines for performing Regularized Risk Minimizations.
 * Sparse matrix vector manipulations are implemented.
 *
 * It provides four modes:
 * "cv": cross validation;
 * "train": model training
 * "train_test": training and then online batch testing; 
 * "test": offline batch testing.
 *
 * Please refer to README for detail description of usage and examples.
 *
 * @see regmin.h
 * @see opt_smo.h
 * @see opt_sgd.h
 */

#include <errno.h>
#include "regmin.h"

char *line = NULL;
index_t max_line_length; // buffer size for read a line

Dataset_sl train_set;
NZ_entry *train_nz_pool; // a pool for all non-zero entries in the training set
Dataset_sl test_set;
NZ_entry *test_nz_pool; // a pool for all non-zero entries in the testing set


static char *ReadLine(FILE *fp_in) {
  index_t length;
  if ( fgets(line, max_line_length, fp_in)==NULL ) {
    return NULL;
  }
  while ( strrchr(line,'\n')==NULL ) {
    max_line_length *= 2;
    line = (char *) realloc(line, max_line_length);
    length = (index_t) strlen(line);
    if ( fgets(line+length, max_line_length-length, fp_in)==NULL ) {
      break;
    }
  }
  return line;
}

int ReadData(Dataset_sl &dataset, struct NZ_entry *nz_pool, FILE *fp) {
  index_t max_index, inst_max_index, i, j;
  char *endptr, *label, *index, *value;
  
  max_index = 0;
  j = 0;
  for (i=0; i<dataset.n_points; i++) {
    inst_max_index = -1;
    ReadLine(fp);
    dataset.x[i] = &nz_pool[j];
    label = strtok(line, " \t");
    dataset.y[i] = strtod(label, &endptr);
    if (endptr == label) {
      printf("No values found for a data point at line %d\n", i+1);
      return 0;
    }
    while (1) {
      index = strtok(NULL, ":");
      value = strtok(NULL, " \t");
      if (value == NULL) {
	break;
      }
      errno = 0;
      // in svmlight's data format, feature index begins from 1, not 0
      nz_pool[j].index = (index_t) strtol(index, &endptr, 10) - 1;
      if (endptr == index || errno !=0 || *endptr !='\0' || nz_pool[j].index<=inst_max_index) {
	printf("No values found for a data point at line %d\n", i+1);
	return 0;
      }
      else {
	inst_max_index = nz_pool[j].index;
      }
      errno = 0;
      nz_pool[j].value = strtod(value, &endptr);
      if ( endptr == value || errno !=0 || (*endptr!='\0' && !isspace(*endptr)) ) {
	printf("No values found for a data point at line %d\n", i+1);
	return 0;
      }
      j++;
    }
    if (inst_max_index > max_index) {
      max_index = inst_max_index;
    }
    nz_pool[j++].index = -1; // an indicator of the end of a data point
  }
  dataset.n_features = max_index + 1;
  return 1;
}

/**
 * Initialize sparse training dataset from a file
 *
 * @param: the training set
 * @param: the training filename
 */
int InitTrainsetFromFile(String param) {
  if (fx_param_exists(NULL, param)) {
    String train_filename = fx_param_str_req(NULL, param);
    FILE *fp = fopen(train_filename, "r");
    if (fp == NULL) {
      fprintf(stderr, "Cannot open the specified training file!!!\n");
      return 0;
    }
    else {
      index_t num_nz_entries = 0;
      train_set.n_points= 0;
      
      // count # of data points, # of non-zero entries
      while ( ReadLine(fp)!= NULL ) {
	// skip the lable
	char *p = strtok(line, " \t");
	// count # of features
	while (1) {
	  p = strtok(NULL, " \t");
	  if (p == NULL || *p == '\n') {
	    break;
	  }
	  num_nz_entries ++;
	}
	num_nz_entries ++; // add an indicator for the end of this data point
	train_set.n_points ++;
      }
      rewind(fp);

      train_set.y = Malloc(double, train_set.n_points);
      train_set.x = Malloc(struct NZ_entry *, train_set.n_points);
      train_nz_pool = Malloc(struct NZ_entry, num_nz_entries);
      
      if (ReadData(train_set, train_nz_pool, fp)) {
	fclose(fp);
	return 1;
      }
      else {
	free(train_set.y);
	free(train_set.x);
	free(train_nz_pool);
	fclose(fp);
	fprintf(stderr, "Errors in training file format!!!\n");
	return 0;
      }
    }
  }
  else {
    fprintf(stderr, "No training filename specified !!!\n");
    return 0;
  }
}

/**
 * Initialize sparse testing dataset from a file
 *
 * @param: the testing set
 * @param: the testing filename
 */
int InitTestsetFromFile(String param) {
  if (fx_param_exists(NULL, param)) {
    String test_filename = fx_param_str_req(NULL, param);
    FILE *fp = fopen(test_filename, "r");
    if (fp == NULL) {
      fprintf(stderr, "Cannot open the specified testing file!!!\n");
      return 0;
    }
    else {
      index_t num_nz_entries = 0;
      test_set.n_points= 0;
      
      // count # of data points, # of non-zero entries
      while ( ReadLine(fp)!= NULL ) {
	// skip the lable
	char *p = strtok(line, " \t");
	// count # of features
	while (1) {
	  p = strtok(NULL, " \t");
	  if (p == NULL || *p == '\n') {
	    break;
	  }
	  num_nz_entries ++;
	}
	num_nz_entries ++; // add an indicator for the end of this data point
	test_set.n_points ++;
      }
      rewind(fp);
      
      test_set.y = Malloc(double, test_set.n_points);
      test_set.x = Malloc(struct NZ_entry *, test_set.n_points);
      test_nz_pool = Malloc(struct NZ_entry, num_nz_entries);
      
      if (ReadData(test_set, test_nz_pool, fp)) {
	fclose(fp);
	return 1;
      }
      else {
	free(test_set.y);
	free(test_set.x);
	free(test_nz_pool);
	fclose(fp);
	fprintf(stderr, "Errors in testing file format!!!\n");
	return 0;
      }
    }
  }
  else {
    fprintf(stderr, "No testing filename specified!!!\n");
    return 0;
  }
}


/**
* Multiclass SVM classification/ SVM regression - Main function
*
* @param: argc
* @param: argv
*/
int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);

  //srand(time(NULL));

  String mode = fx_param_str_req(NULL, "mode");
  String kernel = fx_param_str_req(NULL, "kernel");
  String learner_name = fx_param_str_req(NULL,"learner_name");
  int learner_typeid;
  
  if (learner_name == "svm_c") { // Support Vector Classfication
    learner_typeid = 0;
  }
  else if (learner_name == "svm_r") { // Support Vector Regression
    learner_typeid = 1;
  }
  else if (learner_name == "svm_q") { // Support Vector Quantile Estimation
    learner_typeid = 2;
  }
  else {
    fprintf(stderr, "Unknown support vector learner name!!!\n");
    return 0;
  }

  max_line_length = 1024;
  line = Malloc(char, max_line_length);

  /* Training Mode, need training data | Training + Testing(online) Mode, need training data + testing data */
  if (mode=="train" || mode=="train_test"){
    fprintf(stderr, "SVM Training... \n");

    /* Load training data */
    if (InitTrainsetFromFile("train_data") == 0) {
      free(line);
      exit(1);
    }
    
    /* Begin SVM Training | Training and Testing */
    datanode *svm_module = fx_submodule(fx_root, "svm");
    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.InitTrain(learner_typeid, train_set, svm_module);

      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "SVM Predicting... \n");
	/* Load testing data */
	if (InitTestsetFromFile("test_data") == 0) {
	  free(line);
	  exit(1);
	}
	svm.BatchPredict(learner_typeid, test_set, "predicted_values");
	free(test_set.y);
	free(test_set.x);
	free(test_nz_pool);
      }

      free(train_set.y);
      free(train_set.x);
      free(train_nz_pool);
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.InitTrain(learner_typeid, train_set, svm_module);

      /* training and testing, thus no need to load model from file */
      if (mode=="train_test"){
	fprintf(stderr, "SVM Predicting... \n");
	/* Load testing data */
	if (InitTestsetFromFile("test_data") == 0) {
	  free(line);
	  exit(1);
	}
	svm.BatchPredict(learner_typeid, test_set, "predicted_values");
	free(test_set.y);
	free(test_set.x);
	free(test_nz_pool);
      }

      free(train_set.y);
      free(train_set.x);
      free(train_nz_pool);
    }
  }
  /* Testing(offline) Mode, need loading model file and testing data */
  else if (mode=="test") {
    fprintf(stderr, "SVM Predicting... \n");

    /* Load testing data */
    if (InitTestsetFromFile("test_data") == 0) {
      free(line);
      exit(1);
    }
    
    /* Begin Prediction */
    datanode *svm_module = fx_submodule(fx_root, "svm");

    if (kernel == "linear") {
      SVM<SVMLinearKernel> svm;
      svm.Init(learner_typeid, test_set, svm_module); 
      svm.LoadModelBatchPredict(learner_typeid, test_set, "svm_model", "predicted_values");
      free(test_set.y);
      free(test_set.x);
      free(test_nz_pool);
    }
    else if (kernel == "gaussian") {
      SVM<SVMRBFKernel> svm;
      svm.Init(learner_typeid, test_set, svm_module); 
      svm.LoadModelBatchPredict(learner_typeid, test_set, "svm_model", "predicted_values");
      free(test_set.y);
      free(test_set.x);
      free(test_nz_pool);
    }
  }
  free(line);
  fx_done(NULL);
}

