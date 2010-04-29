/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file crossvalidation.h
 *
 * Cross validation support.
 */

#ifndef DATA_CROSSVALIDATION
#define DATA_CROSSVALIDATION

#include <armadillo>

#include "dataset.h"

#include "../la/matrix.h"
#include "../fx/fx.h"

/**
 * Cross-validator for simple classifiers, integrating tightly with
 * FastExec.
 *
 * Cross-validation runs go under path you give it (kfold_fx_name),
 * by default "kfold".
 * Suppose the classifier you are using is "knn", which you specify
 * as classifier_fx_name.  KFold has its own k (the number of folds),
 * but KNN has its own idea of k (the number of nearest neighbors).
 * The results would look like the following:
 *
 * @code
 * /kfold/params/k 1                # number of folds
 * /kfold/params/dataset foo.csv    # number of folds
 * /kfold/params/n_points 15460     # dataset size
 * /kfold/params/n_features 5
 * /kfold/0/knn/params/k 5 
 * /kfold/0/params/fold 0
 * /kfold/0/results/n_correct 1234  # number of correct and incorrect per run
 * /kfold/0/results/n_incorrect 312
 * /kfold/0/results/p_correct .798
 * /kfold/1/params/fold 0
 * /kfold/1/knn/params/k 5
 * /kfold/1/results/n_correct 1324
 * /kfold/1/results/n_incorrect 222
 * /kfold/1/results/p_correct .856
 * ...
 * /kfold/results/n_correct 13123   # overall totals
 * /kfold/results/n_incorrect 2337
 * /kfold/results/p_correct .849
 * @endcode
 *
 * To do a plot of KNN k versus cross validation correctness, you would
 * use the following select strings:
 *
 * @code
 * /kfold/params/dataset      # the name of the dataset
 * /kfold/0/knn/params/k      # this ensures you'll get default params
 * /kfold
 * @endcode
 *
 * Before the cross-validator runs, it will copy parameters from the module
 * you specify -- if it is module_root, this will just take the original
 * command line parameters that are stored in "/params".  In the previous
 * example, the command line parameters from "/params/knn/" and
 * "/params/kfold/" are used.  These parameters are specified by the user
 * as "--params/knn/someparameter=3" or "--param/kfold/k=4" to set KNN's
 * "someparameter" to 3, and the cross-validator's number of folds to 4.
 *
 *
 * To build a classifier suitable for use with SimpleCrossValidator, you
 * must create a class with the following methods:
 *
 * @code
 * class MyClassifier {
 *   ...
 *   // Trains on the dataset specified.  n_classes is the number of class
 *   // labels.  Tweak parameters can be obtained from the "datanode" passed
 *   // using fx_param_int, fx_param_double, etc, but passing in "module" as
 *   // the first parameter instead of NULL.
 *   //
 *   void InitTrain(const Dataset& dataset, int n_classes, datanode *module);
 *   // For a test datum, returns the class label 0 <= label < n_classes
 *   int Classify(const Vector& test_datum);
 * };
 * @endcode
 */
template<class TClassifier>
class SimpleCrossValidator {
  FORBID_ACCIDENTAL_COPIES(SimpleCrossValidator);
  
 public:
  /** Typedef of internal classifier used. */
  typedef TClassifier Classifier;
  
 private:
  /** The dataset. */
  const Dataset *data_;
  /** The originating module. */
  datanode *root_module_;
  /** The fastexec module for cross validation and result storage. */
  datanode *kfold_module_;
  /** Number of folds. */
  int n_folds_;
  /** Number of labels. */
  int n_classes_;
  /** The FastExec name of the classifier. */
  const char *classifier_fx_name_;
  /** Total number correct classified. */
  index_t n_correct_;
  /** Confusion matrix. */
  arma::mat confusion_matrix_;
  
 public:
  SimpleCrossValidator() {}
  ~SimpleCrossValidator() {}
  
  /**
   * Uses FastExec to initialize this.
   *
   * See details about this class for more information.
   *
   * @param data_with_labels dataset with labels as the last feature
   * @param n_labels the number of labels (setting this to 0 means to
   *        automatically determine from the dataset); the labels must
   *        be integers from 0 to n_labels - 1
   * @param default_k the default number of folds (overridden by
   *        command-line parameter kfold/k)
   * @param module_root the fastexec module this is under (usually use fx_root)
   * @param classifier_fx_name short name to give it under fastexec
   * @param kfold_fx_name the fastexec name of the cross-validator
   */
  void Init(
      const Dataset *data_with_labels,
      int n_labels,
      int default_k,
      struct datanode *module_root,
      const char *classifier_fx_name,
      const char *kfold_fx_name = "kfold");
  
  /**
   * Runs cross-validation.
   *
   * @param randomized whether to use a random permutation of the data,
   *        or just to stride it
   */
  void Run(bool randomized = false);
  
  
  /** Gets the number correctly classified over all folds. */
  index_t n_correct() {
    return n_correct_;
  }
  
  /** Gets the number incorrect over all folds. */
  index_t n_incorrect() {
    return data_->n_points() - n_correct_;
  }
  
  /** Gets the portion calculated correct. */
  double portion_correct() {
    return n_correct_ * 1.0 / data_->n_points();
  }
  
  /**
   * Gets the confusion matrix.
   *
   * The element at row i column j is the number of training samples where
   * the actual classification is i but the predicted classification is j.
   */
  const arma::mat& confusion_matrix() const {
    return confusion_matrix_;
  }
  
  /** Gets the dataset. */
  const Dataset& data() const {
    return *data_;
  }

 private:
  void SaveTrainTest_(int i_fold,
      const Dataset& train, const Dataset& test) const;
};

template<class TClassifier>
void SimpleCrossValidator<TClassifier>::SaveTrainTest_(
    int i_fold,
    const Dataset& train, const Dataset& test) const {
  String train_name;
  String test_name;
  
  train_name.InitSprintf("train_%d.csv", i_fold);
  test_name.InitSprintf("test_%d.csv", i_fold);
  
  train.WriteCsv(train_name);
  test.WriteCsv(test_name);
}


template<class TClassifier>
void SimpleCrossValidator<TClassifier>::Init(
    const Dataset *data_with_labels,
    int n_labels,
    int default_k,
    struct datanode *module_root,
    const char *classifier_fx_name,
    const char *kfold_fx_name) {
  data_ = data_with_labels;
  
  if (n_labels <= 0) {
    const DatasetFeature *feature =
        &data_->info().feature(data_->n_features() - 1);
    DEBUG_ASSERT_MSG(feature->type() == DatasetFeature::NOMINAL,
        "Must specify number of classes/labels if the feature is not nominal.");
    n_classes_ = feature->n_values();
  } else {
    n_classes_ = n_labels;
  }
  
  root_module_ = module_root;
  kfold_module_ = fx_submodule(module_root, kfold_fx_name);
  classifier_fx_name_ = classifier_fx_name;
  
  n_folds_ = fx_param_int(kfold_module_, "k", default_k);
  
  DEBUG_ONLY(n_correct_ = BIG_BAD_NUMBER);
  
  confusion_matrix_.zeros(n_classes_, n_classes_);
}

template<class TClassifier>
void SimpleCrossValidator<TClassifier>::Run(bool randomized) {
  ArrayList<index_t> permutation;
  
  if (randomized) {
    math::MakeRandomPermutation(data_->n_points(), &permutation);
  } else {
    math::MakeIdentityPermutation(data_->n_points(), &permutation);
  }
  
  n_correct_ = 0;
  
  fx_timer_start(kfold_module_, "total");
  
  for (int i_fold = 0; i_fold < n_folds_; i_fold++) {
    Classifier classifier;
    Dataset test;
    Dataset train;
    index_t local_n_correct = 0;
    datanode *classifier_module = fx_copy_module(root_module_,
        classifier_fx_name_, "%s/%d/%s",
        kfold_module_->key, i_fold, classifier_fx_name_);
    datanode *foldmodule = fx_submodule(classifier_module, "..");

    data_->SplitTrainTest(n_folds_, i_fold, permutation, train, test);
    
    if (fx_param_bool(kfold_module_, "save", 0)) {
      SaveTrainTest_(i_fold, train, test);
    }
  
    VERBOSE_MSG(1, "cross: Training fold %d", i_fold);
    fx_timer_start(foldmodule, "train");
    classifier.InitTrain(train, n_classes_, classifier_module);
    fx_timer_stop(foldmodule, "train");
    
    fx_timer_start(foldmodule, "test");
    VERBOSE_MSG(1, "cross: Testing fold %d", i_fold);
    for (index_t i = 0; i < test.n_points(); i++) {
      Vector test_vector;
      
      test_vector.Init(test.n_features() - 1);
      for(int j = 0; j < test.n_features() - 1; j++)
        test_vector[j] = test.matrix()(i, j);
      
      int label_predict = classifier.Classify(test_vector);
      double label_expect_dbl = test.matrix()(i, test.n_features() - 1);
      int label_expect = int(label_expect_dbl);
      
      DEBUG_ASSERT(double(label_expect) == label_expect_dbl);
      DEBUG_ASSERT(label_expect < n_classes_);
      DEBUG_ASSERT(label_expect >= 0);
      DEBUG_ASSERT(label_predict < n_classes_);
      DEBUG_ASSERT(label_predict >= 0);
      
      if (label_expect == label_predict) {
        local_n_correct++;
      }
      
      confusion_matrix_(label_expect, label_predict) += 1;
    }
    fx_timer_stop(foldmodule, "test");
    
    fx_format_result(foldmodule, "n_correct", "%"LI"d",
        local_n_correct);
    fx_format_result(foldmodule, "n_incorrect", "%"LI"d",
        test.n_points() - local_n_correct);
    fx_format_result(foldmodule, "p_correct", "%.03f",
        local_n_correct * 1.0 / test.n_points());
    
    n_correct_ += local_n_correct;
  }
  fx_timer_stop(kfold_module_, "total");

  fx_format_result(kfold_module_, "n_points", "%"LI"d",
      data_->n_points());
  fx_format_result(kfold_module_, "n_correct", "%"LI"d",
      n_correct());
  fx_format_result(kfold_module_, "n_incorrect", "%"LI"d",
      n_incorrect());
  fx_format_result(kfold_module_, "p_correct", "%.03f",
      1.0 * portion_correct());
}






/** ALPHA VERSION, STILL UNDER CONSTRUCTION
 *
 * k-fold Cross-validator for general learners 
 * (can support Classification, Regression, Density Estimation and other learners)
 *
 * For classification, the Stratified Cross-Validation is used to ensure that 
 * approximately same portion of data (training/validation) are used for each class.
 *
 * For regression, density estimation and other learners, training and validation data 
 * are then drawn from the partitions of input dataset (may be firstly randomized if necessary).
 * 
 **/

template<class TLearner>
class GeneralCrossValidator {
  FORBID_ACCIDENTAL_COPIES(GeneralCrossValidator);
  
 public:
  /** Typedef of internal learners used. */
  typedef TLearner Learner;
  
 private:

  /** General parameters for the cross validator */
  /** 
   * Type id of the learner: 
   *  0:Classification;
   *  1:Regression
   *  2:density estimation;
   *  3:others
   *
   * Develpers may add more learner types if necessary
   */
  int learner_typeid_;
  /** Number of folds */
  int n_folds_;
  /** The input dataset */
  const Dataset *data_;
  /** Number of data points  */
  index_t num_data_points_;
  /** The originating module */
  datanode *root_module_;
  /** The fastexec module for cross validation and result storage */
  datanode *kfold_module_;
  /** The FastExec name of the learner */
  const char *learner_fx_name_;

  
  /** variables for type 0: classification ONLY */
  /** Number of labels */
  int clsf_n_classes_;
  /** Total number correct classified */
  index_t clsf_n_correct_;
  /** Confusion matrix */
  arma::mat clsf_confusion_matrix_;

  /** variables for type 1/2: regression, density estimation, etc. */
  /** mean squared error over all folds*/
  double msq_err_all_folds_;


 public:
  GeneralCrossValidator() {}
  ~GeneralCrossValidator() {}
  /**
   * Uses FastExec to initialize this.
   *
   * See details about this class for more information.
   *
   * @param learner_typeid type id of the learner; 0:classification 1:regression 2:density estimation;
   *        3:others
   * @param default_k the default number of folds (overridden by
   *        command-line parameter kfold/k)
   * @param data_input the input dataset (with labels in the last feature for classification case)
   * @param module_root the fastexec module this is under (usually use fx_root)
   * @param learner_fx_name short name to give it under fastexec
   * @param kfold_fx_name the fastexec name of the cross-validator
   */
  void Init(int learner_typeid,
	    int default_k,
	    const Dataset *data_input,
	    struct datanode *module_root,
	    const char *learner_fx_name,
	    const char *kfold_fx_name = "kfold");

  /** Gets the dataset. */
  const Dataset& data() const {
    return *data_;
  }

  /**
   * Runs cross-validation.
   *
   * @param randomized whether to use a random permutation of the data,
   *        or just to stride it
   */
  void Run(bool randomized);

  /** Functions for type 0: classification ONLY */
  /** Gets the number correctly classified over all folds. */
  index_t clsf_n_correct() {
    return clsf_n_correct_;
  }
  /** Gets the number incorrect over all folds */
  index_t clsf_n_incorrect() {
    return data_->n_points() - clsf_n_correct_;
  }
  /** Gets the portion calculated correct */
  double clsf_portion_correct() {
    return clsf_n_correct_ * 1.0 / data_->n_points();
  }
  /**
   * Gets the confusion matrix.
   *
   * The element at row i column j is the number of training samples where
   * the actual classification is i but the predicted classification is j.
   */
  const arma::mat& clsf_confusion_matrix() const {
    return clsf_confusion_matrix_;
  }


 private:
  /** Save the splited training and validation sets */
  void SaveTrainValidationSet_(int i_fold,
      const Dataset& train, const Dataset& validation) const;

  /** For classification ONLY*/
  /** Stratified spliting of cross validation set to ensure that approximately 
   * the same portion of data (training/validation) are used for each class */
  void StratifiedSplitCVSet_(int i_fold, index_t num_classes, ArrayList<index_t>& cv_labels_ct, 
			     ArrayList<index_t>& cv_labels_startpos, const ArrayList<index_t>& permutation, Dataset *train, Dataset *validation){
    // Begin stratified splitting for the i-th fold stratified CV
    index_t n_cv_features = data_->n_features();
    
    // detemine the number of data samples for training and validation according to i_fold
    index_t n_cv_validation, i_validation, i_train;
    n_cv_validation = 0;
    for (index_t i_classes = 0; i_classes < num_classes; i_classes++) {
      i_validation = 0;
      for (index_t j = 0; j < cv_labels_ct[i_classes]; j++) {
	if ((j - i_fold) % n_folds_ == 0) { // point for validation
	  i_validation++;
	}
      }
      n_cv_validation = n_cv_validation + i_validation;
    }
    index_t n_cv_train = num_data_points_ - n_cv_validation;
    train->InitBlank();
    train->info().InitContinuous(n_cv_features);
    train->matrix().set_size(n_cv_features, n_cv_train);

    validation->InitBlank();
    validation->info().InitContinuous(n_cv_features);
    validation->matrix().set_size(n_cv_features, n_cv_validation);

    // make training set and vaidation set by concatenation
    i_train = 0;
    i_validation = 0;
    for (index_t i_classes = 0; i_classes < num_classes; i_classes++) {
      for (index_t j = 0; j < cv_labels_ct[i_classes]; j++) {
	double *dest;
	if ((j - i_fold) % n_folds_ != 0) { // add to training set
	  dest = train->matrix().colptr(i_train);
	  i_train++;
	}
	else { // add to validation set
	  dest = validation->matrix().colptr(i_validation);
	  i_validation++;
	}
        memcpy(dest, data_->matrix().colptr(cv_labels_startpos[i_classes] + j),
          sizeof(double) * data_->matrix().n_rows);
      }
    }
  }

};
  
template<class TLearner>
void GeneralCrossValidator<TLearner>::SaveTrainValidationSet_(
    int i_fold, const Dataset& train, const Dataset& validation) const {
  String train_name;
  String validation_name;
  
  // save training and validation sets for this fold
  train_name.InitSprintf("cv_train_%d.csv", i_fold);
  validation_name.InitSprintf("cv_validation_%d.csv", i_fold);
  
  train.WriteCsv(train_name);
  validation.WriteCsv(validation_name);
}

template<class TLearner>
void GeneralCrossValidator<TLearner>::Init(
    int learner_typeid,
    int default_k,
    const Dataset *data_input,
    struct datanode *module_root,
    const char *learner_fx_name,
    const char *kfold_fx_name) {
  /** initialization for general parameters */
  learner_typeid_ = learner_typeid;
  data_ = data_input;

  root_module_ = module_root;
  kfold_module_ = fx_submodule(module_root, kfold_fx_name);
  n_folds_ = fx_param_int(kfold_module_, "k", default_k);
  learner_fx_name_ = learner_fx_name;

  /** initialization for type 0: classification ONLY */
  if(learner_typeid_ == 0) {
    // get the number of classes
    clsf_n_classes_ = data_->n_labels();
    clsf_n_correct_ = 0;
    // initialize confusion matrix
    clsf_confusion_matrix_.zeros(clsf_n_classes_, clsf_n_classes_);
  }
  else if (learner_typeid_ == 1 || learner_typeid_ == 2) {
    clsf_confusion_matrix_ = 0.0; /* 1x1 matrix */
    // initialize mean squared error over all folds
    msq_err_all_folds_ = 0.0;
  }

}

template<class TLearner>
void GeneralCrossValidator<TLearner>::Run(bool randomized) {  
  fx_timer_start(kfold_module_, "total");
  num_data_points_ = data_->n_points();

  /** for type 0: Classification ONLY */
  if (learner_typeid_ == 0) {
    // get label information
    /* list of labels, need to be integers. e.g. [0,1,2] for a 3-class dataset */
    ArrayList<double> cv_labels_list;
    /* array of label indices, after grouping. e.g. [c1[0,5,6,7,10,13,17],c2[1,2,4,8,9],c3[...]]*/
    ArrayList<index_t> cv_labels_index;
    /* counted number of label for each class. e.g. [7,5,8]*/
    ArrayList<index_t> cv_labels_ct;
    /* start positions of each classes in the cv label list. e.g. [0,7,12] */
    ArrayList<index_t> cv_labels_startpos;
    // Get label list and label indices from the cross validation data set
    index_t num_classes = data_->n_labels();

    cv_labels_list.Init();
    cv_labels_index.Init();
    cv_labels_ct.Init();
    cv_labels_startpos.Init();
    data_->GetLabels(cv_labels_list, cv_labels_index, cv_labels_ct, cv_labels_startpos);

    // randomize the original data set within each class if necessary
    ArrayList<index_t> permutation;

    if (randomized) {
      permutation.Init(num_data_points_);
      for (index_t i_classes=0; i_classes<num_classes; i_classes++) {
	ArrayList<index_t> sub_permutation; // within class permut indices
	math::MakeRandomPermutation(cv_labels_ct[i_classes], &sub_permutation);
	// use sub-permutation indicies to form the whole permutation
	if (i_classes==0){
	  for (index_t j=0; j<cv_labels_ct[i_classes]; j++)
	    permutation[cv_labels_startpos[i_classes]+j] = cv_labels_index[ sub_permutation[j] ];
	}
	else {
	  for (index_t j=0; j<cv_labels_ct[i_classes]; j++)
	    permutation[cv_labels_startpos[i_classes]+j] = cv_labels_index[ cv_labels_ct[i_classes-1]+sub_permutation[j] ];
	}
	sub_permutation.Clear();
      }
    } // e.g. [10,13,5,17,0,6,7,,4,9,8,1,2,,...]
    else {
      permutation.InitCopy(cv_labels_index, cv_labels_index.size()); // e.g. [0,5,6,7,10,13,17,,1,2,4,8,9,,...]
    }
    // begin CV
    for (int i_fold = 0; i_fold < n_folds_; i_fold++) {
      Learner classifier;
      Dataset train;
      Dataset validation;
      
      index_t local_n_correct = 0;
      datanode *learner_module = fx_copy_module(root_module_,
          learner_fx_name_, "%s/%d/%s",
          kfold_module_->key, i_fold, learner_fx_name_);
      datanode *foldmodule = fx_submodule(learner_module, "..");

      // Split labeled data sets according to i_fold. Use Stratified Cross-Validation to ensure 
      // that approximately the same portion of data (training/validation) are used for each class.
      StratifiedSplitCVSet_(i_fold, num_classes, cv_labels_ct, cv_labels_startpos, permutation, &train, &validation);
      if (fx_param_bool(kfold_module_, "save", 0)) {
	SaveTrainValidationSet_(i_fold, train, validation);
      }
      
      VERBOSE_MSG(1, "cross: Training fold %d", i_fold);
      fx_timer_start(foldmodule, "train");
      // training
      classifier.InitTrain(learner_typeid_, train, learner_module);
      fx_timer_stop(foldmodule, "train");

      // validation; measure method: percent of correctly classified validation samples
      fx_timer_start(foldmodule, "validation");
      VERBOSE_MSG(1, "cross: Validation fold %d", i_fold);

      for (index_t i = 0; i < validation.n_points(); i++) {
	Vector validation_vector;
        validation_vector.Init(validation.n_features() - 1);

        memcpy(validation_vector.ptr(), validation.matrix().colptr(i),
          sizeof(double) * (validation.n_features() - 1));
	
	// testing (classification)
	int label_predict = int(classifier.Predict(learner_typeid_, validation_vector));
	double label_expect_dbl = validation.matrix()(i, validation.n_features() - 1);
	int label_expect = int(label_expect_dbl);

	DEBUG_ASSERT(double(label_expect) == label_expect_dbl);
	DEBUG_ASSERT(label_expect < clsf_n_classes_);
	DEBUG_ASSERT(label_expect >= 0);
	DEBUG_ASSERT(label_predict < clsf_n_classes_);
	DEBUG_ASSERT(label_predict >= 0);
	
	if (label_expect == label_predict) {
	  local_n_correct++;
	}
	clsf_confusion_matrix_(label_expect, label_predict) += 1;
      }
      fx_timer_stop(foldmodule, "validation");

      fx_format_result(foldmodule, "local_n_correct", "%"LI"d", local_n_correct);
      fx_format_result(foldmodule, "local_n_incorrect", "%"LI"d", validation.n_points() - local_n_correct);
      fx_format_result(foldmodule, "local_p_correct", "%.03f", local_n_correct * 1.0 / validation.n_points());

      clsf_n_correct_ += local_n_correct;
    }
    fx_timer_stop(kfold_module_, "total");
    
    fx_format_result(kfold_module_, "n_points", "%"LI"d", num_data_points_);
    fx_format_result(kfold_module_, "n_correct", "%"LI"d", clsf_n_correct());
    fx_format_result(kfold_module_, "n_incorrect", "%"LI"d", clsf_n_incorrect());
    fx_format_result(kfold_module_, "p_correct", "%.03f", 1.0 * clsf_portion_correct());
  }
  /** For type 1:regression & 2:density estimation */
  else if (learner_typeid_ == 1 || learner_typeid_ == 2) {
    double accu_msq_err_all_folds = 0.0;

    // randomize the original data set if necessary
    ArrayList<index_t> permutation;  
    if (randomized) {
      math::MakeRandomPermutation(num_data_points_, &permutation);
    } else {
      math::MakeIdentityPermutation(num_data_points_, &permutation);
    }
    // begin CV
    for (int i_fold = 0; i_fold < n_folds_; i_fold++) {
      Learner learner;
      Dataset train;
      Dataset validation;
      
      double msq_err_local = 0.0;
      double accu_sq_err_local = 0.0;
      datanode *learner_module = fx_copy_module(root_module_,
          learner_fx_name_, "%s/%d/%s",
          kfold_module_->key, i_fold, learner_fx_name_);
      datanode *foldmodule = fx_submodule(learner_module, "..");
      
      // Split general data sets according to i_fold
      data_->SplitTrainTest(n_folds_, i_fold, permutation, train, validation);
      
      if (fx_param_bool(kfold_module_, "save", 0)) {
	SaveTrainValidationSet_(i_fold, train, validation);
      }
      
      VERBOSE_MSG(1, "cross: Training fold %d", i_fold);
      fx_timer_start(foldmodule, "train");
      // training
      learner.InitTrain(learner_typeid_, train, learner_module); // 0: dummy number of classes
      fx_timer_stop(foldmodule, "train");
      
      // validation
      fx_timer_start(foldmodule, "validation");
      VERBOSE_MSG(1, "cross: Validation fold %d", i_fold);
      for (index_t i = 0; i < validation.n_points(); i++) {
	Vector validation_vector;
        validation_vector.Init(validation.n_features() - 1);

        memcpy(validation_vector.ptr(), validation.matrix().colptr(i),
          sizeof(double) * (validation.n_features() - 1));
	
        // testing
	double value_predict = learner.Predict(learner_typeid_, validation_vector);
	double value_true = validation.matrix()(i, validation.n_features() - 1);
	double value_err = value_predict - value_true;
	
	// Calculate squared error: sublevel
	accu_sq_err_local  += pow(value_err, 2);
      }
      fx_timer_stop(foldmodule, "validation");
      
      msq_err_local = accu_sq_err_local / validation.n_points();
      fx_format_result(foldmodule, "local_msq_err", "%f", msq_err_local);

      accu_msq_err_all_folds += msq_err_local;
    }
    fx_timer_stop(kfold_module_, "total");
    
    // Calculate mean squared error: over all folds
    msq_err_all_folds_ = accu_msq_err_all_folds / n_folds_;
    fx_format_result(kfold_module_, "msq_err_all_folds", "%f", msq_err_all_folds_);
  }
  else {
    fprintf(stderr, "Other learner types or Unknown learner type id! Cross validation stops!\n");
    return;
  }
}

#endif
