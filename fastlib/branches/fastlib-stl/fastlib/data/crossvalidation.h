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
#include "../fx/io.h"

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
 *   int Classify(const vec& test_datum);
 * };
 * @endcode
 */
template<class TClassifier>
class SimpleCrossValidator {
 public:
  /** Typedef of internal classifier used. */
  typedef TClassifier Classifier;
  
 private:
  /** The dataset. */
  const Dataset *data_;
  /** The originating module. */
  std::string root_path;
  /** The fastexec module for cross validation and result storage. */
  std::string kfold_path;
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
      const char* path_to_root,
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


 /**
  * Copies a certain folder structure to another folder.
  *
  * Will copy the following parameters:
  * --timers: train, test, validation
  * --ints: local_n_correct, local_n_incorrect
  * --doubles: local_p_correct
  *
  */
 std::string CopyFolder(std::string& root, const char* classifier, 
                          int i_num, std::string& kfold) {
  //Take root, prepend it to the source path (root+classifier) 
  std::string source_path = root+classifier;

  //Take root, prepend it to the destination path (kfold/i_fold/class)
  std::string dest_path = kfold_path + 
                          std::string("/") + 
                          std::string(i_num) + 
                          std::string("/") + 
                          std::string(classifier_fx_name_);

  mlpack::IO::GetParam<timeval>((dest_path+"train").c_str()) =
    mlpack::IO::GetParam<timeval>((root+"train").c_str());
  mlpack::IO::GetParam<timeval>((dest_path+"test").c_str()) =
    mlpack::IO::GetParam<timeval>((root+"train").c_str());
  mlpack::IO::GetParam<timeval>((dest_path+"validation").c_str()) =
    mlpack::IO::GetParam<timeval>((root+"train").c_str());
  mlpack::IO::GetParam<int>((dest_path+"local_n_correct").c_str()) =
    mlpack::IO::GetParam<timeval>((root+"train").c_str());
  mlpack::IO::GetParam<int>((dest_path+"local_in_correct").c_str()) =
    mlpack::IO::GetParam<timeval>((root+"train").c_str());
  mlpack::IO::GetParam<double>((dest_path+"local_p_correct").c_str()) =
    mlpack::IO::GetParam<timeval>((root+"train").c_str());

   return dest_path; 
 }

 private:
  void SaveTrainTest_(int i_fold,
      const Dataset& train, const Dataset& test) const;
};

template<class TClassifier>
void SimpleCrossValidator<TClassifier>::SaveTrainTest_(
    int i_fold,
    const Dataset& train, const Dataset& test) const {
  std::string train_name;
  std::string test_name;
  std::ostringstream o;
  
  o << i_fold;
  train_name = "train_" + o.str() + ".csv";
  test_name = "test_" + o.str() + ".csv";
  
  train.WriteCsv(train_name);
  test.WriteCsv(test_name);
}

template<class TClassifier>
void SimpleCrossValidator<TClassifier>::Init(
    const Dataset *data_with_labels,
    int n_labels,
    int default_k,
    const char* path_to_root,
    const char *classifier_fx_name,
    const char *kfold_name) {
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
  
  root_path = path_to_root;
  kfold_path = kfold_name;
  classifier_fx_name_ = classifier_fx_name;

  if(!mlpack::IO::HasParam((kfold_path+"k").c_str()))
    mlpack::IO::GetParam<int>((kfold_path+"k").c_str()) = default_k;  

  n_folds_ = mlpack::IO::GetParam<int>((kfold_path+"k").c_str());
  
  DEBUG_ONLY(n_correct_ = BIG_BAD_NUMBER);
  
  confusion_matrix_.zeros(n_classes_, n_classes_);
}

template<class TClassifier>
void SimpleCrossValidator<TClassifier>::Run(bool randomized) {
  std::vector<index_t> permutation;
  
  if (randomized) {
    math::MakeRandomPermutation(data_->n_points(), &permutation.front());
  } else {
    math::MakeIdentityPermutation(data_->n_points(), &permutation.front());
  }
  
  n_correct_ = 0;
  
  mlpack::IO::StartTimer((kfold_path+"total").c_str());
  
  for (int i_fold = 0; i_fold < n_folds_; i_fold++) {
    Classifier classifier;
    Dataset test;
    Dataset train;
    index_t local_n_correct = 0;

    /** COPY MODULE 
    datanode *classifier_module = fx_copy_module(root_module_,
        classifier_fx_name_, "%s/%d/%s",
        kfold_module_->key, i_fold, classifier_fx_name_);
    datanode *foldmodule = fx_submodule(classifier_module, "..");
    //foldpath = blah
     COPY MODULE **/

    
    std::string fold_path = CopyFolder(root_path, classifier_fx_name_, 
                              i_fold, kfold_path);

    data_->SplitTrainTest(n_folds_, i_fold, permutation, train, test);
    
    if (mlpack::IO::HasParam((kfold_path+"save").c_str())) {
      SaveTrainTest_(i_fold, train, test);
    }
  
    mlpack::IO::StartTimer((fold_path+"train").c_str());
    classifier.InitTrain(train, n_classes_, NULL);
    mlpack::IO::StopTimer((fold_path+"train").c_str());
    
    mlpack::IO::StartTimer((fold_path+"test").c_str());
    for (index_t i = 0; i < test.n_points(); i++) {
      arma::vec test_vector(test.n_features() - 1);
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
    mlpack::IO::StopTimer((fold_path+"test").c_str());
    
    mlpack::IO::GetParam<int>((fold_path+"n_correct").c_str()) = 
      local_n_correct;
    mlpack::IO::GetParam<int>((fold_path+"n_incorrect").c_str()) = 
      test.n_points() - local_n_correct;
    mlpack::IO::GetParam<double>((fold_path+"p_correct").c_str()) = 
      local_n_correct * 1.0 / test.n_points();
    n_correct_ += local_n_correct;

  }
  mlpack::IO::StopTimer((kfold_path+"total").c_str());


  mlpack::IO::GetParam<int>((kfold_path+"n_points").c_str()) = 
    data_->n_points();
  mlpack::IO::GetParam<int>((kfold_path+"n_correct").c_str()) = 
    n_correct();
  mlpack::IO::GetParam<int>((kfold_path+"n_incorrect").c_str()) = 
    n_incorrect();
  mlpack::IO::GetParam<double>((kfold_path+"p_correct").c_str()) = 
    1.0 * portion_correct();
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
  std::string root_path;
  /** The fastexec module for cross validation and result storage */
  std::string kfold_path;
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
	    const char* path_to_root,
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
  void StratifiedSplitCVSet_(int i_fold, index_t num_classes, std::vector<index_t>& cv_labels_ct, 
			     std::vector<index_t>& cv_labels_startpos, const std::vector<index_t>& permutation, Dataset *train, Dataset *validation){
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
  std::string train_name;
  std::string validation_name;
  std::ostringstream o;
  
  // save training and validation sets for this fold
  
  o << i_fold;
  train_name = "cv_train_" + o.str() + ".csv";
  validation_name = "cv_validation_" + o.str() + ".csv";

  train.WriteCsv(train_name.c_str());
  validation.WriteCsv(validation_name.c_str());
}

template<class TLearner>
void GeneralCrossValidator<TLearner>::Init(
    int learner_typeid,
    int default_k,
    const Dataset *data_input,
    const char* path_to_root,
    const char *learner_fx_name,
    const char *kfold_fx_name) {
  /** initialization for general parameters */
  learner_typeid_ = learner_typeid;
  data_ = data_input;

  root_path = path_to_root;
  kfold_path = kfold_fx_name;

  if(!mlpack::IO::HasParam((kfold_path+"k").c_str()))
    mlpack::IO::GetParam<int>((kfold_path+"k").c_str()) = default_k;
  n_folds_ = mlpack::IO::GetParam<int>((kfold_path+"k").c_str());

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
  mlpack::IO::StartTimer((kfold_path+"total").c_str());
  num_data_points_ = data_->n_points();

  /** for type 0: Classification ONLY */
  if (learner_typeid_ == 0) {
    // get label information
    /* list of labels, need to be integers. e.g. [0,1,2] for a 3-class dataset */
    std::vector<double> cv_labels_list;
    /* array of label indices, after grouping. e.g. [c1[0,5,6,7,10,13,17],c2[1,2,4,8,9],c3[...]]*/
    std::vector<index_t> cv_labels_index;
    /* counted number of label for each class. e.g. [7,5,8]*/
    std::vector<index_t> cv_labels_ct;
    /* start positions of each classes in the cv label list. e.g. [0,7,12] */
    std::vector<index_t> cv_labels_startpos;
    // Get label list and label indices from the cross validation data set
    index_t num_classes = data_->n_labels();

    data_->GetLabels(cv_labels_list, cv_labels_index, cv_labels_ct, cv_labels_startpos);

    // randomize the original data set within each class if necessary
    std::vector<index_t> permutation;

    if (randomized) {
      permutation.reserve(num_data_points_);
      for (index_t i_classes=0; i_classes<num_classes; i_classes++) {
	std::vector<index_t> sub_permutation; // within class permut indices
	math::MakeRandomPermutation(cv_labels_ct[i_classes], &sub_permutation.front());
	// use sub-permutation indicies to form the whole permutation
	if (i_classes==0){
	  for (index_t j=0; j<cv_labels_ct[i_classes]; j++)
	    permutation[cv_labels_startpos[i_classes]+j] = cv_labels_index[ sub_permutation[j] ];
	}
	else {
	  for (index_t j=0; j<cv_labels_ct[i_classes]; j++)
	    permutation[cv_labels_startpos[i_classes]+j] = cv_labels_index[ cv_labels_ct[i_classes-1]+sub_permutation[j] ];
	}
      }
    } // e.g. [10,13,5,17,0,6,7,,4,9,8,1,2,,...]
    else {
      permutation.assign(cv_labels_index.begin(), cv_labels_index.end()); // e.g. [0,5,6,7,10,13,17,,1,2,4,8,9,,...]
    }
    // begin CV
    for (int i_fold = 0; i_fold < n_folds_; i_fold++) {
      Learner classifier;
      Dataset train;
      Dataset validation;
      
      index_t local_n_correct = 0;

      /** COPY
      datanode *learner_module = fx_copy_module(root_module_,
          learner_fx_name_, "%s/%d/%s",
          kfold_module_->key, i_fold, learner_fx_name_);
      datanode *foldmodule = fx_submodule(learner_module, "..");
      */
      std::string fold_path;

      // Split labeled data sets according to i_fold. Use Stratified Cross-Validation to ensure 
      // that approximately the same portion of data (training/validation) are used for each class.
      StratifiedSplitCVSet_(i_fold, num_classes, cv_labels_ct, cv_labels_startpos, permutation, &train, &validation);
      if (mlpack::IO::HasParam((kfold_path+"save").c_str())) {
	SaveTrainValidationSet_(i_fold, train, validation);
      }
      
      mlpack::IO::StartTimer((fold_path+"train").c_str());
      // training
      classifier.InitTrain(learner_typeid_, train);
      mlpack::IO::StopTimer((fold_path+"train").c_str());

      // validation; measure method: percent of correctly classified validation samples
      mlpack::IO::StartTimer((fold_path+"validation").c_str());

      for (index_t i = 0; i < validation.n_points(); i++) {
	arma::vec validation_vector(validation.n_features() - 1);

        memcpy(validation_vector.memptr(), validation.matrix().colptr(i),
          sizeof(double) * (validation_vector.n_elem));
	
	// testing (classification)
	int label_predict = int(classifier.Predict(learner_typeid_, validation_vector));
	double label_expect_dbl = validation.matrix()(i, validation_vector.n_elem);
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
      mlpack::IO::StopTimer((fold_path+"validation").c_str());
      mlpack::IO::GetParam<int>((fold_path+"local_n_correct").c_str()) = 
        local_n_correct;
      mlpack::IO::GetParam<int>((fold_path+"local_n_incorrect").c_str()) = 
        validation.n_points() - local_n_correct;
      mlpack::IO::GetParam<double>((fold_path+"local_p_correct").c_str()) = 
        local_n_correct * 1.0 / validation.n_points();

      clsf_n_correct_ += local_n_correct;
    }
    mlpack::IO::StopTimer((kfold_path+"total").c_str());

    mlpack::IO::GetParam<int>((kfold_path+"n_points").c_str()) = 
      num_data_points_;
    mlpack::IO::GetParam<int>((kfold_path+"n_correct").c_str()) = 
      clsf_n_correct();
    mlpack::IO::GetParam<int>((kfold_path+"n_incorrect").c_str()) = 
      clsf_n_incorrect();
    mlpack::IO::GetParam<double>((kfold_path+"p_correct").c_str()) = 
      1.0 * clsf_portion_correct();
  }
  /** For type 1:regression & 2:density estimation */
  else if (learner_typeid_ == 1 || learner_typeid_ == 2) {
    double accu_msq_err_all_folds = 0.0;

    // randomize the original data set if necessary
    std::vector<index_t> permutation;  
    if (randomized) {
      math::MakeRandomPermutation(num_data_points_, &permutation.front());
    } else {
      math::MakeIdentityPermutation(num_data_points_, &permutation.front());
    }
    // begin CV
    for (int i_fold = 0; i_fold < n_folds_; i_fold++) {
      Learner learner;
      Dataset train;
      Dataset validation;
      
      double msq_err_local = 0.0;
      double accu_sq_err_local = 0.0;
      /* COPY
      datanode *learner_module = fx_copy_module(root_module_,
          learner_fx_name_, "%s/%d/%s",
          kfold_module_->key, i_fold, learner_fx_name_);
      datanode *foldmodule = fx_submodule(learner_module, "..");
      */
      std::string fold_path;

      // Split general data sets according to i_fold
      data_->SplitTrainTest(n_folds_, i_fold, permutation, train, validation);
      
      if (mlpack::IO::HasParam((fold_path+"save").c_str())) {
	SaveTrainValidationSet_(i_fold, train, validation);
      }
      
      mlpack::IO::StartTimer((fold_path+"train").c_str());
      // training
      learner.InitTrain(learner_typeid_, train); // 0: dummy number of classes
      mlpack::IO::StopTimer((fold_path+"train").c_str());
      
      // validation
      mlpack::IO::StartTimer((fold_path+"validation").c_str());
      for (index_t i = 0; i < validation.n_points(); i++) {
	arma::vec validation_vector(validation.n_features() - 1);

        memcpy(validation_vector.memptr(), validation.matrix().colptr(i),
          sizeof(double) * (validation_vector.n_elem));
	
        // testing
	double value_predict = learner.Predict(learner_typeid_, validation_vector);
	double value_true = validation.matrix()(i, validation_vector.n_elem);
	double value_err = value_predict - value_true;
	
	// Calculate squared error: sublevel
	accu_sq_err_local  += pow(value_err, 2);
      }
      mlpack::IO::StopTimer((fold_path+"validation").c_str());
      
      msq_err_local = accu_sq_err_local / validation.n_points();
//      fx_format_result(foldmodule, "local_msq_err", "%f", msq_err_local);

      accu_msq_err_all_folds += msq_err_local;
    }
    mlpack::IO::StopTimer((kfold_path+"total").c_str());
    
    // Calculate mean squared error: over all folds
    msq_err_all_folds_ = accu_msq_err_all_folds / n_folds_;
  //  fx_format_result(kfold_module_, "msq_err_all_folds", "%f", msq_err_all_folds_);
  }
  else {
    fprintf(stderr, "Other learner types or Unknown learner type id! Cross validation stops!\n");
    return;
  }
}

#endif
