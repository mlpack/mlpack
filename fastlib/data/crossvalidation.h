/**
 * @file crossvalidation.h
 *
 * Cross validation support.
 */

#ifndef DATA_CROSSVALIDATION
#define DATA_CROSSVALIDATION

#include "dataset.h"

#include "la/matrix.h"
#include "fx/fx.h"

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
  FORBID_COPY(SimpleCrossValidator);
  
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
  Matrix confusion_matrix_;
  
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
   */
  void Init(const Dataset *data_with_labels,
      int n_labels,
      int default_k,
      struct datanode *module_root,
      const char *classifier_fx_name,
      const char *kfold_fx_name = "kfold") {
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
    kfold_module_ = fx_submodule(module_root, kfold_fx_name, kfold_fx_name);
    classifier_fx_name_ = classifier_fx_name;
    
    n_folds_ = fx_param_int(kfold_module_, "k", default_k);
    
    DEBUG_ONLY(n_correct_ = BIG_BAD_NUMBER);
    
    confusion_matrix_.Init(n_classes_, n_classes_);
    confusion_matrix_.SetZero();
  }
  
  /**
   * Runs cross-validation.
   *
   * @param randomized whether to use a random permutation of the data,
   *        or just to stride it
   */
  void Run(bool randomized = false) {
    ArrayList<index_t> permutation;
    
    if (randomized) {
      math::MakeRandomPermutation(data_->n_points(), &permutation);
    } else {
      math::MakeIdentityPermutation(data_->n_points(), &permutation);
    }
    
    n_correct_ = 0;
    
    fx_timer_start(kfold_module_, "total");
    
    for (int i_folds = 0; i_folds < n_folds_; i_folds++) {
      Classifier classifier;
      Dataset test;
      Dataset train;
      index_t local_n_correct = 0;
      datanode *foldmodule = fx_submodule(kfold_module_, NULL,
          String().InitSprintf("%d", i_folds).c_str());
      datanode *classifier_module = fx_submodule(foldmodule, NULL,
          classifier_fx_name_);
      
      fx_default_param_node(classifier_module, "", root_module_,
          classifier_fx_name_);

      data_->SplitTrainTest(n_folds_, i_folds, permutation, &train, &test);
    
      DEBUG_MSG(1, "cross: Training fold %d", i_folds);
      fx_timer_start(foldmodule, "train");
      classifier.InitTrain(train, n_classes_, classifier_module);
      fx_timer_stop(foldmodule, "train");
      
      fx_timer_start(foldmodule, "test");
      DEBUG_MSG(1, "cross: Testing fold %d", i_folds);
      for (index_t i = 0; i < test.n_points(); i++) {
        Vector test_vector_with_label;
        Vector test_vector;
        
        test.matrix().MakeColumnVector(i, &test_vector_with_label);
        test_vector_with_label.MakeSubvector(
            0, test.n_features()-1, &test_vector);
        
        int label_predict = classifier.Classify(test_vector);
        double label_expect_dbl = test_vector_with_label[test.n_features()-1];
        int label_expect = int(label_expect_dbl);
        
        DEBUG_ASSERT(double(label_expect) == label_expect_dbl);
        DEBUG_ASSERT(label_expect < n_classes_);
        DEBUG_ASSERT(label_expect >= 0);
        DEBUG_ASSERT(label_predict < n_classes_);
        DEBUG_ASSERT(label_predict >= 0);
        
        if (label_expect == label_predict) {
          local_n_correct++;
        }
        
        confusion_matrix_.ref(label_expect, label_predict) += 1;
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

    fx_format_result(kfold_module_, "n_correct", "%"LI"d",
        n_correct());
    fx_format_result(kfold_module_, "n_incorrect", "%"LI"d",
        n_incorrect());
    fx_format_result(kfold_module_, "p_correct", "%.03f",
        1.0 * portion_correct());
  }
  
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
  const Matrix& confusion_matrix() const {
    return confusion_matrix_;
  }
  
  /** Gets the dataset. */
  const Dataset& data() const {
    return *data_;
  }
};

#endif
