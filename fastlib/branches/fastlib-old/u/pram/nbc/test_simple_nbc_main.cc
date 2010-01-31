#include "simple_nbc.h"
#include "base/test.h"

class TestClassSimpleNBC{
 private:
  SimpleNaiveBayesClassifier *nbc_test_;
  const char *filename_train_, *filename_test_;
  const char *train_result_, *test_result_;
  index_t number_of_classes_;

 public:

  void Init(const char *filename_train, const char *filename_test, const char *train_result, const char *test_result, const int number_of_classes) {
    nbc_test_ = new SimpleNaiveBayesClassifier();
    filename_train_ = filename_train;
    filename_test_ = filename_test;
    train_result_ = train_result;
    test_result_ = test_result;
    number_of_classes_ = number_of_classes;
  }

  void Destruct() {
    delete nbc_test_;
    delete filename_train_;
    delete filename_test_;
    delete train_result_;
    delete test_result_;
  }

  void TestInitTrain() {
    Matrix train_data, train_res, calc_mat;
    data::Load(filename_train_, &train_data);
    data::Load(train_result_, &train_res); 
    struct datanode* nbc_module = fx_submodule(NULL,"nbc","nbc");
    fx_format_param(nbc_module, "classes", "%d", 2);
    nbc_test_->InitTrain(train_data, nbc_module);
    index_t number_of_features = nbc_test_->means_.n_rows();
    calc_mat.Init(2*number_of_features + 1, number_of_classes_);
    for(index_t i = 0; i < number_of_features; i++) {
      for(index_t j = 0; j < number_of_classes_; j++) {
	calc_mat.set(i, j, nbc_test_->means_.get(i, j));
	calc_mat.set(i + number_of_features, j, nbc_test_->variances_.get(i, j));	
      }
    }
    for(index_t i = 0; i < number_of_classes_; i++) {
      calc_mat.set(2 * number_of_features, i, nbc_test_->class_probabilities_[i]);      
    }
    
    for(index_t i = 0; i < calc_mat.n_rows(); i++) {
      for(index_t j = 0; j < number_of_classes_; j++) {
	TEST_DOUBLE_APPROX(train_res.get(i, j), calc_mat.get(i, j), 0.0001);
      }
    }
    NONFATAL("Test InitTrain passed...\n");
    
  }

  void TestClassify() {
    Matrix test_data, test_res;
    Vector test_res_vec, calc_vec;
    data::Load(filename_test_, &test_data);
    data::Load(test_result_, &test_res); 
    nbc_test_->Classify(test_data, &calc_vec);
    index_t number_of_datum = test_data.n_cols();
    test_res.MakeColumnVector(0, &test_res_vec);
    for(index_t i = 0; i < number_of_datum; i++) {
      TEST_ASSERT(test_res_vec.get(i) == calc_vec.get(i));
    }
    NONFATAL("Test Classify passed...\n");
  }

  void TestAll() {
    TestInitTrain();
    TestClassify();
  }
};

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  TestClassSimpleNBC test;

  const char *train_data = fx_param_str(NULL, "training_set", "trainSet.arff");
  const char *train_res = fx_param_str(NULL, "training_results", "trainRes.arff");
  const char *test_data = fx_param_str(NULL, "testing_set", "testSet.arff");
  const char *test_res = fx_param_str(NULL, "testing_results", "testRes.arff");
  const int num_classes = fx_param_int(NULL, "num_classes", 2);

  test.Init(train_data, test_data, train_res, test_res, num_classes);
  test.TestAll();
  
  fx_done();
}
