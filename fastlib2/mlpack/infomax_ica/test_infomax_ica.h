/**
 * @file test_infomax_ica.h
 * @author Chip Mappus
 *
 * Unit tests for infomax ica method.
 */

#include "infomax_ica.h"
#include "fastlib/base/test.h"

class TestInfomaxICA {

 public:
  void Init(){
    // load some test data that has been verified using the matlab
    // implementation of infomax
    //testdata_.InitFromFile("../../example/fake.arff");
    data::Load("../../fastlib/fake.arff",&testdata_);
    lambda_=0.001;
    b_=5;
    epsilon_=0.001;
    ica_ = new InfomaxICA(lambda_,b_,epsilon_);
  }

  void Destruct(){
    delete ica_;
  }    

  void TestCov(){
    ica_->sampleCovariance(testdata_);
  }
  
  void TestSqrtm(){
    Matrix intermediate = ica_->sampleCovariance(testdata_);
    ica_->sqrtm(intermediate);
  }

  void TestICA(){
    Matrix unmixing;
    ica_->applyICA(testdata_);
    ica_->getUnmixing(unmixing);
    ica_->displayMatrix(unmixing);
  }

  void TestAll() {
    TestSqrtm();
    TestCov();
    TestICA();
  }

 private:
  InfomaxICA *ica_;
  Matrix testdata_;
  double lambda_;
  int b_;
  double epsilon_;
};
