#include "infomax_ica.h"
#include "base/test.h"

class TestInfomaxICA {

 public:
  void Init(){
    // load some test data that has been verified using the matlab
    // implementation of infomax
    testdata_.InitFromFile("../../example/fake.arff");
    lambda_=0.001;
    b_=5;
    ica_ = new InfomaxICA(lambda_,b_);
  }

  void Destruct(){
    delete ica_;
  }    

  void TestCov(){
    ica_->sampleCovariance(testdata_.matrix());
  }
  
  void TestSqrtm(){
    Matrix intermediate = ica_->sampleCovariance(testdata_.matrix());
    ica_->sqrtm(intermediate);
  }

  void TestICA(){
    ica_->applyICA(testdata_);
    ica_->displayMatrix(ica_->getUnmixing());
  }

  void TestAll() {
    TestSqrtm();
    TestCov();
    TestICA();
  }

 private:
  InfomaxICA *ica_;
  Dataset testdata_;
  double lambda_;
  int b_;
  
};
