/**
 * @file test_infomax_ica.h
 * @author Chip Mappus
 *
 * Unit tests for infomax ica method.
 */
/*
#include "infomax_ica.h"
#include "fastlib/base/test.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

class TestInfomaxICA {

 public:
  void Init(){
    // load some test data that has been verified using the matlab
    // implementation of infomax
    arma::mat tmpdata;
    data::Load("fake.arff", tmpdata);
    arma_compat::armaToMatrix(tmpdata, testdata_);
    lambda_=0.001;
    b_=5;
    epsilon_=0.001;
    ica_ = new InfomaxICA(lambda_,b_,epsilon_);
  }


  void NoahAll() { 


  //InfomaxICA *icab_;
  Matrix testdatab_;
  double lambdab_;
  int bb_;
  double epsilonb_;


    arma::mat tmpdatab;
    data::Load("fake.arff", tmpdatab);
    arma_compat::armaToMatrix(tmpdatab, testdatab_);
    lambdab_=0.001;
    bb_=5;
    epsilonb_=0.001;
    
    InfomaxICA icab_(lambdab_, bb_, epsilonb_);

    //icab_ = new InfomaxICA(lambdab_,bb_,epsilonb_);


   
    Matrix intermediateb = icab_.sampleCovariance(testdatab_);
    icab_.sqrtm(intermediateb);



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
    
  
    NoahAll();
  }

 private:
  InfomaxICA *ica_;
  Matrix testdata_;
  double lambda_;
  int b_;
  double epsilon_;
};
*/

