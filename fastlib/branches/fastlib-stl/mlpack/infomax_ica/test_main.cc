/**
 * @file main.cc
 *
 * Test driver for our infomax ICA method.
 */

#include <fastlib/fx/io.h>
#include "infomax_ica.h"
#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>


using namespace mlpack;

#define BOOST_TEST_MODULE TestInfomaxICA
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(TestCov) { 

    InfomaxICA *ica_;
    Matrix testdata_;
    double lambda_;
    int b_;
    double epsilon_;

    // load some test data that has been verified using the matlab
    // implementation of infomax
    arma::mat tmpdata;
    data::Load("fake.arff", tmpdata);
    arma_compat::armaToMatrix(tmpdata, testdata_);
    lambda_=0.001;
    b_=5;
    epsilon_=0.001;
    ica_ = new InfomaxICA(lambda_,b_,epsilon_);

    ica_->sampleCovariance(testdata_);
 }


BOOST_AUTO_TEST_CASE(TestSqrtm) { 
    
    InfomaxICA *ica_;
    Matrix testdata_;
    double lambda_;
    int b_;
    double epsilon_;
  
    // load some test data that has been verified using the matlab
    // implementation of infomax
    arma::mat tmpdata;
    data::Load("fake.arff", tmpdata);
    arma_compat::armaToMatrix(tmpdata, testdata_);
    lambda_=0.001;
    b_=5;
    epsilon_=0.001;
    ica_ = new InfomaxICA(lambda_,b_,epsilon_);

    Matrix intermediate = ica_->sampleCovariance(testdata_);
    ica_->sqrtm(intermediate);
 }


BOOST_AUTO_TEST_CASE(TestICA) { 
    
    InfomaxICA *ica_;
    Matrix testdata_;
    double lambda_;
    int b_;
    double epsilon_;
  
    // load some test data that has been verified using the matlab
    // implementation of infomax
    arma::mat tmpdata;
    data::Load("fake.arff", tmpdata);
    arma_compat::armaToMatrix(tmpdata, testdata_);
    lambda_=0.001;
    b_=5;
    epsilon_=0.001;
    ica_ = new InfomaxICA(lambda_,b_,epsilon_);

    Matrix unmixing;
    ica_->applyICA(testdata_);
    ica_->getUnmixing(unmixing);
    ica_->displayMatrix(unmixing);
  }

