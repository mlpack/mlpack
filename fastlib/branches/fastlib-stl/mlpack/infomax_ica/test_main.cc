/**
 * @file main.cc
 *
 * Test driver for our infomax ICA method.
 */

#include <fastlib/fx/io.h>
#include "infomax_ica.h"
#include "fastlib/fastlib.h"


using namespace mlpack;


#define BOOST_TEST_MODULE TestInfomaxICA 
#include <boost/test/unit_test.hpp> 


BOOST_AUTO_TEST_CASE(SqrtM) { 

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

    Matrix intermediateb = icab_.sampleCovariance(testdatab_);
    icab_.sqrtm(intermediateb);
}


BOOST_AUTO_TEST_CASE(TestCov) {

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

    InfomaxICA ica_(lambda_,b_,epsilon_);
    ica_.sampleCovariance(testdata_);

}

BOOST_AUTO_TEST_CASE(TestICA) {

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


    InfomaxICA ica_(lambda_,b_,epsilon_);
    Matrix unmixing;
    ica_.applyICA(testdata_);
    ica_.getUnmixing(unmixing);
    ica_.displayMatrix(unmixing);
 }

