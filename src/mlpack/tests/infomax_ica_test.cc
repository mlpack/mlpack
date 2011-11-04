/**
 * @file main.cc
 *
 * Test driver for our infomax ICA method.
 */
#include <mlpack/core.h>

#include <mlpack/methods/infomax_ica/infomax_ica.h>

using namespace mlpack;
using namespace infomax_ica;

#include <boost/test/unit_test.hpp>

void testSQRTM(InfomaxICA& icab_, arma::mat& m) {
  icab_.sqrtm(m);
}

BOOST_AUTO_TEST_SUITE(InfomaxIcaTest);

  BOOST_AUTO_TEST_CASE(SqrtM) {
    arma::mat testdatab_;
    double lambdab_ = 0.001;
    int bb_ = 5;
    double epsilonb_ = 0.001;

    data::Load("fake.csv", testdatab_);

    InfomaxICA icab_(lambdab_, bb_, epsilonb_);

    arma::mat intermediateb = icab_.sampleCovariance(testdatab_);
    testSQRTM(icab_, intermediateb);
  }

  BOOST_AUTO_TEST_CASE(TestCov) {
    arma::mat testdata_;

    double lambda_ = 0.001;
    int b_ = 5;
    double epsilon_ = 0.001;

    // load some test data that has been verified using the matlab
    // implementation of infomax
    data::Load("fake.csv", testdata_);

    InfomaxICA ica_(lambda_, b_, epsilon_);
    ica_.sampleCovariance(testdata_);
  }

  BOOST_AUTO_TEST_CASE(TestICA) {
    arma::mat testdata_;
    double lambda_ = 0.001;
    int b_ = 5;
    double epsilon_ = 0.001;

    // load some test data that has been verified using the matlab
    // implementation of infomax
    data::Load("fake.csv", testdata_);

    InfomaxICA ica_(lambda_, b_, epsilon_);
    arma::mat unmixing;
    ica_.applyICA(testdata_);
    ica_.getUnmixing(unmixing);
  }

BOOST_AUTO_TEST_SUITE_END();
