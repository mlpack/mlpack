/**
 * @file svm_test.cpp
 *
 * Test for SVM.
 */
#include <mlpack/core.h>
#include <mlpack/methods/svm/svm.h>

#include <boost/test/unit_test.hpp>

using std::string;
using namespace mlpack;
using namespace mlpack::svm;

BOOST_AUTO_TEST_SUITE(SVMTest);

// Create test data
arma::mat matrix(20, 3);
bool first = true;

/**
 * Creates the data to train and test with and prints it to stdout.  Should only
 * have any effect once.
 */
void setup()
{
  if(!first)
    return;
  first = false;

  CLI::GetParam<bool>("svm/shrink") = true;
  CLI::GetParam<double>("svm/epsilon") = .1;
  CLI::GetParam<double>("svm/sigma") = 1;
  // Protect the test from taking forever
  CLI::GetParam<size_t>("svm/n_iter") = 10000;

  matrix <<
    7.19906628001437787e-01 << 1.83250823399634477e+00 << 0 << arma::endr <<
    1.37899419263889733e+01 << 1.78198235122579263e+00 << 1 << arma::endr <<
    6.68859485848275703e-01 << 2.14083320956715983e+00 << 0 << arma::endr <<
    1.84729928795588165e+01 << 2.25024702760868101e+00 << 1 << arma::endr <<
    9.22802773268335819e-01 << 1.61469358350834513e+00 << 0 << arma::endr <<
    2.06209849662245204e-01 << 6.34699695340683490e-01 << 1 << arma::endr <<
    4.01062068250524817e-01 << 1.65802752932441777e+00 << 0 << arma::endr <<
    5.02985607135568635e+00 << 1.39976642741810831e+00 << 1 << arma::endr <<
    3.66471199955079319e-01 << 1.62780588172739638e+00 << 0 << arma::endr <<
    1.56912570240400999e+01 << 2.16941541650770953e+00 << 1 << arma::endr <<
    9.98909584711729304e-01 << 2.00337906391517206e+00 << 0 << arma::endr <<
    1.31430438780891912e+01 << 1.34410346059319719e+00 << 1 << arma::endr <<
    3.41572957272442523e-01 << 1.16758463655951639e+00 << 0 << arma::endr <<
    9.53941410851637528e-01 << 6.30271704462483373e-01 << 1 << arma::endr <<
    7.07135529120981432e-01 << 2.17763537339756041e+00 << 0 << arma::endr <<
    9.68899714280338742e+00 << 1.26922579378319256e+00 << 1 << arma::endr <<
    9.82393905512240706e-01 << 2.36790583090293483e+00 << 0 << arma::endr <<
    1.31583349281727973e+01 << 1.45115094722767868e+00 << 1 << arma::endr <<
    3.80991188521027202e-01 << 9.05379134419085019e-01 << 0 << arma::endr <<
    1.86057436180327755e+01 << 2.26941891469499968e+00 << 1 << arma::endr;
  matrix = trans(matrix);

  std::cout << matrix << std::endl;
}

/**
 * Compares predicted values with known values to see if the prediction/training
 * works.
 *
 * @param learner_typeid Magic number for selecting between classification and
 *     regression.
 * @param data The dataset with the data to predict with.
 * @param svm The SVM class instance that has been trained for this data, et al.
 */
template<typename T>
void verify(size_t learner_typeid, arma::mat& data, SVM<T>& svm)
{
  for(size_t i = 0; i < data.n_cols; i++)
  {
    arma::vec testvec = data.col(i);

    double predictedvalue = svm.Predict(learner_typeid, testvec);
    BOOST_REQUIRE_CLOSE(predictedvalue, data(data.n_rows - 1, i), 1e-6);
  }
}

/**
 * Trains a classifier with a linear kernel and checks predictions against
 * classes.
 */
//BOOST_AUTO_TEST_CASE(svm_classification_linear_kernel_test) {
//  setup();
//
//  arma::mat trainingdata;
//  trainingdata = matrix;
//  SVM<SVMLinearKernel> svm;
//  svm.InitTrain(0,trainingdata); // 0 for classification
//  verify(0,trainingdata,svm);
//}

/**
 * Trains a classifier with a gaussian kernel and checks predictions against
 * classes.
 */
BOOST_AUTO_TEST_CASE(svm_classification_gaussian_kernel_test) {
  setup();

  arma::mat trainingdata;
  trainingdata = matrix;
  SVM<SVMRBFKernel> svm;
  svm.InitTrain(0,trainingdata); // 0 for classification
  verify(0,trainingdata,svm);
}

/**
 * Trains a classifier with a linear kernel and checks predictions against
 * classes, using regression. TODO: BROKEN
 */
//BOOST_AUTO_TEST_CASE(svm_regression_linear_kernel_test) {
//  setup();
//
//  arma::mat trainingdata;
//  trainingdata = matrix;
//  SVM<SVMLinearKernel> svm;
//  svm.InitTrain(1,trainingdata); // 0 for classification
//  //verify(1,trainingdata,svm);
//}

/**
 * Trains a classifier with a gaussian kernel and checks predictions against
 * classes, using regression. TODO: BROKEN
 */
//BOOST_AUTO_TEST_CASE(svm_regression_gaussian_kernel_test) {
//  setup();
//
//  arma::mat trainingdata;
//  trainingdata = matrix;
//  SVM<SVMRBFKernel> svm;
//  svm.InitTrain(1,trainingdata); // 0 for classification
//  //verify(1,trainingdata,svm);
//}

BOOST_AUTO_TEST_SUITE_END();
