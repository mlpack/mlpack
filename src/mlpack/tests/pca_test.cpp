/**
 * @file pca_test.cpp
 *
 * Test file for PCA class
 *
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(PcaTest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::pca;

BOOST_AUTO_TEST_CASE(LinearRegressionTest)
{
  int n_rows;
  int n_cols;

  mat coeff, coeff1;
  vec eigVal, eigVal1;
  mat score, score1;

  mat data = randu<mat>(100,100);

  mlpack::pca::PCA p;

  p.Apply(data, score1, eigVal1, coeff1);
  princomp(coeff, score, eigVal, trans(data)); score = trans(score); coeff = trans(coeff);

  n_rows = eigVal.n_rows;
  n_cols = eigVal.n_cols;

  //verify the PCA results based on the eigen Values
  for(int i = 0; i < n_rows; i++)
  {
    for(int j = 0; j < n_cols; j++)
    {
      assert(fabs(eigVal(i, j) - eigVal1(i, j)) < 0.0001);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
