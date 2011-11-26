/**
 * @file pca_test.cpp
 * @author Ajinkya Kale
 *
 * Test file for PCA class.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(PCATest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::pca;

BOOST_AUTO_TEST_CASE(ArmaComparisonPCATest)
{
  int n_rows;
  int n_cols;

  mat coeff, coeff1;
  vec eigVal, eigVal1;
  mat score, score1;

  mat data = randu<mat>(100,100);

  PCA p;

  p.Apply(data, score1, eigVal1, coeff1);
  princomp(coeff, score, eigVal, trans(data));
  score = trans(score);
  coeff = trans(coeff);

  n_rows = eigVal.n_rows;
  n_cols = eigVal.n_cols;

  // Verify the PCA results based on the eigenvalues.
  for(int i = 0; i < n_rows; i++)
  {
    for(int j = 0; j < n_cols; j++)
    {
      BOOST_REQUIRE_CLOSE(eigVal(i, j), eigVal1(i, j), 1e-5);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
