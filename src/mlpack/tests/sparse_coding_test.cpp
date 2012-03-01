/**
 * @file sparse_coding_test.cpp
 *
 * Test for Sparse Coding
 */

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.


#include <armadillo>
#include <mlpack/methods/sparse_coding/sparse_coding.hpp>

#include <boost/test/unit_test.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::sparse_coding;

BOOST_AUTO_TEST_SUITE(SparseCodingTest);


void VerifyCorrectness(vec beta, vec errCorr, double lambda)
{
  const double tol = 1e-12;
  size_t nDims = beta.n_elem;
  for(size_t j = 0; j < nDims; j++)
  {
    if (beta(j) == 0)
    {
      // make sure that errCorr(j) <= lambda
      BOOST_REQUIRE_SMALL(std::max(fabs(errCorr(j)) - lambda, 0.0), tol);
    }
    else if (beta(j) < 0)
    {
      // make sure that errCorr(j) == lambda
      BOOST_REQUIRE_SMALL(errCorr(j) - lambda, tol);
    }
    else
    { // beta(j) > 0
      // make sure that errCorr(j) == -lambda
      BOOST_REQUIRE_SMALL(errCorr(j) + lambda, tol);
    }
  }
}


BOOST_AUTO_TEST_CASE(SparseCodingTestCodingStepLasso)
{
  double lambda1 = 0.1;
  u32 nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  u32 nPoints = X.n_cols;
  
  // normalize each point since these are images
  for(u32 i = 0; i < nPoints; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }  

  SparseCoding sc(X, nAtoms, lambda1);
  sc.DataDependentRandomInitDictionary();
  sc.OptimizeCode();  

  mat D = sc.MatD();
  mat Z = sc.MatZ();
  
  for(u32 i = 0; i < nPoints; i++) {
    vec errCorr = trans(D) * (D * Z.unsafe_col(i) - X.unsafe_col(i));
    VerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
  
}

BOOST_AUTO_TEST_CASE(SparseCodingTestCodingStepElasticNet)
{
  double lambda1 = 0.1;
  double lambda2 = 0.2;
  u32 nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  u32 nPoints = X.n_cols;
  
  // normalize each point since these are images
  for(u32 i = 0; i < nPoints; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }  

  SparseCoding sc(X, nAtoms, lambda1, lambda2);
  sc.DataDependentRandomInitDictionary();
  sc.OptimizeCode();  

  mat D = sc.MatD();
  mat Z = sc.MatZ();
  
  for(u32 i = 0; i < nPoints; i++) {
    vec errCorr = 
      (trans(D) * D + lambda2 *
       eye(nAtoms, nAtoms)) * Z.unsafe_col(i) - trans(D) * X.unsafe_col(i);
    
    VerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

BOOST_AUTO_TEST_CASE(SparseCodingTestDictionaryStep)
{
  const double tol = 1e-12;

  double lambda1 = 0.1;
  u32 nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  u32 nPoints = X.n_cols;
  
  // normalize each point since these are images
  for(u32 i = 0; i < nPoints; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }  

  SparseCoding sc(X, nAtoms, lambda1);
  sc.DataDependentRandomInitDictionary();
  sc.OptimizeCode();  
  
  mat D = sc.MatD();
  mat Z = sc.MatZ();
  
  X = D * Z;
  
  sc.SetData(X);
  sc.DataDependentRandomInitDictionary();

  uvec adjacencies = find(Z);
  sc.OptimizeDictionary(adjacencies);
  
  mat D_hat = sc.MatD();

  BOOST_REQUIRE_SMALL(norm(D - D_hat, "fro"), tol);
  
}

/*
BOOST_AUTO_TEST_CASE(SparseCodingTestWhole)
{

}
*/


BOOST_AUTO_TEST_SUITE_END();
