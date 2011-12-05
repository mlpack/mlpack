/**
 * @file lars_test.cpp
 *
 * Test for LARS
 */

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.


#include <armadillo>
#include "lars.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE HELLO
#include <boost/test/unit_test.hpp>

//BOOST_AUTO_TEST_SUITE(LARS_Test);

void GenerateProblem(mat& X, vec& y, u32 nPoints, u32 nDims) {
  X = randn(nPoints, nDims);
  vec beta = randn(nDims, 1);
  y = X * beta;
}


void VerifyCorrectness(vec beta, vec errCorr, double lambda) {
  u32 nDims = beta.n_elem;
  const double tol = 1e-12;
  for(u32 j = 0; j < nDims; j++) {
    if(beta(j) == 0) {
      // make sure that errCorr(j) <= lambda
      BOOST_REQUIRE_SMALL(max(errCorr(j) - lambda, 0.0), tol);
    }
    else if(beta(j) < 0) {
      // make sure that errCorr(j) == lambda
      BOOST_REQUIRE_SMALL(errCorr(j) - lambda, tol);
    }
    else { // beta(j) > 0
      // make sure that errCorr(j) == -lambda
      BOOST_REQUIRE_SMALL(errCorr(j) + lambda, tol);
    }
  }
}


void LassoTest(u32 nPoints, u32 nDims, bool elasticNet, bool useCholesky) {
  mat X;
  vec y;
  
  for(u32 i = 0; i < 100; i++) {
    GenerateProblem(X, y, nPoints, nDims);
    
    // Armadillo's median is broken, so...
    vec sortedAbsCorr = sort(abs(trans(X) * y));
    double lambda_1 = sortedAbsCorr(nDims/2);
    double lambda_2;
    if(elasticNet) {
      lambda_2 = lambda_1 / 2;
    }
    else {
      lambda_2 = 0;
    }
    
    Lars lars;
    lars.Init(X, y, useCholesky, lambda_1, lambda_2);
    lars.DoLARS();
    
    vec betaOpt;
    lars.Solution(betaOpt);
    vec errCorr = (trans(X) * X + lambda_2 * eye(nDims, nDims)) * betaOpt - trans(X) * y;
    
    VerifyCorrectness(betaOpt, errCorr, lambda_1);
  }
}


BOOST_AUTO_TEST_CASE(LARS_Test_Lasso_Cholesky) {
  LassoTest(100, 10, true, false);
}


BOOST_AUTO_TEST_CASE(LARS_Test_Lasso_Gram) {
  LassoTest(100, 10, false, false);
}


BOOST_AUTO_TEST_CASE(LARS_Test_ElasticNet_Cholesky) {
  LassoTest(100, 10, true, true);
}


BOOST_AUTO_TEST_CASE(LARS_Test_ElasticNet_Gram) {
  LassoTest(100, 10, true, false);
}



//BOOST_AUTO_TEST_SUITE_END();
