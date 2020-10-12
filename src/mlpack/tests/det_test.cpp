/**
 * @file tests/det_test.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Unit tests for the functions of the class DTree and the utility functions
 * using this class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "catch.hpp"

// This trick does not work on Windows.  We will have to comment out the tests
// that depend on it.
#ifndef _WIN32
  #define protected public
  #define private public
#endif

#include <mlpack/methods/det/dtree.hpp>
#include <mlpack/methods/det/dt_utils.hpp>

#ifndef _WIN32
  #undef protected
  #undef private
#endif

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

// Tests for the private functions.  We cannot perform these if we are on
// Windows because we cannot make private functions accessible using the macro
// trick above.
#ifndef _WIN32
TEST_CASE("TestGetMaxMinVals", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<arma::mat> tree(testData);

  REQUIRE(tree.MaxVals()[0] == 7);
  REQUIRE(tree.MinVals()[0] == 3);
  REQUIRE(tree.MaxVals()[1] == 7);
  REQUIRE(tree.MinVals()[1] == 0);
  REQUIRE(tree.MaxVals()[2] == 8);
  REQUIRE(tree.MinVals()[2] == 1);
}

TEST_CASE("TestComputeNodeError", "[DETTest]")
{
  arma::vec maxVals("7 7 8");
  arma::vec minVals("3 0 1");

  DTree<arma::mat> testDTree(maxVals, minVals, 5);
  double trueNodeError = -log(4.0) - log(7.0) - log(7.0);

  REQUIRE((double) testDTree.logNegError ==
      Approx(trueNodeError).epsilon(1e-12));

  testDTree.start = 3;
  testDTree.end = 5;

  double nodeError = testDTree.LogNegativeError(5);
  trueNodeError = 2 * log(2.0 / 5.0) - log(4.0) - log(7.0) - log(7.0);
  REQUIRE(nodeError == Approx(trueNodeError).epsilon(1e-12));
}

TEST_CASE("TestWithinRange", "[DETTest]")
{
  arma::vec maxVals("7 7 8");
  arma::vec minVals("3 0 1");

  DTree<arma::mat> testDTree(maxVals, minVals, 5);

  arma::vec testQuery(3);
  testQuery << 4.5 << 2.5 << 2;

  REQUIRE(testDTree.WithinRange(testQuery) == true);

  testQuery << 8.5 << 2.5 << 2;

  REQUIRE(testDTree.WithinRange(testQuery) == false);
}

TEST_CASE("TestFindSplit", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<arma::mat> testDTree(testData);

  size_t obDim;
  double obLeftError, obRightError, obSplit;

  size_t trueDim = 2;
  double trueSplit = 5.5;
  double trueLeftError = 2 * log(2.0 / 5.0) - (log(7.0) + log(4.0) + log(4.5));
  double trueRightError = 2 * log(3.0 / 5.0) - (log(7.0) + log(4.0) +
      log(2.5));

  testDTree.logVolume = log(7.0) + log(4.0) + log(7.0);
  REQUIRE(testDTree.FindSplit(
      testData, obDim, obSplit, obLeftError, obRightError, 1));

  REQUIRE(trueDim == obDim);
  REQUIRE(trueSplit == Approx(obSplit).epsilon(1e-12));

  REQUIRE(trueLeftError == Approx(obLeftError).epsilon(1e-12));
  REQUIRE(trueRightError == Approx(obRightError).epsilon(1e-12));
}

TEST_CASE("TestSplitData", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<arma::mat> testDTree(testData);

  arma::Col<size_t> oTest(5);
  oTest << 1 << 2 << 3 << 4 << 5;

  size_t splitDim = 2;
  double trueSplitVal = 5.5;

  size_t splitInd = testDTree.SplitData(
      testData, splitDim, trueSplitVal, oTest);

  REQUIRE(splitInd == 2); // 2 points on left side.

  REQUIRE(oTest[0] == 1);
  REQUIRE(oTest[1] == 4);
  REQUIRE(oTest[2] == 3);
  REQUIRE(oTest[3] == 2);
  REQUIRE(oTest[4] == 5);
}

TEST_CASE("TestSparseFindSplit", "[DETTest]")
{
  arma::mat realData(4, 7);

  realData << .0 << 4 << 5 << 7 << 0 << 5 << 0 << arma::endr
           << .0 << 5 << 0 << 0 << 1 << 7 << 1 << arma::endr
           << .0 << 5 << 6 << 7 << 1 << 0 << 8 << arma::endr
           << -1 << 2 << 5 << 0 << 0 << 0 << 0 << arma::endr;

  arma::sp_mat testData(realData);

  DTree<arma::sp_mat> testDTree(testData);

  size_t obDim;
  double obLeftError, obRightError, obSplit;

  size_t trueDim = 1;
  double trueSplit = .5;
  double trueLeftError = 2 * log(3.0 / 7.0) -
      (log(7.0) + log(0.5) + log(8.0) + log(6.0));
  double trueRightError = 2 * log(4.0 / 7.0) -
      (log(7.0) + log(6.5) + log(8.0) + log(6.0));

  testDTree.logVolume = log(7.0) + log(7.0) + log(8.0) + log(6.0);
  REQUIRE(testDTree.FindSplit(
      testData, obDim, obSplit, obLeftError, obRightError, 1));

  REQUIRE(trueDim == obDim);
  REQUIRE(trueSplit == Approx(obSplit).epsilon(1e-12));

  REQUIRE(trueLeftError == Approx(obLeftError).epsilon(1e-12));
  REQUIRE(trueRightError == Approx(obRightError).epsilon(1e-12));
}

TEST_CASE("TestSparseSplitData", "[DETTest]")
{
  arma::mat realData(4, 7);

  realData << .0 << 4 << 5 << 7 << 0 << 5 << 0 << arma::endr
           << .0 << 5 << 0 << 0 << 1 << 7 << 1 << arma::endr
           << .0 << 5 << 6 << 7 << 1 << 0 << 8 << arma::endr
           << -1 << 2 << 5 << 0 << 0 << 0 << 0 << arma::endr;

  arma::sp_mat testData(realData);

  DTree<arma::sp_mat> testDTree(testData);

  arma::Col<size_t> oTest(7);
  oTest << 1 << 2 << 3 << 4 << 5 << 6 << 7;

  size_t splitDim = 1;
  double trueSplitVal = .5;

  size_t splitInd = testDTree.SplitData(
      testData, splitDim, trueSplitVal, oTest);

  REQUIRE(splitInd == 3); // 2 points on left side.

  REQUIRE(oTest[0] == 1);
  REQUIRE(oTest[1] == 4);
  REQUIRE(oTest[2] == 3);
  REQUIRE(oTest[3] == 2);
  REQUIRE(oTest[4] == 5);
  REQUIRE(oTest[5] == 6);
  REQUIRE(oTest[6] == 7);
}

#endif

// Tests for the public functions.

TEST_CASE("TestGrow", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  double rootError, lError, rError, rlError, rrError;

  rootError = -log(4.0) - log(7.0) - log(7.0);

  lError = 2 * log(2.0 / 5.0) - (log(7.0) + log(4.0) + log(4.5));
  rError =  2 * log(3.0 / 5.0) - (log(7.0) + log(4.0) + log(2.5));

  rlError = 2 * log(1.0 / 5.0) - (log(0.5) + log(4.0) + log(2.5));
  rrError = 2 * log(2.0 / 5.0) - (log(6.5) + log(4.0) + log(2.5));

  DTree<arma::mat> testDTree(testData);
  double alpha = testDTree.Grow(testData, oTest, false, 2, 1);

  REQUIRE(oTest[0] == 0);
  REQUIRE(oTest[1] == 3);
  REQUIRE(oTest[2] == 1);
  REQUIRE(oTest[3] == 2);
  REQUIRE(oTest[4] == 4);

  // Test the structure of the tree.
  REQUIRE(testDTree.Left()->Left() == NULL);
  REQUIRE(testDTree.Left()->Right() == NULL);
  REQUIRE(testDTree.Right()->Left()->Left() == NULL);
  REQUIRE(testDTree.Right()->Left()->Right() == NULL);
  REQUIRE(testDTree.Right()->Right()->Left() == NULL);
  REQUIRE(testDTree.Right()->Right()->Right() == NULL);

  REQUIRE(testDTree.SubtreeLeaves() == 3);

  REQUIRE(testDTree.SplitDim() == 2);
  REQUIRE(testDTree.SplitValue() == Approx(5.5).epsilon(1e-7));
  REQUIRE(testDTree.Right()->SplitDim() == 1);
  REQUIRE(testDTree.Right()->SplitValue() == Approx(0.5).epsilon(1e-7));

  // Test node errors for every node (these are private functions).
#ifndef _WIN32
  REQUIRE(testDTree.logNegError == Approx(rootError).epsilon(1e-12));
  REQUIRE(testDTree.Left()->logNegError == Approx(lError).epsilon(1e-12));
  REQUIRE(testDTree.Right()->logNegError == Approx(rError).epsilon(1e-12));
  REQUIRE(testDTree.Right()->Left()->logNegError ==
      Approx(rlError).epsilon(1e-12));
  REQUIRE(testDTree.Right()->Right()->logNegError ==
      Approx(rrError).epsilon(1e-12));
#endif

  // Test alpha.
  double rootAlpha, rAlpha;
  rootAlpha = std::log(-((std::exp(rootError) - (std::exp(lError) +
      std::exp(rlError) + std::exp(rrError))) / 2));
  rAlpha = std::log(-(std::exp(rError) - (std::exp(rlError) +
      std::exp(rrError))));

  REQUIRE(alpha == Approx(min(rootAlpha, rAlpha)).epsilon(1e-12));
}

TEST_CASE("TestPruneAndUpdate", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;
  DTree<arma::mat> testDTree(testData);
  double alpha = testDTree.Grow(testData, oTest, false, 2, 1);
  alpha = testDTree.PruneAndUpdate(alpha, testData.n_cols, false);

  REQUIRE(alpha == Approx(numeric_limits<double>::max()).epsilon(1e-12));
  REQUIRE(testDTree.SubtreeLeaves() == 1);

  double rootError = -log(4.0) - log(7.0) - log(7.0);

  REQUIRE(testDTree.LogNegError() == Approx(rootError).epsilon(1e-12));
  REQUIRE(testDTree.SubtreeLeavesLogNegError() ==
      Approx(rootError).epsilon(1e-12));
  REQUIRE(testDTree.Left() == NULL);
  REQUIRE(testDTree.Right() == NULL);
}

TEST_CASE("TestComputeValue", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::vec q1(3), q2(3), q3(3), q4(3);

  q1 << 4 << 2 << 2;
  q2 << 5 << 0.25 << 6;
  q3 << 5 << 3 << 7;
  q4 << 2 << 3 << 3;

  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::mat> testDTree(testData);
  double alpha = testDTree.Grow(testData, oTest, false, 2, 1);

  double d1 = (2.0 / 5.0) / exp(log(4.0) + log(7.0) + log(4.5));
  double d2 = (1.0 / 5.0) / exp(log(4.0) + log(0.5) + log(2.5));
  double d3 = (2.0 / 5.0) / exp(log(4.0) + log(6.5) + log(2.5));

  REQUIRE(d1 == Approx(testDTree.ComputeValue(q1)).epsilon(1e-12));
  REQUIRE(d2 == Approx(testDTree.ComputeValue(q2)).epsilon(1e-12));
  REQUIRE(d3 == Approx(testDTree.ComputeValue(q3)).epsilon(1e-12));
  REQUIRE(0.0 == Approx(testDTree.ComputeValue(q4)).epsilon(1e-12));

  alpha = testDTree.PruneAndUpdate(alpha, testData.n_cols, false);

  double d = 1.0 / exp(log(4.0) + log(7.0) + log(7.0));

  REQUIRE(d == Approx(testDTree.ComputeValue(q1)).epsilon(1e-12));
  REQUIRE(d == Approx(testDTree.ComputeValue(q2)).epsilon(1e-12));
  REQUIRE(d == Approx(testDTree.ComputeValue(q3)).epsilon(1e-12));
  REQUIRE(0.0 == Approx(testDTree.ComputeValue(q4)).epsilon(1e-12));
}

TEST_CASE("TestVariableImportance", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  double rootError, lError, rError, rlError, rrError;

  rootError = -1.0 * exp(-log(4.0) - log(7.0) - log(7.0));

  lError = -1.0 * exp(2 * log(2.0 / 5.0) - (log(7.0) + log(4.0) + log(4.5)));
  rError =  -1.0 * exp(2 * log(3.0 / 5.0) - (log(7.0) + log(4.0) + log(2.5)));

  rlError = -1.0 * exp(2 * log(1.0 / 5.0) - (log(0.5) + log(4.0) + log(2.5)));
  rrError = -1.0 * exp(2 * log(2.0 / 5.0) - (log(6.5) + log(4.0) + log(2.5)));

  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::mat> testDTree(testData);
  testDTree.Grow(testData, oTest, false, 2, 1);

  arma::vec imps;

  testDTree.ComputeVariableImportance(imps);

  REQUIRE((double) 0.0 == Approx(imps[0]).epsilon(1e-12));
  REQUIRE((double) (rError - (rlError + rrError)) ==
      Approx(imps[1]).epsilon(1e-12));
  REQUIRE((double) (rootError - (lError + rError)) ==
      Approx(imps[2]).epsilon(1e-12));
}

TEST_CASE("TestSparsePruneAndUpdate", "[DETTest]")
{
  arma::mat realData(3, 5);

  realData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::sp_mat testData(realData);

  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::sp_mat> testDTree(testData);
  double alpha = testDTree.Grow(testData, oTest, false, 2, 1);
  alpha = testDTree.PruneAndUpdate(alpha, testData.n_cols, false);

  REQUIRE(alpha == Approx(numeric_limits<double>::max()).epsilon(1e-12));
  REQUIRE(testDTree.SubtreeLeaves() == 1);

  double rootError = -log(4.0) - log(7.0) - log(7.0);

  REQUIRE(testDTree.LogNegError() == Approx(rootError).epsilon(1e-12));
  REQUIRE(testDTree.SubtreeLeavesLogNegError() ==
      Approx(rootError).epsilon(1e-12));
  REQUIRE(testDTree.Left() == NULL);
  REQUIRE(testDTree.Right() == NULL);
}

TEST_CASE("TestSparseComputeValue", "[DETTest]")
{
  arma::mat realData(3, 5);

  realData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::vec q1d(3), q2d(3), q3d(3), q4d(3);

  q1d << 4 << 2 << 2;
  q2d << 5 << 0.25 << 6;
  q3d << 5 << 3 << 7;
  q4d << 2 << 3 << 3;

  arma::sp_mat testData(realData);
  arma::sp_vec q1(q1d), q2(q2d), q3(q3d), q4(q4d);

  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::sp_mat> testDTree(testData);
  double alpha = testDTree.Grow(testData, oTest, false, 2, 1);

  double d1 = (2.0 / 5.0) / exp(log(4.0) + log(7.0) + log(4.5));
  double d2 = (1.0 / 5.0) / exp(log(4.0) + log(0.5) + log(2.5));
  double d3 = (2.0 / 5.0) / exp(log(4.0) + log(6.5) + log(2.5));

  REQUIRE(d1 == Approx(testDTree.ComputeValue(q1)).epsilon(1e-12));
  REQUIRE(d2 == Approx(testDTree.ComputeValue(q2)).epsilon(1e-12));
  REQUIRE(d3 == Approx(testDTree.ComputeValue(q3)).epsilon(1e-12));
  REQUIRE(0.0 == Approx(testDTree.ComputeValue(q4)).epsilon(1e-12));

  alpha = testDTree.PruneAndUpdate(alpha, testData.n_cols, false);

  double d = 1.0 / exp(log(4.0) + log(7.0) + log(7.0));

  REQUIRE(d == Approx(testDTree.ComputeValue(q1)).epsilon(1e-12));
  REQUIRE(d == Approx(testDTree.ComputeValue(q2)).epsilon(1e-12));
  REQUIRE(d == Approx(testDTree.ComputeValue(q3)).epsilon(1e-12));
  REQUIRE(0.0 == Approx(testDTree.ComputeValue(q4)).epsilon(1e-12));
}

/**
 * These are not yet implemented.
 *
TEST_CASE("TestTagTree", "[DETTest]")
{
  MatType testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
            << 5 << 0 << 1 << 7 << 1 << arma::endr
            << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<>* testDTree = new DTree<>(&testData);

  delete testDTree;
}

TEST_CASE("TestFindBucket", "[DETTest]")
{
  MatType testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
            << 5 << 0 << 1 << 7 << 1 << arma::endr
            << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<>* testDTree = new DTree<>(&testData);

  delete testDTree;
}

// Test functions in dt_utils.hpp

TEST_CASE("TestTrainer", "[DETTest]")
{

}

TEST_CASE("TestPrintVariableImportance", "[DETTest]")
{

}

TEST_CASE("TestPrintLeafMembership", "[DETTest]")
{

}
*/

// Test the copy constructor and the copy operator.
TEST_CASE("CopyConstructorAndOperatorTest", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  // Construct another DTree for testing the children.
  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::mat> *testDTree = new DTree<arma::mat>(testData);
  testDTree->Grow(testData, oTest, false, 2, 1);

  DTree<arma::mat> testDTree2(*testDTree);
  DTree<arma::mat> testDTree3 = *testDTree;

  double maxVals0 = testDTree->MaxVals()[0];
  double maxVals1 = testDTree->MaxVals()[1];
  double maxVals2 = testDTree->MaxVals()[2];
  double minVals0 = testDTree->MinVals()[0];
  double minVals1 = testDTree->MinVals()[1];
  double minVals2 = testDTree->MinVals()[2];

  double maxValsL0 = testDTree->Left()->MaxVals()[0];
  double maxValsL1 = testDTree->Left()->MaxVals()[1];
  double maxValsL2 = testDTree->Left()->MaxVals()[2];
  double minValsL0 = testDTree->Left()->MinVals()[0];
  double minValsL1 = testDTree->Left()->MinVals()[1];
  double minValsL2 = testDTree->Left()->MinVals()[2];

  double maxValsR0 = testDTree->Right()->MaxVals()[0];
  double maxValsR1 = testDTree->Right()->MaxVals()[1];
  double maxValsR2 = testDTree->Right()->MaxVals()[2];
  double minValsR0 = testDTree->Right()->MinVals()[0];
  double minValsR1 = testDTree->Right()->MinVals()[1];
  double minValsR2 = testDTree->Right()->MinVals()[2];

  // Delete the original tree.
  delete testDTree;

  // Test the data of copied tree (using copy constructor).
  REQUIRE(testDTree2.MaxVals()[0] == maxVals0);
  REQUIRE(testDTree2.MinVals()[0] == minVals0);
  REQUIRE(testDTree2.MaxVals()[1] == maxVals1);
  REQUIRE(testDTree2.MinVals()[1] == minVals1);
  REQUIRE(testDTree2.MaxVals()[2] == maxVals2);
  REQUIRE(testDTree2.MinVals()[2] == minVals2);

  // Test the data of the copied tree (using the copy operator).
  REQUIRE(testDTree3.MaxVals()[0] == maxVals0);
  REQUIRE(testDTree3.MinVals()[0] == minVals0);
  REQUIRE(testDTree3.MaxVals()[1] == maxVals1);
  REQUIRE(testDTree3.MinVals()[1] == minVals1);
  REQUIRE(testDTree3.MaxVals()[2] == maxVals2);
  REQUIRE(testDTree3.MinVals()[2] == minVals2);

  // Test the structure of the tree copied using the copy constructor.
  REQUIRE(testDTree2.Left()->Left() == NULL);
  REQUIRE(testDTree2.Left()->Right() == NULL);
  REQUIRE(testDTree2.Right()->Left()->Left() == NULL);
  REQUIRE(testDTree2.Right()->Left()->Right() == NULL);
  REQUIRE(testDTree2.Right()->Right()->Left() == NULL);
  REQUIRE(testDTree2.Right()->Right()->Right() == NULL);

  // Test the structure of the tree copied using the copy operator.
  REQUIRE(testDTree3.Left()->Left() == NULL);
  REQUIRE(testDTree3.Left()->Right() == NULL);
  REQUIRE(testDTree3.Right()->Left()->Left() == NULL);
  REQUIRE(testDTree3.Right()->Left()->Right() == NULL);
  REQUIRE(testDTree3.Right()->Right()->Left() == NULL);
  REQUIRE(testDTree3.Right()->Right()->Right() == NULL);

  // Test the data of the tree copied using the copy constructor.
  REQUIRE(testDTree2.Left()->MaxVals()[0] == maxValsL0);
  REQUIRE(testDTree2.Left()->MaxVals()[1] == maxValsL1);
  REQUIRE(testDTree2.Left()->MaxVals()[2] == maxValsL2);
  REQUIRE(testDTree2.Left()->MinVals()[0] == minValsL0);
  REQUIRE(testDTree2.Left()->MinVals()[1] == minValsL1);
  REQUIRE(testDTree2.Left()->MinVals()[2] == minValsL2);
  REQUIRE(testDTree2.Right()->MaxVals()[0] == maxValsR0);
  REQUIRE(testDTree2.Right()->MaxVals()[1] == maxValsR1);
  REQUIRE(testDTree2.Right()->MaxVals()[2] == maxValsR2);
  REQUIRE(testDTree2.Right()->MinVals()[0] == minValsR0);
  REQUIRE(testDTree2.Right()->MinVals()[1] == minValsR1);
  REQUIRE(testDTree2.Right()->MinVals()[2] == minValsR2);
  REQUIRE(testDTree2.SplitDim() == 2);
  REQUIRE(testDTree2.SplitValue() == Approx(5.5).epsilon(1e-7));
  REQUIRE(testDTree2.Right()->SplitDim() == 1);
  REQUIRE(testDTree2.Right()->SplitValue() == Approx(0.5).epsilon(1e-7));

  // Test the data of the tree copied using the copy operator.
  REQUIRE(testDTree3.Left()->MaxVals()[0] == maxValsL0);
  REQUIRE(testDTree3.Left()->MaxVals()[1] == maxValsL1);
  REQUIRE(testDTree3.Left()->MaxVals()[2] == maxValsL2);
  REQUIRE(testDTree3.Left()->MinVals()[0] == minValsL0);
  REQUIRE(testDTree3.Left()->MinVals()[1] == minValsL1);
  REQUIRE(testDTree3.Left()->MinVals()[2] == minValsL2);
  REQUIRE(testDTree3.Right()->MaxVals()[0] == maxValsR0);
  REQUIRE(testDTree3.Right()->MaxVals()[1] == maxValsR1);
  REQUIRE(testDTree3.Right()->MaxVals()[2] == maxValsR2);
  REQUIRE(testDTree3.Right()->MinVals()[0] == minValsR0);
  REQUIRE(testDTree3.Right()->MinVals()[1] == minValsR1);
  REQUIRE(testDTree3.Right()->MinVals()[2] == minValsR2);
  REQUIRE(testDTree3.SplitDim() == 2);
  REQUIRE(testDTree3.SplitValue() == Approx(5.5).epsilon(1e-7));
  REQUIRE(testDTree3.Right()->SplitDim() == 1);
  REQUIRE(testDTree3.Right()->SplitValue() == Approx(0.5).epsilon(1e-7));
}

// Test the move constructor.
TEST_CASE("MoveConstructorTest", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  // Construct another DTree for testing the children.
  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::mat> *testDTree = new DTree<arma::mat>(testData);
  testDTree->Grow(testData, oTest, false, 2, 1);

  double maxVals0 = testDTree->MaxVals()[0];
  double maxVals1 = testDTree->MaxVals()[1];
  double maxVals2 = testDTree->MaxVals()[2];
  double minVals0 = testDTree->MinVals()[0];
  double minVals1 = testDTree->MinVals()[1];
  double minVals2 = testDTree->MinVals()[2];

  double maxValsL0 = testDTree->Left()->MaxVals()[0];
  double maxValsL1 = testDTree->Left()->MaxVals()[1];
  double maxValsL2 = testDTree->Left()->MaxVals()[2];
  double minValsL0 = testDTree->Left()->MinVals()[0];
  double minValsL1 = testDTree->Left()->MinVals()[1];
  double minValsL2 = testDTree->Left()->MinVals()[2];

  double maxValsR0 = testDTree->Right()->MaxVals()[0];
  double maxValsR1 = testDTree->Right()->MaxVals()[1];
  double maxValsR2 = testDTree->Right()->MaxVals()[2];
  double minValsR0 = testDTree->Right()->MinVals()[0];
  double minValsR1 = testDTree->Right()->MinVals()[1];
  double minValsR2 = testDTree->Right()->MinVals()[2];

  // Construct a new tree using the move constructor.
  DTree<arma::mat> testDTree2(std::move(*testDTree));

  // Check default values of the original tree.
  REQUIRE(testDTree->LogNegError() == -DBL_MAX);
  REQUIRE(testDTree->Left() == (DTree<arma::mat>*) NULL);
  REQUIRE(testDTree->Right() == (DTree<arma::mat>*) NULL);

  // Delete the original tree.
  delete testDTree;

  // Test the data of the moved tree.
  REQUIRE(testDTree2.MaxVals()[0] == maxVals0);
  REQUIRE(testDTree2.MinVals()[0] == minVals0);
  REQUIRE(testDTree2.MaxVals()[1] == maxVals1);
  REQUIRE(testDTree2.MinVals()[1] == minVals1);
  REQUIRE(testDTree2.MaxVals()[2] == maxVals2);
  REQUIRE(testDTree2.MinVals()[2] == minVals2);

  // Test the structure of the moved tree.
  REQUIRE(testDTree2.Left()->Left() == NULL);
  REQUIRE(testDTree2.Left()->Right() == NULL);
  REQUIRE(testDTree2.Right()->Left()->Left() == NULL);
  REQUIRE(testDTree2.Right()->Left()->Right() == NULL);
  REQUIRE(testDTree2.Right()->Right()->Left() == NULL);
  REQUIRE(testDTree2.Right()->Right()->Right() == NULL);

  // Test the data of the moved tree.
  REQUIRE(testDTree2.Left()->MaxVals()[0] == maxValsL0);
  REQUIRE(testDTree2.Left()->MaxVals()[1] == maxValsL1);
  REQUIRE(testDTree2.Left()->MaxVals()[2] == maxValsL2);
  REQUIRE(testDTree2.Left()->MinVals()[0] == minValsL0);
  REQUIRE(testDTree2.Left()->MinVals()[1] == minValsL1);
  REQUIRE(testDTree2.Left()->MinVals()[2] == minValsL2);
  REQUIRE(testDTree2.Right()->MaxVals()[0] == maxValsR0);
  REQUIRE(testDTree2.Right()->MaxVals()[1] == maxValsR1);
  REQUIRE(testDTree2.Right()->MaxVals()[2] == maxValsR2);
  REQUIRE(testDTree2.Right()->MinVals()[0] == minValsR0);
  REQUIRE(testDTree2.Right()->MinVals()[1] == minValsR1);
  REQUIRE(testDTree2.Right()->MinVals()[2] == minValsR2);
  REQUIRE(testDTree2.SplitDim() == 2);
  REQUIRE(testDTree2.SplitValue() == Approx(5.5).epsilon(1e-7));
  REQUIRE(testDTree2.Right()->SplitDim() == 1);
  REQUIRE(testDTree2.Right()->SplitValue() == Approx(0.5).epsilon(1e-7));
}

// Test the move operator.
TEST_CASE("MoveOperatorTest", "[DETTest]")
{
  arma::mat testData(3, 5);

  testData << 4 << 5 << 7 << 3 << 5 << arma::endr
           << 5 << 0 << 1 << 7 << 1 << arma::endr
           << 5 << 6 << 7 << 1 << 8 << arma::endr;

  // Construct another DTree for testing the children.
  arma::Col<size_t> oTest(5);
  oTest << 0 << 1 << 2 << 3 << 4;

  DTree<arma::mat> *testDTree = new DTree<arma::mat>(testData);
  testDTree->Grow(testData, oTest, false, 2, 1);

  double maxVals0 = testDTree->MaxVals()[0];
  double maxVals1 = testDTree->MaxVals()[1];
  double maxVals2 = testDTree->MaxVals()[2];
  double minVals0 = testDTree->MinVals()[0];
  double minVals1 = testDTree->MinVals()[1];
  double minVals2 = testDTree->MinVals()[2];

  double maxValsL0 = testDTree->Left()->MaxVals()[0];
  double maxValsL1 = testDTree->Left()->MaxVals()[1];
  double maxValsL2 = testDTree->Left()->MaxVals()[2];
  double minValsL0 = testDTree->Left()->MinVals()[0];
  double minValsL1 = testDTree->Left()->MinVals()[1];
  double minValsL2 = testDTree->Left()->MinVals()[2];

  double maxValsR0 = testDTree->Right()->MaxVals()[0];
  double maxValsR1 = testDTree->Right()->MaxVals()[1];
  double maxValsR2 = testDTree->Right()->MaxVals()[2];
  double minValsR0 = testDTree->Right()->MinVals()[0];
  double minValsR1 = testDTree->Right()->MinVals()[1];
  double minValsR2 = testDTree->Right()->MinVals()[2];

  // Construct a new tree using the move constructor.
  DTree<arma::mat> testDTree2 = std::move(*testDTree);

  // Check default values of the original tree.
  REQUIRE(testDTree->LogNegError() == -DBL_MAX);
  REQUIRE(testDTree->Left() == (DTree<arma::mat>*) NULL);
  REQUIRE(testDTree->Right() == (DTree<arma::mat>*) NULL);

  // Delete the original tree.
  delete testDTree;

  // Test the data of the moved tree.
  REQUIRE(testDTree2.MaxVals()[0] == maxVals0);
  REQUIRE(testDTree2.MinVals()[0] == minVals0);
  REQUIRE(testDTree2.MaxVals()[1] == maxVals1);
  REQUIRE(testDTree2.MinVals()[1] == minVals1);
  REQUIRE(testDTree2.MaxVals()[2] == maxVals2);
  REQUIRE(testDTree2.MinVals()[2] == minVals2);

  // Test the structure of the moved tree.
  REQUIRE(testDTree2.Left()->Left() == NULL);
  REQUIRE(testDTree2.Left()->Right() == NULL);
  REQUIRE(testDTree2.Right()->Left()->Left() == NULL);
  REQUIRE(testDTree2.Right()->Left()->Right() == NULL);
  REQUIRE(testDTree2.Right()->Right()->Left() == NULL);
  REQUIRE(testDTree2.Right()->Right()->Right() == NULL);

  // Test the data of moved tree.
  REQUIRE(testDTree2.Left()->MaxVals()[0] == maxValsL0);
  REQUIRE(testDTree2.Left()->MaxVals()[1] == maxValsL1);
  REQUIRE(testDTree2.Left()->MaxVals()[2] == maxValsL2);
  REQUIRE(testDTree2.Left()->MinVals()[0] == minValsL0);
  REQUIRE(testDTree2.Left()->MinVals()[1] == minValsL1);
  REQUIRE(testDTree2.Left()->MinVals()[2] == minValsL2);
  REQUIRE(testDTree2.Right()->MaxVals()[0] == maxValsR0);
  REQUIRE(testDTree2.Right()->MaxVals()[1] == maxValsR1);
  REQUIRE(testDTree2.Right()->MaxVals()[2] == maxValsR2);
  REQUIRE(testDTree2.Right()->MinVals()[0] == minValsR0);
  REQUIRE(testDTree2.Right()->MinVals()[1] == minValsR1);
  REQUIRE(testDTree2.Right()->MinVals()[2] == minValsR2);
  REQUIRE(testDTree2.SplitDim() == 2);
  REQUIRE(testDTree2.SplitValue() == Approx(5.5).epsilon(1e-7));
  REQUIRE(testDTree2.Right()->SplitDim() == 1);
  REQUIRE(testDTree2.Right()->SplitValue() == Approx(0.5).epsilon(1e-7));
}
