/**
 * @file det_test.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * Unit tests for the functions of the class DTree
 * and the utility functions using this class.
 */

#define protected public
#define private public
#include <mlpack/methods/det/dtree.hpp>
#include <mlpack/methods/det/dt_utils.hpp>
#undef protected
#undef private

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::det;
using namespace std;

BOOST_AUTO_TEST_SUITE(DETTest);

// Testing functions of the DTree class

typedef arma::mat MatType;
typedef arma::vec VecType;


// the private functions

BOOST_AUTO_TEST_CASE(TestGetMaxMinVals)
{
  DTree<>* testDTree = new DTree<>();

  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  VecType* max_vals = new VecType();
  VecType* min_vals = new VecType();

  testDTree->GetMaxMinVals_(&test_data, max_vals, min_vals);

  BOOST_REQUIRE((*max_vals)[0] == 7);
  BOOST_REQUIRE((*min_vals)[0] == 3);
  BOOST_REQUIRE((*max_vals)[1] == 7);
  BOOST_REQUIRE((*min_vals)[1] == 0);
  BOOST_REQUIRE((*max_vals)[2] == 8);
  BOOST_REQUIRE((*min_vals)[2] == 1);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestComputeNodeError)
{
  VecType* max_vals = new VecType(3);
  VecType* min_vals = new VecType(3);

  *max_vals << 7 << 7 << 8;
  *min_vals << 3 << 0 << 1;

  DTree<>* testDTree = new DTree<>(max_vals, min_vals, 5);
  double true_node_error = -1.0 * exp(-(double) log((double) 4.0)
					   - (double) log((double) 7.0)
					   - (double) log((double) 7.0));

  BOOST_REQUIRE_CLOSE(testDTree->error_, true_node_error, 1e-10);

  testDTree->start_ = 3;
  testDTree->end_ = 5;

  double node_error = -std::exp(testDTree->LogNegativeError(5));
  true_node_error = -1.0 * exp(2 * log((double) 2 / (double) 5)
			       -(double) log((double) 4.0)
			       - (double) log((double) 7.0)
			       - (double) log((double) 7.0));
  BOOST_REQUIRE_CLOSE(node_error, true_node_error, 1e-10);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestWithinRange)
{
  VecType* max_vals = new VecType(3);
  VecType* min_vals = new VecType(3);

  *max_vals << 7 << 7 << 8;
  *min_vals << 3 << 0 << 1;

  DTree<>* testDTree = new DTree<>(max_vals, min_vals, 5);

  VecType test_query(3);
  test_query << 4.5 << 2.5 << 2;

  BOOST_REQUIRE(testDTree->WithinRange_(&test_query));

  test_query << 8.5 << 2.5 << 2;

  BOOST_REQUIRE(!testDTree->WithinRange_(&test_query));

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestFindSplit)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<>* testDTree = new DTree<>(&test_data);

  size_t ob_dim, true_dim, ob_ind, true_ind;
  long double true_left_error, ob_left_error, true_right_error, ob_right_error;

  true_dim = 2;
  true_ind = 1;
  true_left_error = -1.0 * exp(2 * log(2.0 / 5.0) - (log(7.0) + log(4.0) +
      log(4.5)));
  true_right_error = -1.0 * exp(2 * log(3.0 / 5.0) - (log(7.0) + log(4.0) +
      log(2.5)));

  BOOST_REQUIRE(testDTree->FindSplit_
		(test_data, &ob_dim, &ob_ind, &ob_left_error,
		 &ob_right_error, 2, 1));

  BOOST_REQUIRE(true_dim == ob_dim);
  BOOST_REQUIRE(true_ind == ob_ind);

  BOOST_REQUIRE_CLOSE(true_left_error, ob_left_error, 1e-10);
  BOOST_REQUIRE_CLOSE(true_right_error, ob_right_error, 1e-10);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestSplitData)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<>* testDTree = new DTree<>(&test_data);

  arma::Col<size_t> o_test(5);
  o_test << 1 << 2 << 3 << 4 << 5;

  size_t split_dim = 2, split_ind = 1;
  double true_split_val, ob_split_val, true_lsplit_val, ob_lsplit_val,
    true_rsplit_val, ob_rsplit_val;

  true_lsplit_val = 5;
  true_rsplit_val = 6;
  true_split_val = (true_lsplit_val + true_rsplit_val) / 2;

  testDTree->SplitData_(&test_data, split_dim, split_ind,
			&o_test, &ob_split_val,
			&ob_lsplit_val, &ob_rsplit_val);

  BOOST_REQUIRE(o_test[0] == 1 && o_test[1] == 4
		&& o_test[2] == 3 && o_test[3] == 2
		&& o_test[4] == 5);

  BOOST_REQUIRE(true_split_val == ob_split_val);
  BOOST_REQUIRE(true_lsplit_val == ob_lsplit_val);
  BOOST_REQUIRE(true_rsplit_val == ob_rsplit_val);

  delete testDTree;
}

// the public functions

BOOST_AUTO_TEST_CASE(TestGrow)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::Col<size_t> o_test(5);
  o_test << 0 << 1 << 2 << 3 << 4;

  double root_error, l_error, r_error, rl_error, rr_error;

  root_error = -1.0 * exp(-log(4.0) - log(7.0) - log(7.0));

  l_error = -1.0 * exp(2 * log(2.0 / 5.0) - (log(7.0) + log(4.0) + log(4.5)));
  r_error =  -1.0 * exp(2 * log(3.0 / 5.0) - (log(7.0) + log(4.0) + log(2.5)));

  rl_error = -1.0 * exp(2 * log(1.0 / 5.0) - (log(0.5) + log(4.0) + log(2.5)));
  rr_error = -1.0 * exp(2 * log(2.0 / 5.0) - (log(6.5) + log(4.0) + log(2.5)));

  DTree<>* testDTree = new DTree<>(&test_data);
  long double alpha = testDTree->Grow(&test_data, &o_test, false, 2, 1);

  BOOST_REQUIRE(o_test[0] == 0 && o_test[1] == 3
		&& o_test[2] == 1 && o_test[3] == 2
		&& o_test[4] == 4);

  // test the structure of the tree
  BOOST_REQUIRE(testDTree->left()->left() == NULL);
  BOOST_REQUIRE(testDTree->left()->right() == NULL);
  BOOST_REQUIRE(testDTree->right()->left()->left() == NULL);
  BOOST_REQUIRE(testDTree->right()->left()->right() == NULL);
  BOOST_REQUIRE(testDTree->right()->right()->left() == NULL);
  BOOST_REQUIRE(testDTree->right()->right()->right() == NULL);

  BOOST_REQUIRE(testDTree->subtree_leaves() == 3);

  BOOST_REQUIRE(testDTree->split_dim() == 2);
  BOOST_REQUIRE_CLOSE(testDTree->split_value(), (float) 5.5, (float) 1e-5);
  BOOST_REQUIRE(testDTree->right()->split_dim() == 1);
  BOOST_REQUIRE_CLOSE(testDTree->right()->split_value(),
		      (float) 0.5, (float) 1e-5);

  // test node errors for every node
  BOOST_REQUIRE_CLOSE(testDTree->error_, root_error, 1e-10);
  BOOST_REQUIRE_CLOSE(testDTree->left()->error_, l_error, 1e-10);
  BOOST_REQUIRE_CLOSE(testDTree->right()->error_, r_error, 1e-10);
  BOOST_REQUIRE_CLOSE(testDTree->right()->left()->error_, rl_error, 1e-10);
  BOOST_REQUIRE_CLOSE(testDTree->right()->right()->error_, rr_error, 1e-10);


  // test alpha
  long double root_alpha, r_alpha;
  root_alpha = (root_error - (l_error + rl_error + rr_error)) / 2;
  r_alpha = r_error - (rl_error + rr_error);

  BOOST_REQUIRE_CLOSE(alpha, min(root_alpha, r_alpha), 1e-10);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestPruneAndUpdate)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  arma::Col<size_t> o_test(5);
  o_test << 0 << 1 << 2 << 3 << 4;
  DTree<>* testDTree = new DTree<>(&test_data);
  long double alpha = testDTree->Grow(&test_data, &o_test,
				      false, 2, 1);
  alpha = testDTree->PruneAndUpdate(alpha, false);

  BOOST_REQUIRE_CLOSE(alpha, numeric_limits<long double>::max(), 1e-10);
  BOOST_REQUIRE(testDTree->subtree_leaves() == 1);

  long double root_error = -1.0 * exp(-log(4.0) - log(7.0) - log(7.0));

  BOOST_REQUIRE_CLOSE(testDTree->error(), root_error, 1e-10);
  BOOST_REQUIRE_CLOSE(testDTree->subtree_leaves_error(), root_error, 1e-10);
  BOOST_REQUIRE(testDTree->left() == NULL);
  BOOST_REQUIRE(testDTree->right() == NULL);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestComputeValue)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  VecType q1(3), q2(3), q3(3), q4(3);

  q1 << 4 << 2 << 2;
  q2 << 5 << 0.25 << 6;
  q3 << 5 << 3 << 7;
  q4 << 2 << 3 << 3;

  arma::Col<size_t> o_test(5);
  o_test << 0 << 1 << 2 << 3 << 4;

  DTree<>* testDTree = new DTree<>(&test_data);
  long double alpha = testDTree->Grow(&test_data, &o_test,
				      false, 2, 1);

  double d1, d2, d3;
  d1 = (2.0 / 5.0) / exp(log(4.0) + log(7.0) + log(4.5));
  d2 = (1.0 / 5.0) / exp(log(4.0) + log(0.5) + log(2.5));
  d3 = (2.0 / 5.0) / exp(log(4.0) + log(6.5) + log(2.5));

  BOOST_REQUIRE_CLOSE(d1, testDTree->ComputeValue(&q1), 1e-10);
  BOOST_REQUIRE_CLOSE(d2, testDTree->ComputeValue(&q2), 1e-10);
  BOOST_REQUIRE_CLOSE(d3, testDTree->ComputeValue(&q3), 1e-10);
  BOOST_REQUIRE_CLOSE((long double) 0.0, testDTree->ComputeValue(&q4), 1e-10);

  alpha = testDTree->PruneAndUpdate(alpha, false);

  long double d = 1.0 / exp(log(4.0) + log(7.0) + log(7.0));

  BOOST_REQUIRE_CLOSE(d, testDTree->ComputeValue(&q1), 1e-10);
  BOOST_REQUIRE_CLOSE(d, testDTree->ComputeValue(&q2), 1e-10);
  BOOST_REQUIRE_CLOSE(d, testDTree->ComputeValue(&q3), 1e-10);
  BOOST_REQUIRE_CLOSE((long double) 0.0, testDTree->ComputeValue(&q4), 1e-10);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestVariableImportance)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  long double root_error, l_error, r_error, rl_error, rr_error;

  root_error = -1.0 * exp(-log(4.0) - log(7.0) - log(7.0));

  l_error = -1.0 * exp(2 * log(2.0 / 5.0) - (log(7.0) + log(4.0) + log(4.5)));
  r_error =  -1.0 * exp(2 * log(3.0 / 5.0) - (log(7.0) + log(4.0) + log(2.5)));

  rl_error = -1.0 * exp(2 * log(1.0 / 5.0) - (log(0.5) + log(4.0) + log(2.5)));
  rr_error = -1.0 * exp(2 * log(2.0 / 5.0) - (log(6.5) + log(4.0) + log(2.5)));

  arma::Col<size_t> o_test(5);
  o_test << 0 << 1 << 2 << 3 << 4;

  DTree<>* testDTree = new DTree<>(&test_data);
  testDTree->Grow(&test_data, &o_test,
		  false, 2, 1);

  arma::vec imps(3);
  imps.zeros();

  testDTree->ComputeVariableImportance(&imps);

  BOOST_REQUIRE_CLOSE((double) 0.0, imps[0], 1e-10);
  BOOST_REQUIRE_CLOSE((double) (r_error - (rl_error + rr_error)),
		      imps[1], 1e-10);
  BOOST_REQUIRE_CLOSE((double) (root_error - (l_error + r_error)),
		      imps[2], 1e-10);

  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestTagTree)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<>* testDTree = new DTree<>(&test_data);


  delete testDTree;
}

BOOST_AUTO_TEST_CASE(TestFindBucket)
{
  MatType test_data(3,5);

  test_data << 4 << 5 << 7 << 3 << 5 << arma::endr
	    << 5 << 0 << 1 << 7 << 1 << arma::endr
	    << 5 << 6 << 7 << 1 << 8 << arma::endr;

  DTree<>* testDTree = new DTree<>(&test_data);


  delete testDTree;
}

// Test functions in dt_utils.hpp

BOOST_AUTO_TEST_CASE(TestTrainer)
{

}

BOOST_AUTO_TEST_CASE(TestPrintVariableImportance)
{

}

BOOST_AUTO_TEST_CASE(TestPrintLeafMembership)
{

}

BOOST_AUTO_TEST_SUITE_END();
