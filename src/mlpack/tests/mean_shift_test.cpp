/**
 * @file mean_shift_test.cpp
 * @author Shangtong Zhang
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/mean_shift/mean_shift.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::meanshift;

BOOST_AUTO_TEST_SUITE(MeanShiftTest);

// Generate dataset; written transposed because it's easier to read.
arma::mat meanShiftData("  0.0   0.0;" // Class 1.
                     "  0.3   0.4;"
                     "  0.1   0.0;"
                     "  0.1   0.3;"
                     " -0.2  -0.2;"
                     " -0.1   0.3;"
                     " -0.4   0.1;"
                     "  0.2  -0.1;"
                     "  0.3   0.0;"
                     " -0.3  -0.3;"
                     "  0.1  -0.1;"
                     "  0.2  -0.3;"
                     " -0.3   0.2;"
                     " 10.0  10.0;" // Class 2.
                     " 10.1   9.9;"
                     "  9.9  10.0;"
                     " 10.2   9.7;"
                     " 10.2   9.8;"
                     "  9.7  10.3;"
                     "  9.9  10.1;"
                     "-10.0   5.0;" // Class 3.
                     " -9.8   5.1;"
                     " -9.9   4.9;"
                     "-10.0   4.9;"
                     "-10.2   5.2;"
                     "-10.1   5.1;"
                     "-10.3   5.3;"
                     "-10.0   4.8;"
                     " -9.6   5.0;"
                     " -9.8   5.1;");


/**
 * 30-point 3-class test case for Mean Shift.
 */
BOOST_AUTO_TEST_CASE(MeanShiftSimpleTest) {
 
  MeanShift<> meanShift;
  
  arma::Col<size_t> assignments;
  arma::mat centroids;
  meanShift.Cluster((arma::mat) trans(meanShiftData), assignments, centroids);
  
  // Now make sure we got it all right.  There is no restriction on how the
  // clusters are ordered, so we have to be careful about that.
  size_t firstClass = assignments(0);
  
  for (size_t i = 1; i < 13; i++)
    BOOST_REQUIRE_EQUAL(assignments(i), firstClass);
  
  size_t secondClass = assignments(13);
  
  // To ensure that class 1 != class 2.
  BOOST_REQUIRE_NE(firstClass, secondClass);
  
  for (size_t i = 13; i < 20; i++)
    BOOST_REQUIRE_EQUAL(assignments(i), secondClass);
  
  size_t thirdClass = assignments(20);
  
  // To ensure that this is the third class which we haven't seen yet.
  BOOST_REQUIRE_NE(firstClass, thirdClass);
  BOOST_REQUIRE_NE(secondClass, thirdClass);
  
  for (size_t i = 20; i < 30; i++)
    BOOST_REQUIRE_EQUAL(assignments(i), thirdClass);
  
}

/**
 * When duplicate thresh is set to 0, any centroids shouldn't removed.
 */
BOOST_AUTO_TEST_CASE(ZeroDuplicateThreshTest) {
    
    // Set the duplicate thresh to 0
    MeanShift<> meanShift(0);
    
    arma::Col<size_t> assignments;
    arma::mat centroids;
    meanShift.Cluster((arma::mat) trans(meanShiftData), assignments, centroids);
    
    /**
     * Make sure the number of centroids is equal to 
     * the number of vectors in dataset.
     */
    BOOST_REQUIRE_EQUAL(centroids.n_cols, meanShiftData.n_rows);
}

BOOST_AUTO_TEST_SUITE_END();