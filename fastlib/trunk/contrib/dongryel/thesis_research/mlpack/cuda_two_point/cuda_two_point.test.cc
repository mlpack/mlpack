/*
 *  two_point.test.cc
 *  
 *
 *  Created by William March on 9/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


// for BOOST testing
#define BOOST_TEST_MODULE cuda_two_point_test
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "mlpack/cuda_two_point/test_cuda_two_point.h"
#include <omp.h>
#include <time.h>

namespace mlpack {
  namespace cuda_two_point {
    namespace test_cuda_two_point {
      int num_dimensions_;
      int num_points_;
    }
  }
}

BOOST_AUTO_TEST_SUITE(TestSuiteCudaTwoPoint)

BOOST_AUTO_TEST_CASE(TestCaseTwoPoint) {
  
  // Call the tests.
  omp_set_num_threads(1);
  mlpack::cuda_two_point::TestCudaTwoPoint cuda_two_point_test;
  int main_val = cuda_two_point_test.StressTestMain();
  
  BOOST_CHECK_EQUAL(main_val, 0);
  
  std::cout << "All tests passed.\n";
  
}

BOOST_AUTO_TEST_SUITE_END()
