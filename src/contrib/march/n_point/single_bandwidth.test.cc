/*
 *  single_bandwidth.test.cc
 *  
 *
 *  Created by William March on 10/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "test_single_bandwidth.h"
//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int_distribution.hpp>
//#include <boost/random/uniform_real_distribution.hpp>
//#include <boost/random/uniform_01.hpp>


#define BOOST_TEST_MODULE single_bandwidth_test
//#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>




//BOOST_AUTO_TEST_SUITE(TestSuiteSingleBandwidth)

BOOST_AUTO_TEST_CASE(TestCaseSingleBandwidth) {

  
  printf("making test class\n");
  npt::TestSingleBandwidth single_test;
  printf("running tests\n");
  bool tests_passed = single_test.StressTestMain();
  printf("tests finished\n");
  BOOST_CHECK_EQUAL(tests_passed, true);
 
}

//BOOST_AUTO_TEST_SUITE_END()

