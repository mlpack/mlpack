/*
 *  single_bandwidth.test.cc
 *  
 *
 *  Created by William March on 10/4/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "test_single_bandwidth.h"


#define BOOST_TEST_MODULE single_bandwidth_test
//#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#ifdef BOOST_TEST_MAIN
#error "shit"
#endif




//BOOST_AUTO_TEST_SUITE(TestSuiteSingleBandwidth)

BOOST_AUTO_TEST_CASE(TestCaseSingleBandwidth) {

  printf("making test class\n");
  //bool tests_passed = true;
  npt::TestSingleBandwidth single_test;
  printf("running tests\n");
  bool tests_passed = single_test.StressTestMain();
  printf("tests finished\n");
  BOOST_CHECK_EQUAL(tests_passed, true);
 

  
  
  
  
}

//BOOST_AUTO_TEST_SUITE_END()

