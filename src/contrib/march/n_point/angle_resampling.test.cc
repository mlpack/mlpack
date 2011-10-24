/*
 *  angle_resampling.test.cc
 *  
 *
 *  Created by William March on 10/10/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "test_angle_resampling.h"



/*
#define BOOST_TEST_MODULE angle_resampling_test
//#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>




BOOST_AUTO_TEST_CASE(TestCaseAngleResampling) {
  
  printf("making test class\n");
  npt::TestAngleResampling angle_test;
  printf("running tests\n");
  bool tests_passed = angle_test.StressTestMain();
  printf("tests finished\n");
  BOOST_CHECK_EQUAL(tests_passed, true);
  
}
 
 */

int main (int argc, char* argv[]) {

  mlpack::CLI::ParseCommandLine(argc, argv);
  
  npt::TestAngleResampling angle_test;

  bool tests_passed = angle_test.StressTestMain();
  
  printf("test_passed: %d\n", tests_passed);

  return 0;
  
}


//BOOST_AUTO_TEST_SUITE_END()

