/** @file armadillo_wrapper.test.cc
 *
 *  A sanity test to make sure that the Armadillo wrappers are
 *  working.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

// for BOOST testing
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "core/table/dense_point.h"

BOOST_AUTO_TEST_SUITE(TestSuiteArmadilloWrapper)
BOOST_AUTO_TEST_CASE(TestCaseArmadilloWrapper) {

  printf("Test to make sure that the pointer aliasing is working correctly.\n");
  core::table::DensePoint vec;
  vec.Init(50);
  vec.SetZero();
  arma::vec vec_alias;
  core::table::DensePointToArmaVec(vec, &vec_alias);
  printf(
    "Pointer address of the original: %p\n", vec.ptr());
  printf(
    "Pointer address of the alias: %p\n", vec_alias.memptr());
  BOOST_ASSERT(vec.ptr() == vec_alias.memptr());
}
BOOST_AUTO_TEST_SUITE_END()
