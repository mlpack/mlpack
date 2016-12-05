/**
 * @file sparse_sort_test.cpp
 * @author Ryan Curtin
 *
 * Some tests for the Armadillo sparse sorting code.
 */
#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;

BOOST_AUTO_TEST_SUITE(SparseSortTest);

BOOST_AUTO_TEST_CASE(SimpleSparseVectorSortTest)
{
  sp_vec sc(10);
  sc[2] = 10.0;
  sc[5] = 3.0;
  sc[6] = -1.0;
  sc[8] = 2.5;
  sc[9] = 0.3;

  // Sort the vector.
  sp_vec out = sort(sc);

  // Check that the output is in the right order.
  BOOST_REQUIRE_CLOSE((double) out[0], -1.0, 1e-5);
  BOOST_REQUIRE_SMALL((double) out[1], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[2], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[3], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[4], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[5], 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[6], 0.3, 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[7], 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[8], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[9], 10.0, 1e-5);
}

BOOST_AUTO_TEST_CASE(RandomSparseVectorSortTest)
{
  sp_vec sc;
  sc.sprandu(1000, 1, 0.6);

  vec c(sc);

  // Sort both.
  sp_vec sout = sort(sc);
  vec out = sort(c);

  // Check that the results are equivalent.
  for (size_t i = 0; i < 1000; ++i)
  {
    if (out[i] < 1e-5)
      BOOST_REQUIRE_SMALL((double) sout[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(out[i], (double) sout[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SimpleSparseRowSortTest)
{
  sp_rowvec sc(10);
  sc[2] = 10.0;
  sc[5] = 3.0;
  sc[6] = -1.0;
  sc[8] = 2.5;
  sc[9] = 0.3;

  // Sort the vector.
  sp_rowvec out = sort(sc);

  // Check that the output is in the right order.
  BOOST_REQUIRE_CLOSE((double) out[0], -1.0, 1e-5);
  BOOST_REQUIRE_SMALL((double) out[1], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[2], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[3], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[4], 1e-5);
  BOOST_REQUIRE_SMALL((double) out[5], 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[6], 0.3, 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[7], 2.5, 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[8], 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE((double) out[9], 10.0, 1e-5);
}

BOOST_AUTO_TEST_CASE(RandomSparseRowSortTest)
{
  sp_rowvec sc;
  sc.sprandu(1, 1000, 0.6);

  rowvec c(sc);

  // Sort both.
  sp_rowvec sout = sort(sc);
  rowvec out = sort(c);

  // Check that the results are equivalent.
  for (size_t i = 0; i < 1000; ++i)
  {
    if (out[i] < 1e-5)
      BOOST_REQUIRE_SMALL((double) sout[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(out[i], (double) sout[i], 1e-5);
  }
}

// These two don't work: we need an overload of sort() for sparse matrices that
// takes a dimension to sort in, just like the dense version.
/*
BOOST_AUTO_TEST_CASE(SparseRandomMatrixSortTest)
{
  sp_mat sc;
  sc.sprandu(50, 50, 0.6);

  mat c(sc);

  // Sort both.
  sp_mat sout = sort(sc, 0);
  mat out = sort(c, 0);

  // Check that both results are equivalent.
  for (size_t i = 0; i < 1000; ++i)
  {
    if (out[i] < 1e-5)
      BOOST_REQUIRE_SMALL((double) sout[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(out[i], (double) sout[i], 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SparseRandomMatrixSortRowTest)
{
  sp_mat sc;
  sc.sprandu(50, 50, 0.6);

  mat c(sc);

  // Sort both.
  sp_mat sout = sort(sc, 1);
  mat out = sort(c, 1);

  // Check that both results are equivalent.
  for (size_t i = 0; i < 1000; ++i)
  {
    if (out[i] < 1e-5)
      BOOST_REQUIRE_SMALL((double) sout[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(out[i], (double) sout[i], 1e-5);
  }
}
*/

BOOST_AUTO_TEST_SUITE_END();
