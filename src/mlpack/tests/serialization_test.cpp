/**
 * @file serialization_test.cpp
 * @author Ryan Curtin
 *
 * Test serialization of mlpack objects.
 */
#include <boost/serialization/serialization.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace boost;
using namespace boost::archive;
using namespace boost::serialization;
using namespace std;

BOOST_AUTO_TEST_SUITE(SerializationTest);

// Test function for loading and saving Armadillo objects.
template<typename MatType,
         typename IArchiveType,
         typename OArchiveType>
void TestArmadilloSerialization(MatType& x)
{
  // First save it.
  ofstream ofs("test");
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << BOOST_SERIALIZATION_NVP(x);
  }
  catch (archive_exception& e)
  {
    success = false;
  }

  BOOST_REQUIRE_EQUAL(success, true);
  ofs.close();

  // Now load it.
  MatType orig(x);
  success = true;
  ifstream ifs("test");
  IArchiveType i(ifs);

  try
  {
    i >> BOOST_SERIALIZATION_NVP(x);
  }
  catch (archive_exception& e)
  {
    success = false;
  }

  BOOST_REQUIRE_EQUAL(success, true);

  BOOST_REQUIRE_EQUAL(x.n_rows, orig.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_cols, orig.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_elem, orig.n_elem);

  for (size_t i = 0; i < x.n_cols; ++i)
    for (size_t j = 0; j < x.n_rows; ++j)
      if (double(orig(j, i)) == 0.0)
        BOOST_REQUIRE_SMALL(double(x(j, i)), 1e-8);
      else
        BOOST_REQUIRE_CLOSE(double(orig(j, i)), double(x(j, i)), 1e-8);
}

// Test all serialization strategies.
template<typename MatType>
void TestAllArmadilloSerialization(MatType& x)
{
  TestArmadilloSerialization<MatType, xml_iarchive, xml_oarchive>(x);
  TestArmadilloSerialization<MatType, text_iarchive, text_oarchive>(x);
  TestArmadilloSerialization<MatType, binary_iarchive, binary_oarchive>(x);
}

/**
 * Can we load and save an Armadillo matrix?
 */
BOOST_AUTO_TEST_CASE(MatrixSerializeXMLTest)
{
  arma::mat m;
  m.randu(50, 50);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
BOOST_AUTO_TEST_CASE(ColSerializeXMLTest)
{
  arma::vec m;
  m.randu(50, 1);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(RowSerializeXMLTest)
{
  arma::rowvec m;
  m.randu(1, 50);
  TestAllArmadilloSerialization(m);
}

// A quick test with an empty matrix.
BOOST_AUTO_TEST_CASE(EmptyMatrixSerializeTest)
{
  arma::mat m;
  TestAllArmadilloSerialization(m);
}

/**
 * Can we load and save a sparse Armadillo matrix?
 */
BOOST_AUTO_TEST_CASE(SparseMatrixSerializeXMLTest)
{
  arma::sp_mat m;
  m.sprandu(50, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about columns?
 */
BOOST_AUTO_TEST_CASE(SparseColSerializeXMLTest)
{
  arma::sp_vec m;
  m.sprandu(50, 1, 0.3);
  TestAllArmadilloSerialization(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(SparseRowSerializeXMLTest)
{
  arma::sp_rowvec m;
  m.sprandu(1, 50, 0.3);
  TestAllArmadilloSerialization(m);
}

// A quick test with an empty matrix.
BOOST_AUTO_TEST_CASE(EmptySparseMatrixSerializeTest)
{
  arma::sp_mat m;
  TestAllArmadilloSerialization(m);
}

BOOST_AUTO_TEST_SUITE_END();
