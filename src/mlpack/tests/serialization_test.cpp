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
      if (orig(j, i) == 0.0)
        BOOST_REQUIRE_SMALL(x(j, i), 1e-8);
      else
        BOOST_REQUIRE_CLOSE(orig(j, i), x(j, i), 1e-8);
}

/**
 * Can we load and save an Armadillo matrix from XML?
 */
BOOST_AUTO_TEST_CASE(MatrixSerializeXMLTest)
{
  arma::mat m;
  m.randu(50, 50);
  TestArmadilloSerialization<arma::mat, xml_iarchive, xml_oarchive>(m);
}

BOOST_AUTO_TEST_CASE(MatrixSerializeTextTest)
{
  arma::mat m;
  m.randu(50, 50);
  TestArmadilloSerialization<arma::mat, text_iarchive, text_oarchive>(m);
}

BOOST_AUTO_TEST_CASE(MatrixSerializeBinaryTest)
{
  arma::mat m;
  m.randu(50, 50);
  TestArmadilloSerialization<arma::mat, binary_iarchive, binary_oarchive>(m);
}

/**
 * How about columns?
 */
BOOST_AUTO_TEST_CASE(ColSerializeXMLTest)
{
  arma::vec m;
  m.randu(50, 1);
  TestArmadilloSerialization<arma::vec, xml_iarchive, xml_oarchive>(m);
}

BOOST_AUTO_TEST_CASE(ColSerializeTextTest)
{
  arma::vec m;
  m.randu(50, 1);
  TestArmadilloSerialization<arma::vec, text_iarchive, text_oarchive>(m);
}

BOOST_AUTO_TEST_CASE(ColSerializeBinaryTest)
{
  arma::vec m;
  m.randu(50, 1);
  TestArmadilloSerialization<arma::vec, binary_iarchive, binary_oarchive>(m);
}

/**
 * How about rows?
 */
BOOST_AUTO_TEST_CASE(RowSerializeXMLTest)
{
  arma::rowvec m;
  m.randu(1, 50);
  TestArmadilloSerialization<arma::rowvec, xml_iarchive, xml_oarchive>(m);
}

BOOST_AUTO_TEST_CASE(RowSerializeTextTest)
{
  arma::rowvec m;
  m.randu(1, 50);
  TestArmadilloSerialization<arma::rowvec, text_iarchive, text_oarchive>(m);
}

BOOST_AUTO_TEST_CASE(RowSerializeBinaryTest)
{
  arma::rowvec m;
  m.randu(1, 50);
  TestArmadilloSerialization<arma::rowvec, binary_iarchive, binary_oarchive>(m);
}

// A quick test with an empty matrix.
BOOST_AUTO_TEST_CASE(EmptyMatrixSerializeTest)
{
  arma::mat m;
  TestArmadilloSerialization<arma::mat, xml_iarchive, xml_oarchive>(m);
}

BOOST_AUTO_TEST_SUITE_END();
