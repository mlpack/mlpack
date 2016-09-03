/**
 * @file file_reader_parser_test.cpp
 * @author Ngap Wei Tham
 *
 * Test the parsers of fast csv
 */

#include <mlpack/core/data/file_reader/csv_reader.hpp>

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mlpack;
using namespace arma;

BOOST_AUTO_TEST_SUITE(CSVReaderTest);

BOOST_AUTO_TEST_CASE(ReadHomogeneousRowTest)
{
  std::fstream f;
  f.open("test.csv", std::fstream::out);
  f << "1, 2, hello  \n";
  f << "hello  , goodbye, coffe  \n";
  f.close();

  io::CSVReader<> reader(3, "test.csv");
  std::vector<std::string> elems(3);

  reader.ReadRow(elems);
  BOOST_REQUIRE_EQUAL(elems.size(), 3);
  BOOST_REQUIRE_EQUAL(elems[0], "1");
  BOOST_REQUIRE_EQUAL(elems[1], "2");
  BOOST_REQUIRE_EQUAL(elems[2], "hello");

  reader.ReadRow(elems);
  BOOST_REQUIRE_EQUAL(elems.size(), 3);
  BOOST_REQUIRE_EQUAL(elems[0], "hello");
  BOOST_REQUIRE_EQUAL(elems[1], "goodbye");
  BOOST_REQUIRE_EQUAL(elems[2], "coffe");
}

BOOST_AUTO_TEST_CASE(ReadHeterogeneousRowTest)
{
  std::fstream f;
  f.open("test.csv", std::fstream::out);
  f << "1, 2.0, hello  \n";
  f << "3  , 4.0, coffe  \n";
  f.close();

  io::CSVReader<> reader(3, "test.csv");
  int integer = 0;
  double dValue = 0;
  std::string str;
  reader.ReadRow(integer, dValue, str);
  BOOST_REQUIRE_EQUAL(integer, 1);
  BOOST_REQUIRE_CLOSE(dValue, 2.0, 1e-5);
  BOOST_REQUIRE_EQUAL(str, "hello");

  reader.ReadRow(integer, dValue, str);
  BOOST_REQUIRE_EQUAL(integer, 3);
  BOOST_REQUIRE_CLOSE(dValue, 4.0, 1e-5);
  BOOST_REQUIRE_EQUAL(str, "coffe");
}

BOOST_AUTO_TEST_SUITE_END();
