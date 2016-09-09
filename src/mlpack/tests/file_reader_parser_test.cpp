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

BOOST_AUTO_TEST_SUITE(randomForest);

template<typename T>
void TestFloatParser()
{
  T num = 0;
  io::Parse<io::ThrowOnOverFlow>("10", num);
  BOOST_REQUIRE_CLOSE(num, 10, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("-10", num);
  BOOST_REQUIRE_CLOSE(num, -10, 1e-5);

  io::Parse<io::ThrowOnOverFlow>("10.01", num);
  BOOST_REQUIRE_CLOSE(num, 10.01, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("-10.01", num);
  BOOST_REQUIRE_CLOSE(num, -10.01, 1e-5);

  io::Parse<io::ThrowOnOverFlow>("10.05e3", num);
  BOOST_REQUIRE_CLOSE(num, 10.05e3, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("-10.05e3", num);
  BOOST_REQUIRE_CLOSE(num, -10.05e3, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("10.05e-3", num);
  BOOST_REQUIRE_CLOSE(num, 10.05e-3, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("-10.05e-3", num);
  BOOST_REQUIRE_CLOSE(num, -10.05e-3, 1e-5);

  io::Parse<io::ThrowOnOverFlow>("10.05E3", num);
  BOOST_REQUIRE_CLOSE(num, 10.05e3, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("-10.05E3", num);
  BOOST_REQUIRE_CLOSE(num, -10.05e3, 1e-5);
  io::Parse<io::ThrowOnOverFlow>("10.05E-3", num);
  BOOST_REQUIRE_CLOSE(num, 10.05e-3, 1E-5);
  io::Parse<io::ThrowOnOverFlow>("-10.05E-3", num);
  BOOST_REQUIRE_CLOSE(num, -10.05e-3, 1e-5);
}

BOOST_AUTO_TEST_CASE(ParseUnsignedIntTest)
{
  unsigned char num_char = 0;
  io::Parse<io::ThrowOnOverFlow>("200", num_char);
  BOOST_REQUIRE_EQUAL(num_char, 200);

  unsigned short num_short = 0;
  io::Parse<io::ThrowOnOverFlow>("300", num_short);
  BOOST_REQUIRE_EQUAL(num_short, 300);

  unsigned int num_int = 0;
  io::Parse<io::ThrowOnOverFlow>("400", num_int);
  BOOST_REQUIRE_EQUAL(num_int, 400);

  unsigned long num_ulong = 0;
  io::Parse<io::ThrowOnOverFlow>("500", num_ulong);
  BOOST_REQUIRE_EQUAL(num_ulong, 500);

  unsigned long long num_ullong = 0;
  io::Parse<io::ThrowOnOverFlow>("600", num_ullong);
  BOOST_REQUIRE_EQUAL(num_ullong, 600);

  size_t num_size_t = 0;
  io::Parse<io::ThrowOnOverFlow>("700", num_size_t);
  BOOST_REQUIRE_EQUAL(num_size_t, 700);
}

BOOST_AUTO_TEST_CASE(ParseSignedIntTest)
{
  signed char num_char = 0;
  io::Parse<io::ThrowOnOverFlow>("10", num_char);
  BOOST_REQUIRE_EQUAL(num_char, 10);

  short num_short = 0;
  io::Parse<io::ThrowOnOverFlow>("300", num_short);
  BOOST_REQUIRE_EQUAL(num_short, 300);

  int num_int = 0;
  io::Parse<io::ThrowOnOverFlow>("400", num_int);
  BOOST_REQUIRE_EQUAL(num_int, 400);

  long num_ulong = 0;
  io::Parse<io::ThrowOnOverFlow>("500", num_ulong);
  BOOST_REQUIRE_EQUAL(num_ulong, 500);

  long long num_ullong = 0;
  io::Parse<io::ThrowOnOverFlow>("600", num_ullong);
  BOOST_REQUIRE_EQUAL(num_ullong, 600);
}

BOOST_AUTO_TEST_CASE(ParseFloatTest)
{
  TestFloatParser<float>();
  TestFloatParser<double>();
  TestFloatParser<long double>();
}

BOOST_AUTO_TEST_CASE(ParseStringTest)
{
  std::string str;

  io::Parse<io::ThrowOnOverFlow>("600", str);
  BOOST_REQUIRE_EQUAL("600", str);

  io::Parse<io::ThrowOnOverFlow>("600 88976", str);
  BOOST_REQUIRE_EQUAL("600 88976", str);

  io::Parse<io::ThrowOnOverFlow>("hh 600 mm gg", str);
  BOOST_REQUIRE_EQUAL("hh 600 mm gg", str);
}

BOOST_AUTO_TEST_CASE(ParseLineTest)
{
  std::vector<char*> chars(3, nullptr);
  std::vector<int> colOrder(chars.size());
  std::iota(std::begin(colOrder), std::end(colOrder), 0);

  char strs[] = "600  ,800  ,300  ";
  io::ParseLine<io::TrimChars<' ', '\t'>, io::NoQuoteEscape<','>>(strs,
                                                                  &chars[0], colOrder);
  BOOST_REQUIRE_EQUAL("600", chars[0]);
  BOOST_REQUIRE_EQUAL("800", chars[1]);
  BOOST_REQUIRE_EQUAL("300", chars[2]);

  char strs2[] = "600\t800\t 300 ";
  io::ParseLine<io::TrimChars<' '>, io::NoQuoteEscape<'\t'>>(strs2,
                                                             &chars[0], colOrder);
  BOOST_REQUIRE_EQUAL("600", chars[0]);
  BOOST_REQUIRE_EQUAL("800", chars[1]);
  BOOST_REQUIRE_EQUAL("300", chars[2]);

  char strs3[] = "600\t800,300";
  io::ParseLine<io::TrimChars<' '>, io::NoQuoteEscapes<'\t',','>>(strs3,
                                                                  &chars[0], colOrder);
  BOOST_REQUIRE_EQUAL("600", chars[0]);
  BOOST_REQUIRE_EQUAL("800", chars[1]);
  BOOST_REQUIRE_EQUAL("300", chars[2]);
}

BOOST_AUTO_TEST_SUITE_END();
