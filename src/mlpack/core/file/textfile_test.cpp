#include "textfile.hpp"
#include <math.h>

#define BOOST_TEST_MODULE TextFileTest
#include <boost/test/unit_test.hpp>

/*void Test1() {
  TextTokenizer scanner;
  const char *input = xrun_param_str("input");
  
  scanner.Open(input, "#", "", TextTokenizer::WANT_NEWLINE);
  
  while (scanner.PeekType() != TextTokenizer::END) {
    fprintf(stderr, "Got: %d, [%s]\n", scanner.PeekType(), scanner.Peek().c_str());
    scanner.Gobble();
  }
}*/

BOOST_AUTO_TEST_CASE(Test2) {
  using namespace mlpack;
  const char *fname = "tmpfile.txt";
  TextWriter writer;
  
  writer.Open(fname);
  writer.Printf("@begin(1, 1.0, 1.0e-31, abc-123, \"123\", '123') # comment here\r");
  writer.Printf("@end(2.0e-21)\r\nabc");
  writer.Close();
  
  TextTokenizer scanner;
  scanner.Open(fname, "#", "-", TextTokenizer::WANT_NEWLINE);

  BOOST_REQUIRE(scanner.MatchPunct());
  BOOST_REQUIRE(scanner.MatchIdentifier());
  BOOST_REQUIRE(scanner.MatchPunct());
  BOOST_REQUIRE(scanner.MatchInteger());
  BOOST_REQUIRE(scanner.Match(","));
  BOOST_REQUIRE(scanner.MatchDouble());
  BOOST_REQUIRE(scanner.Match(","));
  BOOST_REQUIRE(scanner.MatchDouble());
  BOOST_REQUIRE(scanner.Match(","));
  BOOST_REQUIRE(scanner.MatchType(TextTokenizer::IDENTIFIER));
  BOOST_REQUIRE(scanner.Match(","));
  BOOST_REQUIRE(scanner.MatchString());
  BOOST_REQUIRE(scanner.Match(","));
  BOOST_REQUIRE(scanner.MatchString());
  BOOST_REQUIRE(scanner.MatchPunct());
  BOOST_REQUIRE(scanner.Match("\n"));
  BOOST_REQUIRE(scanner.Match("@"));
  BOOST_REQUIRE(scanner.Match("end"));
  BOOST_REQUIRE(scanner.Match("("));
  BOOST_REQUIRE(scanner.Match("2.0e-21"));
  //assert(scanner.Current() == "2.0e-21");
  BOOST_REQUIRE(scanner.Current() == "2.0e-21");
  //assert(fabs(strtod(scanner.Current().c_str(), NULL) - 2.0e-21) < 1.0e-30);
  BOOST_REQUIRE_CLOSE(strtod(scanner.Current().c_str(), NULL),2.0e-21, 1e-5);
  BOOST_REQUIRE(scanner.Match(")"));
  BOOST_REQUIRE(scanner.Match("\n"));
  BOOST_REQUIRE(scanner.Match("abc"));
}
