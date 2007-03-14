
#include "textfile.h"

#include "xrun/xrun.h"

#include <cmath>

void Test1() {
  TextTokenizer scanner;
  const char *input = xrun_param_str("input");
  
  scanner.Open(input, "#", "", TextTokenizer::WANT_NEWLINE);
  
  while (scanner.PeekType() != TextTokenizer::END) {
    fprintf(stderr, "Got: %d, [%s]\n", scanner.PeekType(), scanner.Peek().c_str());
    scanner.Gobble();
  }
}

#define REQUIRE(scanner, cond) \
    if (!(scanner.cond)) { fprintf(stderr, "FAILED: %s: '%s'\n", #cond, \
        scanner.Peek().c_str()); } else
    

void Test2() {
  const char *fname = "tmpfile.txt";
  TextWriter writer;
  
  writer.Open(fname);
  writer.Printf("@begin(1, 1.0, 1.0e-31, abc-123, \"123\", '123') # comment here\r");
  writer.Printf("@end(2.0e-21)\r\nabc");
  writer.Close();
  
  TextTokenizer scanner;
  scanner.Open(fname, "#", "-", TextTokenizer::WANT_NEWLINE);
  
  REQUIRE(scanner, MatchPunct());
  REQUIRE(scanner, MatchIdentifier());
  REQUIRE(scanner, MatchPunct());
  REQUIRE(scanner, MatchInteger());
  REQUIRE(scanner, Match(","));
  REQUIRE(scanner, MatchDouble());
  REQUIRE(scanner, Match(","));
  REQUIRE(scanner, MatchDouble());
  REQUIRE(scanner, Match(","));
  REQUIRE(scanner, MatchType(TextTokenizer::IDENTIFIER));
  REQUIRE(scanner, Match(","));
  REQUIRE(scanner, MatchString());
  REQUIRE(scanner, Match(","));
  REQUIRE(scanner, MatchString());
  REQUIRE(scanner, MatchPunct());
  REQUIRE(scanner, Match("\n"));
  REQUIRE(scanner, Match("@"));
  REQUIRE(scanner, Match("end"));
  REQUIRE(scanner, Match("("));
  REQUIRE(scanner, Match("2.0e-21"));
  fprintf(stderr, "scanner.Current() == [%s]\n", scanner.Current().c_str());
  assert(scanner.Current() == "2.0e-21");
  assert(fabs(strtod(scanner.Current().c_str(), NULL) - 2.0e-21) < 1.0e-30);
  REQUIRE(scanner, Match(")"));
  REQUIRE(scanner, Match("\n"));
  REQUIRE(scanner, Match("abc"));
}

int main(int argc, char *argv[]) {
  xrun_init(argc, argv);
  
  Test1();
  Test2();
  
  xrun_done();
}
