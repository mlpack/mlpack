/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#include "textfile.h"
//#include "textfile.h"
#include "base/test.h"

#include <math.h>

TEST_SUITE_BEGIN(textfile)

/*void Test1() {
  TextTokenizer scanner;
  const char *input = xrun_param_str("input");
  
  scanner.Open(input, "#", "", TextTokenizer::WANT_NEWLINE);
  
  while (scanner.PeekType() != TextTokenizer::END) {
    fprintf(stderr, "Got: %d, [%s]\n", scanner.PeekType(), scanner.Peek().c_str());
    scanner.Gobble();
  }
}*/

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
  //assert(scanner.Current() == "2.0e-21");
  TEST_ASSERT(scanner.Current() == "2.0e-21");
  //assert(fabs(strtod(scanner.Current().c_str(), NULL) - 2.0e-21) < 1.0e-30);
  TEST_DOUBLE_APPROX(strtod(scanner.Current().c_str(), NULL), 2.0e-21, 1.0e-30);
  REQUIRE(scanner, Match(")"));
  REQUIRE(scanner, Match("\n"));
  REQUIRE(scanner, Match("abc"));
}

TEST_SUITE_END(textfile, Test2)

