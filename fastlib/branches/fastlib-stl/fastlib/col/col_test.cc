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
#include "tokenizer.h"

#include <assert.h>
#include <string>

#include <iostream>

#include "../base/test.h"

TEST_SUITE_BEGIN(col)

void TestTokenizer() {
  // This test could probably be trimmed down without losing value
  int i;
  std::string test[] = {"a","a,b","a,b;c",",;,;,,,;;a",
    "a,,,,;;;;,;,;,;,", ";,a,,b,,c;;d,;"};
  std::string del[] = {",",";",",;"};

  std::vector<std::string> tokens;

  // Can we do basic tokenizing?
  tokenizeString(test[0], del[0], tokens);
  assert(tokens.size() == 1);
  assert(tokens[0] == "a");

  tokens.clear();
  tokenizeString(test[1], del[0], tokens);
  assert(tokens.size() == 2);
  assert(tokens.front() == "a");
  assert(tokens.back() == "b");

  tokens.clear();
  tokenizeString(test[2], del[0], tokens);
  assert(tokens.size() == 2);
  assert(tokens.front() == "a");
  assert(tokens.back() == "b;c");

  tokens.clear();
  tokenizeString(test[3], del[0], tokens);
  assert(tokens.size() == 3);
  assert(tokens[0] == ";" && tokens[1] == ";");
  assert(tokens[2] == ";;a");
  
  tokens.clear();
  tokenizeString(test[4], del[0], tokens);
  assert(tokens.size() == 5);
  assert(tokens[0] == "a");
  assert(tokens[1] == ";;;;");
  for( i = 2; i < 5; ++i )
    assert( tokens[i] == ";" );

  tokens.clear();
  tokenizeString(test[5], del[0], tokens);
  assert(tokens.size() == 5);
  assert(tokens[0] == ";");
  assert(tokens[1] == "a");
  assert(tokens[2] == "b");
  assert(tokens[3] == "c;;d");
  assert(tokens[4] == ";");

  // And with a different delimeter?
  tokens.clear();
  tokenizeString(test[0], del[1], tokens);
  assert(tokens.size() == 1);
  assert(tokens[0] == "a");

  tokens.clear();
  tokenizeString(test[1], del[1], tokens);
  assert(tokens.size() == 1);
  assert(tokens.front() == "a,b");

  tokens.clear();
  tokenizeString(test[2], del[1], tokens);
  assert(tokens.size() == 2);
  assert(tokens.front() == "a,b");
  assert(tokens.back() == "c");

  tokens.clear();
  tokenizeString(test[3], del[1], tokens);
  assert(tokens.size() == 4);
  assert(tokens[0] == "," && tokens[1] == ",");
  assert(tokens[2] == ",,,");
  assert(tokens[3] == "a");
  
  tokens.clear();
  tokenizeString(test[4], del[1], tokens);
  assert(tokens.size() == 5);
  assert(tokens[0] == "a,,,,");
  for( i = 1; i < 5; ++i )
    assert( tokens[i] == "," );

  tokens.clear();
  tokenizeString(test[5], del[1], tokens);
  assert(tokens.size() == 2);
  assert(tokens[0] == ",a,,b,,c");
  assert(tokens[1] == "d,");

  // With multiple delimeters?
  tokens.clear();
  tokenizeString(test[0], del[2], tokens);
  assert(tokens.size() == 1);
  assert(tokens[0] == "a");

  tokens.clear();
  tokenizeString(test[1], del[2], tokens);
  assert(tokens.size() == 2);
  assert(tokens.front() == "a");
  assert(tokens.back() == "b");

  tokens.clear();
  tokenizeString(test[2], del[2], tokens);
  assert(tokens.size() == 3);
  assert(tokens[0] == "a");
  assert(tokens[1] == "b");
  assert(tokens[2] == "c");

  tokens.clear();
  tokenizeString(test[3], del[2], tokens);
  assert(tokens.size() == 1);
  assert(tokens.front() == "a");
  
  tokens.clear();
  tokenizeString(test[4], del[2], tokens);
  assert(tokens.size() == 1);
  assert(tokens.front() == "a");

  tokens.clear();
  tokenizeString(test[5], del[2], tokens);
  assert(tokens.size() == 4);
  assert(tokens[0] == "a");
  assert(tokens[1] == "b");
  assert(tokens[2] == "c");
  assert(tokens[3] == "d");

  // Test skipping ahead some number of characters
  tokens.clear();
  tokenizeString(test[3],del[0],tokens,4);
  assert(tokens.size() == 1);
  assert(tokens[0] == ";;a");

  // Test stopping on a specific character
  tokens.clear();
  tokenizeString(test[2],del[0],tokens,0,";");
  assert(tokens.size() == 2); 

  // Test stopping after some number of tokens found
  tokens.clear();
  tokenizeString(test[5], del[2], tokens, 0, "", 2);
  assert(tokens.size() == 2);
  assert(tokens[0] == "a");
  assert(tokens[1] == "b");

  // Test saving last token when requested
  tokens.clear();
  tokenizeString(test[4],del[2],tokens,0,";",0,true);
  assert(tokens.size() == 2);
  assert(tokens.front() == "a");
  assert(tokens.back() == ";;;;,;,;,;,");

  // Test empty strings
  tokens.clear();
  tokenizeString("",del[0], tokens);
  assert(tokens.size() == 0);

  tokens.clear();
  tokenizeString(test[5], "", tokens);
  assert(tokens.size() == 1 );

  // We don't need to test any arguments with defaults, as we know those work,
  // by tests that don't specify values above.
}

TEST_SUITE_END(col, TestTokenizer)

