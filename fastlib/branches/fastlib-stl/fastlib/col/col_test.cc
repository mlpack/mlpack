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
#include "heap.h"
#include "fastalloc.h"
#include "intmap.h"
#include "rangeset.h"
#include "tokenizer.h"

#include <assert.h>
#include <string>

#include <iostream>

#include "../base/test.h"

TEST_SUITE_BEGIN(col)

void TestTokenizer() {
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

// right now, just a plug to make sure some instantiation of ArrayList
// compiles.
// TODO(garryb): this should really be a runnable unit test
void TestMinHeap() {
  MinHeap<double, int> h;
  int a[] = {31,41,59,26,53,58,97,93,23,84,62,64,33,83,27,92};
  int n = sizeof(a)/sizeof(a[0]);
  
  h.Init();
  
  
  for (int i = 0; i < n; i++) {
    h.Put(a[i], a[i]);
  }

  MinHeap<double, int> h2(h);
  
  int last = -1;
  
  DEBUG_ASSERT_MSG(h.top() == 23, "%d", h.top());
  h.set_top(2);
  DEBUG_ASSERT_MSG(h.top() == 2, "%d", h.top());
  
  for (int i = 0; i < n; i++) {
    int v = h.Pop();
    assert(v > last); // no duplicates
    last = v;
  }
  
  DEBUG_ASSERT_MSG(last == 97, "%d", h.top());

  h = h2;
  
  last = -1;
  
  DEBUG_ASSERT_MSG(h.top() == 23, "%d", h.top());
  h.set_top(2);
  DEBUG_ASSERT_MSG(h.top() == 2, "%d", h.top());
  
  for (int i = 0; i < n; i++) {
    int v = h.Pop();
    assert(v > last); // no duplicates
    last = v;
  }
  
  DEBUG_ASSERT_MSG(last == 97, "%d", h.top());
}

struct MyStruct {
  int a;
  int b;
  MyStruct *next;
  
  MyStruct() {}
  MyStruct(int a_in, int b_in, MyStruct *next_in) {
    a = a_in;
    b = b_in;
    next = next_in;
  }
};

void TestFastAlloc() {
  MyStruct *a = fast_new(MyStruct)(3, 4, NULL);
  MyStruct *b = fast_new(MyStruct)(1, 2, a);
  
  fast_delete(a);
  fast_delete(b);
}

void TestIntMap() {
  DenseIntMap<double> map;
  map.Init();
  map.default_value() = 0.0;
  map[31] = 31;
  map[41] = 41;
  map[59] = 59;
  map[26] = 26;
  TEST_DOUBLE_EXACT(map[0], 0);
  TEST_DOUBLE_EXACT(map[499], 0);
  TEST_DOUBLE_EXACT(map[31], 31);
  TEST_DOUBLE_EXACT(map[41], 41);
  TEST_DOUBLE_EXACT(map[59], 59);
}

void TestRangeSet() {
  RangeSet<int> set;
  
  set.Init();
  set.Union(1, 4);
  set.Union(10, 16);
  set.Union(4, 10);
  assert(set.size()==1);
  assert(set.ranges()[0].begin == 1);
  assert(set.ranges()[0].end == 16);
  
  set.Reset();
  assert(set.size() == 0);
  set.Union(3, 7);
  set.Union(50, 100);
  set.Union(4, 8);
  assert(set.size() == 2);
  assert(set.ranges()[0].begin == 3);
  assert(set.ranges()[0].end == 8);
  assert(set.ranges()[1].begin == 50);
  assert(set.ranges()[1].end == 100);
  
  set.Reset();
  set.Union(1, 5);
  set.Union(59, 79);
  set.Union(6, 10);
  set.Union(9, 18);
  set.Union(48, 60);
  set.Union(19, 44);
  set.Union(44, 46);
  set.Union(18, 19);
  set.Union(42, 48);
  set.Union(5, 6);
  assert(set.size() == 1);
  assert(set.ranges()[0].begin == 1);
  assert(set.ranges()[0].end == 79);
  set.Union(-55, -52);
  set.Union(500, 600);
  set.Union(800, 900);
  set.Union(550, 801);
  set.Union(-400, -200);
  set.Union(6, 81);
  assert(set.ranges().size() == 4);
  assert(set[0].begin == -400);
  assert(set[0].end == -200);
  assert(set[1].begin == -55);
  assert(set[1].end == -52);
  assert(set[2].begin == 1);
  assert(set[2].end == 81);
  assert(set[3].begin == 500);
  assert(set[3].end == 900);
  set.Reset();
  set.Union(3, 5);
  set.Union(3, 5);
  set.Union(3, 5);
  assert(set.size() == 1);
}

TEST_SUITE_END(col, TestMinHeap, TestTokenizer,
    TestFastAlloc, TestIntMap, TestRangeSet)

