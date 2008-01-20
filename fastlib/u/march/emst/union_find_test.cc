/**
 * @file union_find_test.cc
 *
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 */

#include "union_find.h"


int main(int argc, char* argv[]) {
 
  
  const index_t test_size = 10;
  
  UnionFind test;
  test.Init(test_size);
  
  for (index_t i = 0; i < test_size; i++) {
    DEBUG_ASSERT_MSG(test.Find(i) == i, "Test 1 failed, i = %d\n", i);
  }
  test.Union(0, 1);
  DEBUG_ASSERT_MSG(test.Find(0) == test.Find(1), 
                   "Test 2 failed, test.Find(0) = %d, test.Find(1) = %d\n",
                   test.Find(0), test.Find(1));
  
  test.Union(0, 1);
  
  test.Union(2, 3);
  test.Union(0, 2);
  
  test.Union(5, 0);
  test.Union(0, 6);
  
  DEBUG_ASSERT_MSG(test.Find(0) == test.Find(1),
                   "Test 3 failed, test.Find(0) = %d, test.Find(1) = %d\n",
                   test.Find(0), test.Find(1));
  DEBUG_ASSERT_MSG(test.Find(2) == test.Find(3),
                   "Test 4 failed, test.Find(2) = %d, test.Find(3) = %d\n",
                   test.Find(2), test.Find(3));
  DEBUG_ASSERT_MSG(test.Find(1) == test.Find(5),
                   "Test 3 failed, test.Find(1) = %d, test.Find(5) = %d\n",
                   test.Find(1), test.Find(5));
  
  
  printf("UnionFind tests passed\n");
  
}