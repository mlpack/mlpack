#include "arraylist.h"
#include "heap.h"
#include "fastalloc.h"
#include "intmap.h"
#include "rangeset.h"
#include "queue.h"

#include "fastlib/base/test.h"

TEST_SUITE_BEGIN(col)

// right now, just a plug to make sure some instantiation of ArrayList
// compiles.
// TODO(garryb): this should really be a runnable unit test
void TestArrayListInt() {
  ArrayList<int> a1;
  ArrayList<int> a2;
  
  a1.Init();
  a2.Init(21);
  
  a1.Resize(30);
  a1.Resize(12);
  a1.Resize(22);
  a2.Resize(22);
  a1[21] = 99;
  a1[20] = 88;
  TEST_ASSERT(a1[21] == 99);
  a1.PopBack();
  TEST_ASSERT(a1.size() == 21);
  TEST_ASSERT(a1[20] == 88);
  TEST_ASSERT(*a1.PopBackPtr() == 88);
  a1.AddBack()[0] = 3;
  TEST_ASSERT(a1[20] == 3);
  a1.Trim();
  TEST_ASSERT(a1.size() == 21);
  TEST_ASSERT(a1.capacity() == a1.size());
  
  for (index_t i = 0; i < a1.size(); i++) {
    a1[i] = i;
    a1[i]++;
  }
  
  for (index_t i = 0; i < a1.size(); i++) {
    TEST_ASSERT(a1[i] == i + 1);
  }
  
  ArrayList<int> a3;
  a3.Copy(a1);
  
  TEST_ASSERT(a1.size() == a3.size());
  for (index_t i = 0; i < a1.size(); i++) {
    TEST_ASSERT(a3[i] == i + 1);
  }
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
    TEST_ASSERT(v > last); // no duplicates
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
    TEST_ASSERT(v > last); // no duplicates
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
  DEBUG_SAME_SIZE(set.size(), 1);
  DEBUG_SAME_SIZE(set.ranges()[0].begin, 1);
  DEBUG_SAME_SIZE(set.ranges()[0].end, 16);
  
  set.Reset();
  DEBUG_SAME_SIZE(set.size(), 0);
  set.Union(3, 7);
  set.Union(50, 100);
  set.Union(4, 8);
  DEBUG_SAME_SIZE(set.size(), 2);
  DEBUG_SAME_SIZE(set.ranges()[0].begin, 3);
  DEBUG_SAME_SIZE(set.ranges()[0].end, 8);
  DEBUG_SAME_SIZE(set.ranges()[1].begin, 50);
  DEBUG_SAME_SIZE(set.ranges()[1].end, 100);
  
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
  DEBUG_SAME_SIZE(set.size(), 1);
  DEBUG_SAME_SIZE(set.ranges()[0].begin, 1);
  DEBUG_SAME_SIZE(set.ranges()[0].end, 79);
  set.Union(-55, -52);
  set.Union(500, 600);
  set.Union(800, 900);
  set.Union(550, 801);
  set.Union(-400, -200);
  set.Union(6, 81);
  DEBUG_SAME_SIZE(set.ranges().size(), 4);
  DEBUG_SAME_SIZE(set[0].begin, -400);
  DEBUG_SAME_SIZE(set[0].end, -200);
  DEBUG_SAME_SIZE(set[1].begin, -55);
  DEBUG_SAME_SIZE(set[1].end, -52);
  DEBUG_SAME_SIZE(set[2].begin, 1);
  DEBUG_SAME_SIZE(set[2].end, 81);
  DEBUG_SAME_SIZE(set[3].begin, 500);
  DEBUG_SAME_SIZE(set[3].end, 900);
  set.Reset();
  set.Union(3, 5);
  set.Union(3, 5);
  set.Union(3, 5);
  DEBUG_SAME_SIZE(set.size(), 1);
}

void TestQueue() {
  Queue<int> q;
  q.Init();
  *q.Add() = 3;
  q.Add(1);
  q.Add(4);
  q.Add(1);
  q.Add(5);
  *q.Add() = 9;
  DEBUG_ASSERT(!q.is_empty());
  DEBUG_SAME_SIZE(q.top(), 3); q.Pop();
  DEBUG_SAME_SIZE(q.top(), 1); q.Pop();
  DEBUG_SAME_SIZE(q.top(), 4); q.PopOnly();
  DEBUG_SAME_SIZE(q.Pop(), 1);
  DEBUG_SAME_SIZE(q.Pop(), 5);
  DEBUG_ASSERT(!q.is_empty());
  DEBUG_SAME_SIZE(q.top(), 9); q.Pop();
  DEBUG_ASSERT(q.is_empty());
  *q.Add() = 551;
  DEBUG_ASSERT(!q.is_empty());
  DEBUG_SAME_SIZE(q.top(), 551); q.PopOnly();
  DEBUG_ASSERT(q.is_empty());
}

TEST_SUITE_END(col, TestArrayListInt, TestMinHeap,
    TestFastAlloc, TestIntMap, TestRangeSet, TestQueue)

