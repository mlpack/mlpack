#include "arraylist.h"
#include "heap.h"
#include "fastalloc.h"
#include "intmap.h"

#include "base/test.h"

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

TEST_SUITE_END(col, TestArrayListInt, TestMinHeap,
    TestFastAlloc, TestIntMap)

