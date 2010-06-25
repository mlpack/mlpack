/**
 * @file union_find_test.cc
 *
 * @author Bill March (march@gatech.edu)
 *
 * Unit tests for the Union-Find data structure.
 */

#include "union_find.h"
#include <fastlib/base/test.h>
#include <fastlib/fastlib.h>

class TestUnionFind {
  
private:
  
  UnionFind* test_union_find_;
  static const index_t test_size_ = 10;
  
public: 
  
  void Init() {
    
    test_union_find_ = new UnionFind();
    test_union_find_->Init(test_size_);
  
  }
  
  void Destruct() {
   
    delete test_union_find_;
    
  }
  
  void TestFind() {
    
    Init();
    
    for (index_t i = 0; i < test_size_; i++) {
     
      TEST_ASSERT(test_union_find_->Find(i) == i);
      
    }
    test_union_find_->Union(0,1);
    test_union_find_->Union(1, 2);
    TEST_ASSERT(test_union_find_->Find(2) == test_union_find_->Find(0));
    
    Destruct();
    
    NONFATAL("TestFind passed.\n");
    
  }
  
  void TestUnion() {
  
    Init();
    
    test_union_find_->Union(0, 1);
    
    test_union_find_->Union(2, 3);
    test_union_find_->Union(0, 2);
    
    test_union_find_->Union(5, 0);
    test_union_find_->Union(0, 6);
    
    
    TEST_ASSERT(test_union_find_->Find(0) == test_union_find_->Find(1));
    TEST_ASSERT(test_union_find_->Find(2) == test_union_find_->Find(3));
    TEST_ASSERT(test_union_find_->Find(1) == test_union_find_->Find(5));
    TEST_ASSERT(test_union_find_->Find(6) == test_union_find_->Find(3));
    
    Destruct();
    
    NONFATAL("TestUnion passed.\n");
    
  }
  
  
  void RunTests() {
   
    TestFind();
    
    TestUnion();
    
    NONFATAL("UnionFind unit tests passed.\n");
    
  }
  
  
  
  
}; // class TestUnionFind

int main(int argc, char* argv[]) {
 
  TestUnionFind test;
  test.RunTests();
  
}
