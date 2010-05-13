/*
 * =====================================================================================
 *
 *       Filename:  memory_manager_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/07/2008 05:51:29 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include "memory_manager.h"
#include "fastlib/math/math_lib.h"

class MemoryManagerTest {
 public:
  void Init(){
    mmapmm::MemoryManager<false>::allocator_ = new mmapmm::MemoryManager<false>();
    mmapmm::MemoryManager<false>::allocator_->set_capacity(65536*1000);
    mmapmm::MemoryManager<false>::allocator_->Init();
  }
 
  void Destruct() {
    mmapmm::MemoryManager<false>::allocator_->Destruct();
    delete mmapmm::MemoryManager<false>::allocator_;
  }
 
  void Test1() {
    NOTIFY("Just testing memory allocation\n");
    Init();
    double *ptr=mmapmm::MemoryManager<false>::allocator_->malloc<double>(100000);
    for(index_t i=0; i<100000; i++) {
      ptr[i]=math::Random(0, 1);
    }
    Destruct();
    NOTIFY("Memory successfully allocated assigned and accessed...!!\n");
  }
 
  void TestAll() {
    Test1();
  }
};

int main(int argc, char* argv[]) {
  MemoryManagerTest test;
  test.TestAll();
}
