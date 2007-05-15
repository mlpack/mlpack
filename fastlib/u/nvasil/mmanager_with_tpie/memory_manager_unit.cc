/*
 * =====================================================================================
 *
 *       Filename:  memory_manager_unit.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/11/2007 11:57:10 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include "fastlib/fastlib.h"
#include "base/test.h"
#include "memory_manager.h"
using namespace tpiemm;
class MemoryManagerTest {
 public:
	static const index_t kChunkSize=210;
	typedef int32 Chunk_t[kChunkSize];
	typedef MemoryManager<false> Allocator_t;
	typedef Allocator_t::Ptr<Chunk_t> ChunkPtr_t;
	void Init() {
   	capacity_=1048576;
    pagesize_=16384;	
		num_of_chunks_=3330;
    MemoryManager<false>::allocator_ = new MemoryManager<false>();
    MemoryManager<false>::allocator_->set_cache_size(capacity_);
    MemoryManager<false>::allocator_->set_page_size(pagesize_);
    MemoryManager<false>::allocator_->Init();

  }
	
	void Destruct() {
    MemoryManager<false>::allocator_->Close();
	  delete MemoryManager<false>::allocator_; 
		unlink("temp.mem");
		unlink("temp.mem.header");
	}
	
	void LoadMemoryAndAccess() {
	  ChunkPtr_t *data= new ChunkPtr_t[num_of_chunks_];
    for(index_t i=0; i<num_of_chunks_; i++) {
      data[i].Reset(Allocator_t::malloc(sizeof(Chunk_t)));		
			for(index_t j=0; j < kChunkSize; j++) {
			  (*data[i])[j]=j*i;
        TEST_ASSERT((*data[i])[j]==j*i);
			}
		}
		
		for(index_t i=0; i<num_of_chunks_; i++) {
			for(index_t j=0; j < kChunkSize; j++) {
			  TEST_ASSERT((*data[i])[j]==j*i);
			}
		}
		delete []data;
	}
  
  void TestAll() {
	  Init();
		LoadMemoryAndAccess();
		Destruct();
		printf("All tests passed!\n");
	}	
	
 private:	 
  index_t capacity_;
	index_t pagesize_;
	index_t num_of_chunks_;
};

int main(int argc, char *argv[]) {
  MemoryManagerTest test;
  test.TestAll();

}
