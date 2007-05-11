// This version of memory manager supports memory allocation for
// objects that are smaller than a page block. This is ok since we
// are dealing with
#ifndef MEMORY_MANAGER_H_
#define MEMORY_MANAGER_H_

#include <string>
#include <iostream>
#include <sys/mman.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/signal.h>
#include <sys/unistd.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include "fastlib/fastlib.h"
#include "sigsegv.h"
#include "memory_manager.h"
#include "page_file_header.h"
#include "app_config.h"
#include "u/nvasil/tpie/ami.h"

using namespace std;

template<bool Logmode, int32 page_size>
class MemoryManager;

template<bool logmode>
struct Logger; 
 
template<bool Logmode, int32 kTPIEPageSize=4096>
class MemoryManager {
 public:
	static const int32 kVersion=1; 
  static  MemoryManager<Logmode> *allocator_;
  friend class MemoryManagerTest;
  typedef char Page_t[kTPIEPageSize];
	
  template<typename T>
  struct Tchar {
   	T t;
  	char c;
  };
  template<typename T> 
  static index_t StrideOf() {	
    return (sizeof(Tchar<T>) > sizeof(T)) ? 
           sizeof(Tchar<T>)-sizeof(T) : sizeof(T);
  }
  template<typename T, bool logmode=Logmode>
  class Ptr {
   public: 
    Ptr() {
      address_=-1;
    }
    Ptr(index_t address) {
   	  address_=address;	
    }
		Ptr(const Ptr<T, logmode> &other) {
		  this->address_ =  other.address_;
		}
    ~Ptr() {}
    void Reset(index_t address) {
      address_=address;
    }	
		void SetNULL() {
			address_=-1;
		}
    bool IsNULL(){
		  return address_==-1;
		}
    Ptr<T, logmode> &operator=(const Ptr<T, logmode> &other) {
   	  address_ = other.address_;
   	  return *this;
    }
		bool operator==(const Ptr<T, logmode> &other) const {
			return this->address_ == other.address_;
		}
    T &operator*() {
      Logger<logmode>::Log(address_);
      return reinterpret_cast<T>(allocator_->Access(address_));
    }
    T *operator->() {
      Logger<logmode>::Log(address_);
			return reinterpret_cast<T>(allocator_->Access(address_)); 
    }
		Ptr<Ptr<T, logmode>, logmode> Reference() {
		  Ptr<Ptr<T, logmode>, logmode>  ptr;
			ptr.Reset(this);
			return ptr;
		}
    T &operator[](index_t ind) {
      Logger<logmode>::Log(address_);
      return reinterpret_cast<T>(allocator_->Access(address_))[ind];
		}
		T *get() {
		  return reinterpret_cast<T>(allocator_->Access(address_));
		}
    
   protected:
    index_t address_;
  };
	template<typename T, bool logmode=Logmode>
	class ArrayPtr : public Ptr<T, logmode> {
	 public:
		ArrayPtr(){
		}
		ArrayPtr(index_t size) {
		  Reset(malloc(size));
		}
		template<typename ARRAYTYPE>
		void Copy(ARRAYTYPE other, index_t length) {
			T *p1=reinterpret_cast<T>(allocator_->Access(address_));
			T *p2=other.get();
		  for(index_t i=0; i<length; i++) {
			  p1[i] = p2[i];
			}
		}
	};

 public:
  static const int32 kMaxNumOfPages=1048576;
  // The Constructor sets some default parameter such as the cache size
  // and the page size, but it doesn't allocate the memory nor it
  // creates a file on the disk
  // this Ctor uses some default parameters
  // The only parameters that can be set, are the size of the cache
  // and the size of the page
  MemoryManager(){
    cache_size_ = 256 * 1024;
    system_page_size_  = sysconf(_SC_PAGESIZE);
    page_size_ = kTPIEPageSize;
		cache_file_="temp.mem";
		header_file_=cache_file_ + string(".header");
  }

  // Trivial Destructor
  ~MemoryManager() {
	}
  // Initialize allocates memory in the RAM and it creates a new
  // file on the drive. It creates an entirelly new memory manager
  void Initialize();
      
  // Load, uses an allready saved file to make a memory manager
  // It actually opens the file for append and it loads all the data
  // (or better the portion of data that fits) in the memory
  void Load();
  
	// Close() deallocates all the memory used up to now
  // Saves all modified pages to disk
  // Closes the file
  void Close();
    
  // Resets the page timestamps
  void ResetPageTimers(){
    page_timer_ = 0;
    for(index_t i=0; i<num_of_pages_; i++) {
  	  page_timestamp_[i]=0;
    }
  }

  // Returns a pointer of the requested size in the RAM, If RAM is
  // full it does the appropriate arrangments
  template <typename T>
  index_t Alloc(index_t size);
  // Align the memory with the stride of the object that has to be
  // allocated in the memory
	static index_t malloc(index_t size) {
	  return allocator_->Alloc<double>(std::max(size/sizeof(double), 
					                           sizeof(double)));
	}
  inline index_t Align(char *ptr, index_t stride);
  // Given the object address in Cache it returns the ObjectAddress
  inline index_t GetLastObjectAddress(char *ptr);
  // Returns the object address of a pointer that is in the cache
  // limits. If it is not it fails. Very useful when we need to make
  // a pointer on a struct member
  inline index_t GetObjectAddress(char *pointer);

  // Access object
  inline char *Access(index_t oaddress);
  char *get_cache() {
    return cache_;
  }
  
	index_t get_page_size() {
    return page_size_;
  }
  
	uint64 get_total_num_of_page_faults() {
  	return total_num_of_page_faults_;
  }
	
	index_t get_num_of_pages();

	void set_cache_size(index_t cache_size) {
	  cache_size_=cache_size;
	}

	void set_page_size(int32 page_size) {
	  page_size_= page_size;
	}
  
	void set_cache_file(string cache_file) {
		cache_file_=cache_file;
	}
	
	void set_header_file(string header_file) {
	  header_file_=header_file_;
	}
 private:
	// cache buffer
  char *cache_;
	// size actually allocated, we allocate a little bit more
  index_t alloc_size_;
	// capacity of the cache
  index_t cache_size_;
  // the system page size 4K for Linux, 64K for windows
	long system_page_size_;
	// total number of pages the cache has
  index_t num_of_pages_;
	// page size for the cache, for optimized performance use
	// multiples of the system page size
  index_t page_size_;
	
	// contains information for the cache
  string header_file_;
	// swap file for the cache
	string cache_file_;
  //Structure that keeps information about the header
	PageFileHeader  page_file_header_;
  // points to the next potential object location
  char  *current_ptr_;
	// the current offset on the page
  index_t current_offset_;
	// the current page, possibly for the next object
  index_t current_page_;
	// this is an array that keeps track the pages that 
	// have been modified
  bool *page_modified_;
	// keeps a pointer to a page on the cache
	// so the cache can accommodate up to kMaxNumOfPages
	// but only cache_size_/page_size_ can be in the main memory
	// so if the pages are not in the main memory they point to null
  char **page_address_;
  // This is an array that maps a page of the cache to the absolute
	// page
	index_t *cache_to_page_;
	// total number of pagefaults during execution
  uint64 total_num_of_page_faults_;
	// keeps the age of the pages in cache
  index_t *page_timestamp_;
  // the timer that increases after every access
  index_t page_timer_;
	// resets the page timer when it reaches the maximum age
  uint32 maximum_page_age_;
  // this is the interface to the hard disk
	AMI_STREAM<Page_t> *disk_;	
  
	// After setting files, page_size and other parameters
	// call this function to do some initializations that are common
	// in Initialize and Load
	void DefaultInitializations();
  bool IsPageModified(index_t page);
  void ClearPageStatus(index_t page);
  index_t PageToCachePage(index_t paddress);
  index_t CachePageToPage(index_t paddress); 
 
  void MapNewAddress(index_t paddress, index_t raddress);
  void UnMapAddress(index_t paddress);
  bool FitsInPage(index_t size, index_t stride);
  void NextPage();
  void MoveToDisk(index_t page);
  void MoveToCache(index_t paddres, index_t ram_page);
  index_t LeastNeededPage();
  void ProtectSysPagesAffected(index_t page, int permission);
  void SetPageModified(index_t page);
  void HandlePageFault(index_t page_requested);
  pair<index_t, index_t> *PagesAffectedBySEGV(long system_page);
  bool get_page_modified(index_t page);
};

template<>
MemoryManager<false>  *MemoryManager<false>::allocator_ = 0;
template<>
MemoryManager<true>  *MemoryManager<true>::allocator_ = 0;

template<bool Logmode, int32 page_size>
int FaultHandler(void *fault_address, int serious) {
  // attempt to write on a page, we have to record that so that
  // we know we have to write the page back when the page has to moved
  // to the disk
	MemoryManager<Logmode,page_size>  allocator=
      MemoryManager<Logmode,page_size>::allocator_;
  if (fault_address>= allocator->cache_ &&
      fault_address < allocator->cache_ + allocator->cache_size_){
    index_t cache_page = (ptrdiff_t)((char*)fault_address - allocator->cache_) 
       / allocator->page_size_;
   // page has to be set as modified 
   allocator->SetPageModified(cache_page);
   index_t system_page = (ptrdiff_t)fault_address  / allocator->system_page_size_;
   pair<index_t, index_t> * p=allocator->PagesAffectedBySEGV(system_page);
   // if all pages that are covered by the system page are modified
   // then the whole page should be set uprotected;
   for(index_t i=p->first; i<=p->second; i++) {
     if (likely(!allocator->IsPageModified(
						 static_cast<index_t>(cache_page + i)))) {
      return 1 ;
     }
   } 
   char *addr = allocator->cache_ + allocator->system_page_size_ *
       ((ptrdiff_t)((char*)fault_address - allocator->cache_)
        / allocator->system_page_size_);
   if (unlikely(mprotect(addr, 
					 allocator->system_page_size_, PROT_READ | PROT_WRITE) !=0)) {
     FATAL("Error %s while trying to change the protection\n", 
            strerror(errno));
   } 
   return 1;
 } 
     
  // This is probably a segmentation violation
  // because of a bug 
	const char *temp = "Faulting address %p\n"
      "Cache limits %p to %p\n"
      "Segmentation violation, There is a bug somewhere\n";
	FATAL(temp, fault_address, allocator->cache_, 
			  allocator->cache_+allocator->cache_size_);
}
#include "memory_manager_impl.h"
#endif /*MEMORY_MANAGER_H_*/
