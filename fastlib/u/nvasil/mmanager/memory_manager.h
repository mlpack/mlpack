#ifndef U_NVASIL_MMANAGER_MEMORY_MANAGER_H_
#define U_NVASIL_MMANAGER_MEMORY_MANAGER_H_

#include <assert.h>
#include <sys/unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <string>
#include <vector>
#include "fastlib/fastlib.h"

using namespace std;

namespace mmapmm {

template<bool Logmode, int32 page_size>
class MemoryManager;

template<bool logmode>
struct Logger; 
 
template<bool Logmode, int32 page_size=65536>
class MemoryManager {
 public:
  static  MemoryManager<Logmode> *allocator_;
  friend class MemoryManagerTest;
  static const int kTypicalSystemPageSize_ = 65536;
  static const uint32 kMinimumCapacity_=33554432;
	static const void *NullValue;
  template<typename T>
  struct Tchar {
   	T t;
  	char c;
  };
  template<typename T> 
  static size_t StrideOf() {	
    return (sizeof(Tchar<T>) > sizeof(T)) ? 
           sizeof(Tchar<T>)-sizeof(T) : sizeof(T);
  }
  template<typename T, bool logmode=Logmode>
  class Ptr {
   public: 
    Ptr() {
      p_=NULL;
    }
    Ptr(T *p) {
   	  p_= p;	
    }
		Ptr(const Ptr<T, logmode> &other) {
		  this->p_ =  other.p_;
		}
    ~Ptr() {}
	  void Reset(const void *p) {
		  p_=(T *)p;
		}	
		void SetNULL() {
			p_=NULL;
		}
    bool IsNULL(){
		  return p_==NULL;
		}
    Ptr<T, logmode> &operator=(const Ptr<T, logmode> &other) {
   	  p_ = other.p_;
   	  return *this;
    }
		bool operator==(const Ptr<T, logmode> &other) const {
			return this->p_ == other.p_;
		}
    T &operator*() {
      Logger<logmode>::Log(p_);
      //allocator->CachePage(p_);
		  return *p_;
    }
    T *operator->() {
      Logger<logmode>::Log(p_);
      //allocator->CachePage(p_);
			return p_;
    }
		Ptr<Ptr<T, logmode>, logmode> Reference() {
		  Ptr<Ptr<T, logmode>, logmode>  ptr;
			ptr.Reset(this);
			return ptr;
		}
    T &operator[](size_t ind) {
      Logger<logmode>::Log(p_);
      //allocator->CachePage(p_);
      return p_[ind];
    }
		T *get() {
		  return p_;
		}
    void Swizzle(ptrdiff_t offset) {
      p_ = (T *)((char*)p_ + offset);
    }
   protected:
    T *p_;
  };
	template<typename T, bool logmode=Logmode>
	class ArrayPtr : public Ptr<T, logmode> {
	 public:
		ArrayPtr(){
		}
		ArrayPtr(size_t size) {
		  Reset(malloc<T>(size));
		}
		template<typename ARRAYTYPE>
		void Copy(ARRAYTYPE other, size_t length) {
		  for(size_t i=0; i<length; i++) {
			  this->operator[](i) = other[i];
			}
		}
	};
 private: 
  char *pool_;
  uint64 pool_size_;
  string pool_name_;
  string page_access_filename_;
  FILE *fp_log_;
  int32 system_page_size_;
  uint64 current_position_;
  uint64 capacity_;
  uint64 realloc_chunk_;
  bool log_flag_;
	ptrdiff_t last_page_logged_;
	struct timeval last_time_a_new_page_accessed_;
	uint64  wasted_time_;
	//void CachePage(void *p) {
	//  struct PageChunk{
	//		char dummy[page_size];
	//	};
	// 	ptrdiff_t page_num=((char *)p-pool_)/page_size;
	//  ((PageChunk *)pool_)[page_num];
	//}
  uint64 frequency_of_logged_page_;
 public:
  MemoryManager() {
    capacity_ = kMinimumCapacity_;
    pool_name_ = "temp_mem";
    page_access_filename_ = "log_access.txt";
    system_page_size_ = getpagesize();
    realloc_chunk_ = kMinimumCapacity_;
		pool_= NULL;
		fp_log_ = NULL;
		last_page_logged_ = 0;
    log_flag_ = false;	
		frequency_of_logged_page_ = 0;
  }
  MemoryManager(string pool_name, uint64 capacity, string page_access_filename) {
    system_page_size_ = getpagesize();
    capacity_ = capacity;
		if (unlikely(capacity % system_page_size_ != 0)) {
			FATAL("\n Error!, the capacity "L64" is not a multiple of the "
            "page size "L32" \n", capacity_, system_page_size_);
    }
    page_access_filename_ = page_access_filename;
		pool_=NULL;
		if (Logmode==true) {
		  set_log_file(page_access_filename_);
			if (fp_log_ == NULL) {
			  FATAL("Could not open %s, error %s encountered\n", 
						page_access_filename_.c_str(), strerror(errno));
			}
		}
  }
	void Destruct() {
	  if (unlikely(munmap(pool_, capacity_)<0)) {
	    FATAL("Failed to unmap memory error: %s\n", strerror(errno));
		  if (Logmode==true && fp_log_!=NULL) {
  	    if (unlikely(fclose(fp_log_)!=0)) {
			    FATAL("Error closing %s\n", page_access_filename_.c_str());
			  }
	    }	
	  }
  }
  ~MemoryManager() {
		
  }

  void Init() {
		// do not use a file just use virtual memory
    if (pool_name_.empty()) {
       pool_ = (char*)mmap(NULL, capacity_, PROT_READ | PROT_WRITE, 
                   MAP_ANONYMOUS | MAP_SHARED, -1, 0);
      if (pool_==MAP_FAILED) {
        FATAL("Memory mapping error, %s\n", strerror(errno));
      }
    } else {
      struct stat info;
      int fd;
      if (stat(pool_name_.c_str(), &info) == 0) {
				NONFATAL("Warning file %s already exists with size %llu\n",
						      pool_name_.c_str(),
					        (unsigned long long)info.st_size);
				if ((uint64)info.st_size < capacity_) {
				 const char *temp="There is a filename for memory manager "
					                "but the size is smaller than the requested "
													"capacity "L64"<"L64"";
				 FATAL(temp,
							 info.st_size,
							 capacity_);
				}
    	  fd = open(pool_name_.c_str(), O_RDWR | O_CREAT);
  	    if (fd < 0) {
  	      FATAL("Error opening file %s, error type %s\n", 
  	             pool_name_.c_str(), strerror(errno));  	  		        
  	    }
			} else {
		    FILE *fp = fopen(pool_name_.c_str(), "w");
  	    char *buff= new char[kMinimumCapacity_];
  	    memset(buff, kMinimumCapacity_, 0);
  	    for(uint64 i=0; i < capacity_ / kMinimumCapacity_+1; i++) {
          if (unlikely(fwrite(buff, 1, kMinimumCapacity_, fp)!=
								kMinimumCapacity_)) {
            FATAL("Error %s while trying to write on file %s\n",
                   strerror(errno), pool_name_.c_str());
          } 
        }
			  delete buff;
			  fclose(fp);
			  fd=open(pool_name_.c_str(), O_RDWR);
      }
      pool_ = (char*)mmap(NULL, capacity_, PROT_READ | PROT_WRITE,  MAP_SHARED, fd, 0);
      if (pool_ == MAP_FAILED) {
        FATAL("Error %s while memmory mapping\n", strerror(errno));
      }
      if (close(fd) <0) {
        FATAL("Error closing file %s, error: %s\n", 
               pool_name_.c_str(), 
               strerror(errno));
      }
		}
    current_position_ = 0;  
   if (log_flag_==true)
     if ((fp_log_=fopen(page_access_filename_.c_str(), "w")) == NULL) {
       FATAL("Error: %s, while trying to open log file %s\n",
              strerror(errno), page_access_filename_.c_str());
     }	          
   }                
  void Reallocate() {
    if (!pool_name_.empty()) {
      int fd = open(pool_name_.c_str(), O_APPEND);
      if (fd < 0) {
         FATAL("Error opening file %s, error type %s\n", 
  	  	        pool_name_.c_str(),
  	  	        strerror(errno));
      }
      char buff[system_page_size_];
      memset(buff, system_page_size_, 0);
      for(uint32 i=0; i < realloc_chunk_ % system_page_size_; i++) {
        write(fd, buff,system_page_size_);
      }
      if (close(fd)<0) {
        FATAL("Error while trying to close file %s\n", 
               pool_name_.c_str());
      }
    } 
    pool_ = (char*)mremap(pool_, capacity_, capacity_+realloc_chunk_, 
				                  !MREMAP_MAYMOVE);
    capacity_+=realloc_chunk_;
    if (pool_==MAP_FAILED) {
      FATAL("You are trying to increase the memory size but "
            "the operating system cannot increase the address space "
            " in a contiguous way, error %s\n",
              strerror(errno));
    }
  } 
  template<typename T> 
  T *Alloc() {
    current_position_ += StrideOf<T>() - current_position_ % StrideOf<T>();
    if (current_position_ >capacity_) {
      Reallocate();
    }
    T *return_ptr = (T *)(pool_+current_position_);
  	current_position_ +=sizeof(T);
    if (current_position_ >capacity_) {
      Reallocate();
    } 
    return return_ptr;
  }

  template<typename T>
  T *Alloc(size_t size) {
    current_position_ += StrideOf<T>() - current_position_ % StrideOf<T>();
    if (current_position_ >capacity_) {
      Reallocate();
    }
    T *return_ptr = (T *)(pool_+current_position_);
    current_position_ +=sizeof(T) * size;
    if (unlikely(current_position_ >capacity_)) {
      Reallocate();
    } 
    return return_ptr;
  }
  void *AllignedAlloc(size_t size) 	{
		current_position_ += StrideOf<double>() - current_position_ % StrideOf<double>();
    if (unlikely(current_position_ >capacity_)) {
      Reallocate();
    }
    void *return_ptr = (void *)(pool_+current_position_);
  	current_position_ +=size;
    if (unlikely(current_position_ >capacity_)) {
      Reallocate();
    } 
    return return_ptr;
  }
	template<typename T>
  static T* malloc() {
	  return allocator_->Alloc<T>;
	}
	template<typename T>
	static T* malloc(size_t size) {
	  return allocator_->Alloc<T>(size);
	}
	static void* malloc(size_t size) {
	  return allocator_->AllignedAlloc(size);
	}
	template<typename T>
	static T* calloc(size_t size, const T init_value) {
	  T* ptr = malloc<T>(size);
		for(size_t i=0; i< size; i++) {
		  ptr[i]=init_value;
		}
    return ptr;
	}
  template<typename T>
  void Log(T *ptr) {
		struct timeval t1;
		gettimeofday(&t1, NULL);
    if (log_flag_ == true) {
			ptrdiff_t page = (ptrdiff_t)((char*)ptr-pool_) / system_page_size_;
			if (page == last_page_logged_) {
			  frequency_of_logged_page_++;
				struct timeval t2;
				gettimeofday(&t2, NULL);
				wasted_time_+=t2.tv_usec-t1.tv_usec;
			} else {
			  struct timeval t2;
			  gettimeofday(&t2, NULL);
        unsigned char flag=0;
				if (mincore(pool_+system_page_size_*page, 
							      system_page_size_, &flag)!=0) {
				  NONFATAL("Warning mincore failed %s\n", strerror(errno));
				}

				fprintf(fp_log_, "%li %lu %lu %lu ", last_page_logged_, 
						                          frequency_of_logged_page_,
					                            t2.tv_usec-
																			last_time_a_new_page_accessed_.tv_usec, 
																			wasted_time_);
				if (flag<<7!=128) {
				  fprintf(fp_log_,"0 0\n");
				} else {
				  struct timeval t1;
					gettimeofday(&t1, NULL);
					madvise(pool_+system_page_size_*page, 1, MADV_WILLNEED);
					struct timeval t2;
					gettimeofday(&t2, NULL);
          *(pool_+system_page_size_*page)+=0;
					struct timeval t3;
          gettimeofday(&t3, NULL);
	        if (mincore(pool_+system_page_size_*page, 
								      system_page_size_, &flag)!=0) {
				    NONFATAL("Warning mincore failed %s\n", strerror(errno));
				  }
	      	if (flag<<7!=128) {
		        FATAL("Error page wasn't fetched\n");
					}
	        fprintf(fp_log_,"%lu %lu  ",
						      t2.tv_usec-t1.tv_usec, // time to do an advise
						      t3.tv_usec-t2.tv_usec   // time to fetch the page
						     );
				}
          fprintf(fp_log_,"\n");
			  last_page_logged_ = page; 
			  frequency_of_logged_page_ = 1;
				wasted_time_=0;
				gettimeofday(&last_time_a_new_page_accessed_, NULL);
      }
		}	
  }
  void Advise(vector<uint64> &pages_needed, vector<uint64> &pages_not_needed) {
    for(uint32 i=0; i< pages_not_needed.size(); i++) {
      if (unlikely(madvise(pool_+pages_not_needed[i] * system_page_size_, system_page_size_,
                  MADV_DONTNEED)<0)) {
        NONFATAL("Warning: Encountered %s error while advising\n", 
               strerror(errno));
      }   	          	
    }
    for(uint32 i=0; i< pages_needed.size(); i++) {
      if (unlikely(madvise(pool_+pages_not_needed[i] * system_page_size_, system_page_size_,
                  MADV_WILLNEED)<0)) {
        NONFATAL("Warning: Encountered %s error while advising\n",
                strerror(errno));
      } 	          	
    }
  }
  void Advise(int advice) {
    if (unlikely(madvise(pool_, capacity_, advice)<0)) {
      NONFATAL("Warning: Encountered %s error while advising\n", 
             strerror(errno));
    }	
  }
  
  float32 VerifyAdvise(vector<uint64> &pages_needed, 
                       vector<uint64> pages_not_needed) {
    uint32 num_of_pages = (capacity_ + system_page_size_ - 1)/system_page_size_;
    unsigned char vec[num_of_pages];
    if (mincore(pool_, capacity_, vec) <0) {
      NONFATAL("Warning: Encountered %s error while executing mincore\n",
              strerror(errno));
    }
    uint32 correct_pages=0;
    for(uint32 i=0; i < pages_needed.size(); i++) {
      if ((vec[pages_needed[i]] >> 1) ==1) {
        correct_pages++;
      }  
    }
    for(uint32 i=0; i < pages_not_needed.size(); i++) {
      if ((vec[pages_needed[i]] >> 1) ==0) {
        correct_pages++;
      }  
    }
    return (1.0 * correct_pages)/num_of_pages;
  }
  char *get_pool() {
    return pool_;
  }  
  
  uint64 get_capacity() {
    return capacity_;
  }

	void set_pool_name(string pool_name) {
	  pool_name_ = pool_name;
	}

	void set_capacity(uint64 capacity) {
	  if (pool_ != NULL) {
		  FATAL("Too late to cahnge capacity, memory manager is already "
					            "initialized\n");
		}
		if (capacity % system_page_size_ != 0) {
      FATAL("\n Error!, the capacity "L64"  is not a multiple of the "
            "page size "L32" \n", (unsigned long long) capacity_
                                       , system_page_size_);
    }
   capacity_ = capacity;
	}
	ptrdiff_t get_usage() {
	 return current_position_;
	}
  void set_log(bool mode) {
    log_flag_=mode;
  }
  
  void set_log_file(string file) {
    if (Logmode==false) {
		  return;
		}
		if (fp_log_!=NULL && fclose(fp_log_)!=0) {
	    FATAL("Could not close %s, error %s encountered\n", 
					  page_access_filename_.c_str(), strerror(errno));
	  }
		page_access_filename_ = file;
		fp_log_ = fopen(page_access_filename_.c_str(), "w");
		if (fp_log_ == NULL) {
		  FATAL("Could not opene %s, error %s encountered\n", 
					page_access_filename_.c_str(), strerror(errno));
		}

	}
 	  	
};

template<bool Logmode, int32 page_size>
const void* MemoryManager<Logmode, page_size>::NullValue=NULL; 

template<bool logmode>
struct Logger {
  template<typename T>
  static void Log(T *p);
};
template<>
struct Logger<true> {
  template<typename T>
  static void Log(T *p) {
		MemoryManager<true>::allocator_->Log(p);
  }
};

template<>
struct Logger<false> {
  template<typename T>
  static void Log(T *p) {
  }
};
 
template<>
MemoryManager<false>  *MemoryManager<false>::allocator_ = 0;
template<>
MemoryManager<true>  *MemoryManager<true>::allocator_ = 0;

};

#endif /*MEMORY_MANAGER_H_*/
