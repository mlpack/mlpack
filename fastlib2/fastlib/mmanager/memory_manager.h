#ifndef FASTLIB_MEMORY_MANAGER_MEMORY_MANAGER_H_
#define FASTLIB_MEMORY_MANAGER_MEMORY_MANAGER_H_

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
//#include "fastlib/fastlib.h"
#include "fastlib/base/common.h"
#include "fastlib/fx/fx.h"
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
  static const int TYPICAL_SYSTEM_PAGE_SIZE = 65536;
  static const uint32 MINIMUM_CAPACITY= 4194304;
	static const void *NullValue;
  /**
   * This is a trick to get the alignment of a struct.
   *  When we allocate memory it has to be aligned
   *  This is the right way to do it
   */
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
  /**
   * This is a smart pointer, behaves exacly like any 
   * other pointer except from the fact that it gets memory
   * from the memory manager
   */
  template<typename T, bool logmode=Logmode>
  class Ptr {
   public: 
    /**
     * Constructors. The default constructor 
     * just sets p_ to NULL.
     */
    Ptr() {
      p_=NULL;
    }
    /**
     * Use this to initialize it with a chunk of memory
     */
    Ptr(T *p) {
   	  p_= p;	
    }

    /**
     * Copy constructor
     */
		inline Ptr(const Ptr<T, logmode> &other) {
		  this->p_ =  other.p_;
		}
    /**
     * The destructor does nothing, since the memory will be massively
     * deallocated by the memory manager
     */
    ~Ptr() {
    }
    /**
     * Use this if you just want to reset the value of the pointer
     */
	  inline void Reset(const void *p) {
		  p_=(T *)p;
		}	
    /**
     * Sets the pointer to NULL
     */
		inline void SetNULL() {
			p_=NULL;
		}
    /**
     * Checks to see if the pointer is NULL
     */
    inline bool IsNULL(){
		  return p_==NULL;
		}
    /**
     * Assignement operator, It is equivalent to Reset
     */
    inline Ptr<T, logmode> &operator=(const Ptr<T, logmode> &other) {
   	  p_ = other.p_;
   	  return *this;
    }
    /**
     * Equality opearator. Checks if the pointers point to the same memory location
     */
		inline bool operator==(const Ptr<T, logmode> &other) const {
			return this->p_ == other.p_;
		}
    /**
     * Access Operator
     */
    inline T &operator*() {
      Logger<logmode>::Log(p_);
      //allocator->CachePage(p_);
		  return *p_;
    }
    inline T *operator->() {
      Logger<logmode>::Log(p_);
      //allocator->CachePage(p_);
			return p_;
    }
		/**
     * Returns a pointer to the pointer
     */
    inline Ptr<Ptr<T, logmode>, logmode> Reference() {
		  Ptr<Ptr<T, logmode>, logmode>  ptr;
			ptr.Reset(this);
			return ptr;
		}
    /**
     * Bracket Operator if you want to use it as an array
     */
    T &operator[](size_t ind) {
      Logger<logmode>::Log(p_);
      //allocator->CachePage(p_);
      return p_[ind];
    }
    /**
     * Gets the actual pointer
     */ 
		inline T *get() {
		  return p_;
		}
    /**
     * I don't remember why I did this
     */
		inline T *get_p() {
		  return p_;
		}
    /**
     * The memory manager allocates addresses to the smart pointer, but after we save 
     * and reload the file all the smart pointers have invalid addresses. The process
     * of making the addresses valid is called Swizzling. So all the addresses are relative to
     * the anchor address of the memory manager. 
     */
    void Swizzle(ptrdiff_t offset) {
      p_ = (T *)((char*)p_ + offset);
    }
    /**
     * These are not used anymore
     */
		inline void Lock() {
		/**
     * This one is obsolete too
     */
		}
		inline void Unlock() {
		
		}

   protected:
    T *p_;
  };

  /**
   * ArrayPtr is useful if you need Array Operations
   */
	template<typename T, bool logmode=Logmode>
	class ArrayPtr : public Ptr<T, logmode> {
	 public:
		ArrayPtr() {
		}
    /**
     * Construct an array of given size
     */
		inline ArrayPtr(size_t size) {
		  Reset(malloc<T>(size));
		}
    /**
     * Copy elements form any other structure that has the []operator
     */
		template<typename ARRAYTYPE>
		inline void Copy(ARRAYTYPE other, size_t length) {
		  for(size_t i=0; i<length; i++) {
			  this->operator[](i) = other[i];
			}
		}
	};
  
 private: 
  // points to the allocated address from the operating system
  char *pool_;
  // the allocated size
  uint64 pool_size_;
  // an identifier of the pool
  std::string pool_name_;
  // filename to save the pool
  std::string page_access_filename_;
  // pointer to the file
  FILE *fp_log_;
  // system page size 
  int32 system_page_size_;
  // current position in the pool. This is the
  // address for the next allocation
  uint64 current_position_;
  // Current capacity of the memory
  // Capacity should be less than the pool_size
  uint64 capacity_;
  // if we need to reallocate memory because  we have reached the
  // capacity we realloc 
  uint64 realloc_chunk_;
  // These were meant to be used for logging the accesses
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
  fx_module *module_;
  
 public:
  MemoryManager() {
    capacity_ = MINIMUM_CAPACITY;
    pool_name_ = "temp_mem";
    page_access_filename_ = "log_access.txt";
    system_page_size_ = getpagesize();
    realloc_chunk_ = MINIMUM_CAPACITY;
		pool_= NULL;
		fp_log_ = NULL;
		last_page_logged_ = 0;
    log_flag_ = false;	
		frequency_of_logged_page_ = 0;
    module_=NULL;
  }

  MemoryManager(std::string pool_name, uint64 capacity, std::string page_access_filename) {
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
#ifdef MAP_ANONYMOUS       
      pool_ = (char*)mmap(NULL, capacity_, PROT_READ | PROT_WRITE, 
                   MAP_ANONYMOUS | MAP_SHARED, -1, 0);
      if (pool_==MAP_FAILED) {
        FATAL("Memory mapping error, %s\n", strerror(errno));
      }    
#else
     FATAL("MAP_ANONYMOUS is not defined for the particular platoform, currently not"
           "supporting virtual memory allocation for this platform");    
#endif 
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
  	    char *buff= new char[MINIMUM_CAPACITY];
  	    memset(buff, MINIMUM_CAPACITY, 0);
  	    for(uint64 i=0; i < capacity_ / MINIMUM_CAPACITY+1; i++) {
          if (unlikely(fwrite(buff, 1, MINIMUM_CAPACITY, fp)!=
								MINIMUM_CAPACITY)) {
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
    if (log_flag_==true) {
      if ((fp_log_=fopen(page_access_filename_.c_str(), "w")) == NULL) {
        FATAL("Error: %s, while trying to open log file %s\n",
               strerror(errno), page_access_filename_.c_str());
      }	          
    }                
  }
  void Init(fx_module *module) {
    module_=module;
    Init();
  }
  /**
   * Reallocate will try to remap but keep pool_ in the same address
   * Usually this will fail. We cannot allow reallocation with change of
   * pool_ pointer, because all the allocated pointers will have invalid addresses
   */
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
  /**
   * Allocates memory for any object type:
   * ie  ClassA *a=Alloc<ClassA>()
   */
  template<typename T> 
  inline T *Alloc() {
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
  /**
   * Allocates a block of memory that can fit n objects of class T
   */
  template<typename T>
  inline T *Alloc(size_t size) {
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
  
  /**
   * This is sort of obsolete and it should be used only for low level 
   * operations. It just allocs n blocks of char
   */
  inline void *AllignedAlloc(size_t size) 	{
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

  /**
   * Prefer this one. It does exacly the same thing with Alloc. I put it here for all of you
   * who are familiar with classical malloc
   */
	template<typename T>
  static inline T* malloc() {
	  return allocator_->Alloc<T>();
	}
	/**
   *  Use this if you want to allocate memory for an array
   */
  template<typename T>
	static T* malloc(size_t size) {
	  return allocator_->Alloc<T>(size);
	}
	/**
   * Obsolete. Use with caution for low level operations
   */
  inline static void* malloc(size_t size) {
	  return allocator_->AllignedAlloc(size);
	}
  /**
   * Works exactly like the traditional calloc. The difference between 
   * malloc is that it initializes the memory
   */
  template<typename T>
	static inline  T* calloc(size_t size, const T init_value) {
	  T* ptr = malloc<T>(size);
		for(size_t i=0; i< size; i++) {
		  ptr[i]=init_value;
		}
    return ptr;
	}
  /**
   * This function logs the accesses to a file
   */
  template<typename T>
  inline void Log(T *ptr) {
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
  /**
   * This is an important function. It advises the kernel which pages to keep and 
   * which to discard from the cache. Use of Advise can speed up memory accesss
   */
  inline void Advise(std::vector<uint64> &pages_needed, std::vector<uint64> &pages_not_needed) {
    fx_timer_start(module_, "advise");
    for(uint32 i=0; i< pages_not_needed.size(); i++) {
      if (unlikely(madvise(pool_+pages_not_needed[i] * system_page_size_, system_page_size_,
                  MADV_DONTNEED)<0)) {
        NONFATAL("Warning: Encountered ...%s... error (%i) while advising\n", 
               strerror(errno), errno);
      }   	          	
    }
    for(uint32 i=0; i< pages_needed.size(); i++) {
      if (unlikely(madvise(pool_+pages_not_needed[i] * system_page_size_, system_page_size_,
                  MADV_WILLNEED)<0)) {
        NONFATAL("Warning: Encountered ...%s... error (%i) while advising\n",
                strerror(errno), errno);
      } 	          	
    }
    fx_timer_stop(module_, "advise");  
  }
  /**
   * Advises a sequence of pages
   * advice: MADV_NORMAL
   *         MADV_RANDOM
   *         MADV_SEQUENTIAL
   *         MADV_WILLNEED
   *         MADV_DONTNEED
   */ 
  inline void Advise(uint64 page, uint64 number_of_pages, int advice) {
    fx_timer_start(module_, "advise");  
    if (unlikely(madvise(pool_+page * system_page_size_, number_of_pages*system_page_size_,
                  advice)<0)) {
      NONFATAL("Warning: Encountered ...%s...  error (%i) while advising\n", 
                strerror(errno), errno);
    }     
    fx_timer_stop(module_, "advise");  
  }
  
 
 /**
   * Advises a sequence of pages
   * advice: MADV_NORMAL
   *         MADV_RANDOM
   *         MADV_SEQUENTIAL
   *         MADV_WILLNEED
   *         MADV_DONTNEED
   */ 
  inline void Advise(void *ptr, size_t length, int advice) {
    // locate the page the start address_begins
    fx_timer_start(module_, "advise");  
    index_t page = (ptrdiff_t)((char*)ptr-pool_)/system_page_size_;
    index_t num_of_pages = ((ptrdiff_t)((char*)ptr-pool_)%system_page_size_
        + length)/system_page_size_;
    if (unlikely(madvise(pool_+page * system_page_size_, num_of_pages*system_page_size_,
                  advice)<0)) {
      NONFATAL("Warning: Encountered ...%s... error (%i)  while advising\n", 
                strerror(errno), errno);
    }  
    fx_timer_stop(module_, "advise");  
  }
  inline void Advise(void *ptr1, void *ptr2, int advice) {
    // locate the page the start address_begins
    fx_timer_start(module_, "advise");  
    index_t page = (ptrdiff_t)((char*)ptr1-pool_)/system_page_size_;
    index_t num_of_pages = (ptrdiff_t)((char*)ptr1-(char*)ptr2)/system_page_size_;
    if (unlikely(madvise(pool_+page * system_page_size_, num_of_pages*system_page_size_,
                  advice)<0)) {
      NONFATAL("Warning: Encountered ...%s... error (%i) while advising\n", 
                strerror(errno), errno);
    }     
    fx_timer_stop(module_, "advise");    
  }
   
  /**
   * This one advises the whole pool
   */
  void Advise(int advice) {
    fx_timer_start(module_, "advise");  
    if (unlikely(madvise(pool_, capacity_, advice)<0)) {
      NONFATAL("Warning: Encountered ...%s... error (%i) while advising\n", 
             strerror(errno), errno);
    }	
    fx_timer_stop(module_, "advise");  
  }
  /**
   *  Verify to see if your system really took your advice into consideration
   */
  float32 VerifyAdvise(std::vector<uint64> &pages_needed, 
                       std::vector<uint64> pages_not_needed) {
    uint32 num_of_pages = (capacity_ + system_page_size_ - 1)/system_page_size_;
    unsigned char vec[num_of_pages];
    if (mincore(pool_, capacity_, vec) <0) {
      NONFATAL("Warning: Encountered ...%s... error (%i) while executing mincore\n",
              strerror(errno), errno);
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

	void set_pool_name(std::string pool_name) {
	  pool_name_ = pool_name;
	}

	void set_capacity(uint64 capacity) {
	  if (pool_ != NULL) {
			const char *temp="Too late to cahnge capacity, memory manager is already "
					       "initialized\n";
		  FATAL(temp);
		}
		if (capacity % system_page_size_ != 0) {
			const char *temp=
				"\n Error!, the capacity "L64"  is not a multiple of the "
        "page size "L32" \n";
      FATAL(temp, (unsigned long long) capacity_, system_page_size_);
    }
   capacity_ = capacity;
	}
  
	ptrdiff_t get_usage() {
	 return current_position_;
	}
  
  void set_log(bool mode) {
    log_flag_=mode;
  }
  
  void set_log_file(std::string file) {
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

};

#endif /*MEMORY_MANAGER_H_*/
