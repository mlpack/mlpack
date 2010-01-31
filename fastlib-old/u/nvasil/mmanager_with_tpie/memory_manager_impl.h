/*
 * =====================================================================================
 * 
 *       Filename:  memory_manager_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/10/2007 11:25:40 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#define TEMPLATE__ template<bool Logmode, int32 kTPIEPageSize>
#define MEMORY_MANAGER__  MemoryManager<Logmode, kTPIEPageSize>

namespace tpiemm {
TEMPLATE__
void MEMORY_MANAGER__::Init() {
  int fd;
  sigsegv_install_handler(&FaultHandler<Logmode, kTPIEPageSize>);
  DefaultInitializations();
	// Allocate one page of memory by mapping /dev/zero. Map the memory
  // as write-only, initially.
  /*fd = open ("/dev/zero", O_RDONLY);
  cache_ = (char *)mmap (NULL, alloc_size_, 
		                   PROT_WRITE, MAP_PRIVATE, fd, 0);*/
	cache_ = (char *)mmap(NULL, alloc_size_, PROT_READ | PROT_WRITE,
			                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (cache_==MAP_FAILED) {
	  FATAL("Couldn't allocate memory for the cache, error: %s\n",
				  strerror(errno));
	}
 /* if (mlock(cache_, alloc_size_) == -1) {
    NONFATAL("Could not lock memory manger's cache, error %s\n",
  	         strerror(errno));
		const char *temp="Memory manager proceeds with unlocked memory,"
                     "this may slow down performance\n"	;			
    NONFATAL(temp);
  }*/
   close(fd);
   page_modified_ = new bool[num_of_pages_];
   memset(page_modified_, false, num_of_pages_);

  // Init internal variables
  current_ptr_ = cache_;
  current_offset_ = 0;
  current_page_ = 0;
  // register all the pages
  page_address_ptr_  = new char*[kMaxNumOfPages];
	page_address_ = new int32[kMaxNumOfPages];
  for(int32 i=0; i < kMaxNumOfPages; i++) {
    page_address_ptr_[i] = NULL;
		page_address_[i] = -1;
  }	
  cache_to_page_ = new index_t[num_of_pages_];  
  for(int32 i=0; i < num_of_pages_; i++) {
    cache_to_page_[i] = -1;
  }	
  for(index_t i=0 ; i<num_of_pages_; i++){
    page_address_ptr_[i] = cache_ + i * page_size_;
		page_address_[i]=i;
    cache_to_page_[i] = i;
  }
  // Init the page timestamps
  page_timestamp_ = new index_t[num_of_pages_];
	// Init the page locks
	page_locks_ = new index_t[num_of_pages_];
	memset(page_locks_, 0, num_of_pages_ * sizeof(index_t));
  ResetPageTimers();
  // Set the first one modified
  SetPageModified(0);
  // fill cache_ with 0
  memset(cache_, 0, cache_size_);
  // note: We do not protect the memory for writes. We know that the pages will
  // be modified so we avoid unnecessary interupts
    

	// Block transfer engine Initializations
	MM_manager.ignore_memory_limit();
	AMI_err ae;
	struct stat info;
  if (stat(cache_file_.c_str(), &info) == 0) {
			NONFATAL("Warning file %s already exists, if it is not a valid TPIE file, "
				       "system will give a floating point exception",
						     cache_file_.c_str());
	}
	disk_= new AMI_STREAM<Page>(cache_file_.c_str());
	for(index_t i=0; i< num_of_pages_; i++) {
	  if ((ae=disk_->write_array((Page *)(cache_+i*page_size_),
		  	 page_size_/kTPIEPageSize))!=AMI_ERROR_NO_ERROR) {
		  cout << "AMI_ERROR " << ae << " during disk_.write_item()\n";
			exit(1);
		}
	}
	disk_->persist(PERSIST_PERSISTENT);
  // open the header file for write
  FILE *header_fp = fopen(header_file_.c_str(), "wb+");
  if (header_fp == NULL) {
    FATAL("Couldn't open %s for write\n", header_file_.c_str());
  }
  // The header of the file contains the following information
  // int          : version
  // unsigned long: total number of number of ram_pages
  // CacheSizeNum : disk page size  (this is the cache_size_)
  // index_t : ram page size
  page_file_header_.version_ = kVersion;
  page_file_header_.total_pages_ = 0;
  page_file_header_.cache_size_ = cache_size_;
  page_file_header_.page_size_ = page_size_;

  if (fwrite(&page_file_header_, sizeof(PageFileHeader), 1, header_fp) != 1) {
    FATAL("Write failure, %s\n", strerror(errno));
  }
	fclose(header_fp);
}
   
// Load, uses an allready saved file to make a memory manager
// It actually opens the file for append and it loads all the data
// (or better the portion of data that fits) in the memory
TEMPLATE__
void MEMORY_MANAGER__::Load() {
  int fd;
  sigsegv_install_handler(&FaultHandler<Logmode, kTPIEPageSize>);
  DefaultInitializations();
  // open the  header file for write
  FILE *header_fp = fopen(header_file_.c_str(), "ab+");
  if (header_fp == NULL) {
    FATAL("Couldn't open %s for write, %s\n", 
  			   header_file_.c_str(),
					 strerror(errno));
  }

  if (fread(&page_file_header_, sizeof(PageFileHeader), 1, header_fp ) !=1) {
    FATAL("Couldn't read the header of the file, %s\n",
		       strerror(errno));
  }
  if (page_file_header_.version_ != kVersion) {
	  const char *temp=
		    "Warning the file you are trying to read has been "
        "saved as version %i and you are running version %i\n";

    NONFATAL(temp, page_file_header_.version_, kVersion);
  }
  fclose(header_fp);
  alloc_size_ = page_file_header_.cache_size_;
  page_size_  = page_file_header_.page_size_;
  num_of_pages_ = alloc_size_ / page_size_;

  // Init internal variables
  current_offset_ = 0;
  current_page_ = page_file_header_.total_pages_;
  current_offset_ = page_file_header_.last_offset_ ;

  // Allocate one page of memory by mapping /dev/zero. Map the memory
  // as write-only, initially.
  fd = open ("/dev/zero", O_RDONLY);
  cache_ = (char *)mmap (NULL, alloc_size_, PROT_WRITE, MAP_PRIVATE, fd, 0);
  /*if (mlock(cache_, alloc_size_)==-1) {
    NONFATAL("Could not lock memory manager's cache, error %s\n",
             strerror(errno));
    NONFATAL("Memory manager proceeds with unlocked memory,"
             "this may slow down performance\n");
  }*/
  close (fd);
  page_modified_ = new bool[num_of_pages_];
  memset(page_modified_, false, num_of_pages_);
	// Init the page locks
	page_locks_ = new index_t[num_of_pages_];
	memset(page_locks_, 0, num_of_pages_ * sizeof(index_t));

  current_ptr_ = cache_;
  // fill cache_ with 0
  memset(cache_, 0, cache_size_);
  
  // register all the pages
  page_address_ptr_  = new char*[kMaxNumOfPages];
	page_address_ = new int32[kMaxNumOfPages];
  for(int32 i=0; i < kMaxNumOfPages; i++) {
     page_address_ptr_[i] = NULL;
		 page_address_[i]=-1;
  }	
  cache_to_page_ = new index_t[num_of_pages_];  
  for(index_t i=0; i < num_of_pages_; i++) {
    cache_to_page_[i] = -2;
  }	
  for(index_t i=0 ; i<num_of_pages_; i++){
    page_address_ptr_[i] = cache_ + i * page_size_;
		page_address_[i]=i;
    MoveToCache(i, i);
    cache_to_page_[i] = i;
  }

  // Reset the page timestamps
  page_timestamp_ = new uint32[num_of_pages_];
  ResetPageTimers();
  //protect all pages
  if (mprotect(cache_, cache_size_, PROT_READ ) !=0) {
    FATAL("Error %s while trying to change the protection\n", 
		  	  strerror(errno));
  }
}

TEMPLATE__
void MEMORY_MANAGER__::Destruct() {
  // scan all the pages and see if there is any modified page
  // and save it to disk
  for(index_t i=0; i<num_of_pages_; i++) {
    if (page_modified_[i]) {
      MoveToDisk(CachePageToPage(i));
    }
  }
  // delete the memory allocated for the page timestamps
  delete []page_timestamp_;
  // delete other help variables;
  delete []page_address_ptr_;
	delete []page_address_;
  delete []cache_to_page_;
  delete []page_locks_; 
  // delete cache
  /*if (unlikely(munlock(cache_, alloc_size_)==-1)) {
    NONFATAL("Could not unlock memory manger's cache, error %s\n",
  	         strerror(errno));
  	NONFATAL("Continue execution\n");
  }*/
  munmap(cache_, alloc_size_);
  delete []page_modified_;
  // Before closing the file we update the number of total ram pages
  page_file_header_.total_pages_ = current_page_;
  page_file_header_.last_offset_ = current_offset_;
    
	FILE *header_fp = fopen(header_file_.c_str(), "ab+");
  if (header_fp == NULL) {
    FATAL("Couldn't open %s for write, %s\n", 
				   header_file_.c_str(),
					 strerror(errno));
  }

  if (fwrite(&page_file_header_, sizeof(PageFileHeader), 1, header_fp) != 1) {
    FATAL("Unable, to write the total size, %s\n",
	  			strerror(errno));
  }

  if (fclose(header_fp) != 0) {
    FATAL("Unable to close the file, %s\n", strerror(errno));
  }
	disk_->persist(PERSIST_PERSISTENT);
	delete disk_;
}

// Align the memory with the stride of the object that has to be
// allocated in the memory
TEMPLATE__
inline index_t MEMORY_MANAGER__::Align(char *ptr, index_t stride) {
  // do proper alignment for the memory
  index_t stride_offset = 0;
  index_t rem = ((ptrdiff_t)ptr) % stride;
  if (rem != 0) {
    stride_offset = stride - rem;
  }
	return stride_offset;
}

TEMPLATE__
inline bool MEMORY_MANAGER__::FitsInPage(index_t size, index_t stride) {
  index_t stride_offset = Align(current_ptr_, stride);
	index_t offset=current_offset_ % page_size_;
  if (likely(offset + stride_offset + size < page_size_)) {
    return true;
  } else {
    return false;
  }
}

TEMPLATE__
inline void MEMORY_MANAGER__::NextPage() {
   current_page_++;
  // check if we are in the end of the cache_
  if (likely(current_page_ >= num_of_pages_)) {
		CreateNewPageOnDisk();
  	index_t page_to_move  = current_page_ % num_of_pages_; 
    if (likely(IsPageModified(page_to_move))){
      MoveToDisk(CachePageToPage(page_to_move));
    }
    // update the current pointer
    index_t new_page =  page_to_move;
    current_ptr_ = cache_ + new_page * page_size_;
		current_offset_=current_page_*page_size_;
		// clean the page
    memset(current_ptr_, 0, page_size_);
    //set the page as modified
    SetPageModified(new_page);
    // Invalidate the old mapping page from the index
    UnMapAddress(new_page);
    // Validate the new page with  the cache
    MapNewAddress(current_page_, new_page);
  } else {
  	current_ptr_ = cache_ + current_page_ * page_size_;
  	current_offset_=current_page_ * page_size_;
		SetPageModified(current_page_);
  }
}

TEMPLATE__
inline char *MEMORY_MANAGER__::Access(index_t address) {
  page_timer_++;
  // Reset the page timestamps  
  if (unlikely(page_timer_ >=  maximum_page_age_)) {
  	ResetPageTimers();
  	page_timer_++;
  }
  index_t page=address/page_size_;
	index_t offset=address%page_size_;
  if (unlikely(page_address_ptr_[page] == NULL)) {
  	HandlePageFault(page);
  	total_num_of_page_faults_++;
  }
  page_timestamp_[PageToCachePage(page)]=page_timer_;
  char *access_ptr_ = page_address_ptr_[page] + offset;
  
  return access_ptr_;
}

TEMPLATE__
inline void MEMORY_MANAGER__::Lock(index_t address) {
  Access(address);
	page_locks_[PageToCachePage(address/page_size_)]++;
}

TEMPLATE__
inline void MEMORY_MANAGER__::Unlock(index_t address) {
  page_locks_[PageToCachePage(address/page_size_)]--;
}

TEMPLATE__
inline char *MEMORY_MANAGER__::LockAndAccess(index_t address) {
  page_timer_++;
  // Reset the page timestamps  
  if (unlikely(page_timer_ >=  maximum_page_age_)) {
  	ResetPageTimers();
  	page_timer_++;
  }
  index_t page=address/page_size_;
	index_t offset=address%page_size_;
  if (unlikely(page_address_ptr_[page] == NULL)) {
  	HandlePageFault(page);
  	total_num_of_page_faults_++;
  }
  page_timestamp_[PageToCachePage(page)]=page_timer_;
  char *access_ptr_ = page_address_ptr_[page] + offset;
  Lock(page);  
  return access_ptr_;
}

// Moves to disk a page that has universal address paddress
TEMPLATE__
inline void MEMORY_MANAGER__::MoveToDisk(index_t paddress){
  if (page_address_ptr_[paddress] == NULL) {
		const char *temp="Attempt to move a page to disk that doesn't exist"
                     " or is not in cache\n";
    FATAL(temp);
  }
  off_t disk_offset = paddress * page_size_/kTPIEPageSize;
	AMI_err ae;
  
	if (unlikely((ae=disk_->seek(disk_offset))!=AMI_ERROR_NO_ERROR)) {
	  cout << "AMI_ERROR " << ae << "\n";
		if (paddress<0) {
		  FATAL("Null pointer exception!\n");
		}
    FATAL("Unable to seek to %llu \n", (unsigned long long)disk_offset);
	}	
 	if ((ae=disk_->write_array((Page *)(page_address_ptr_[paddress]), 
					                  page_size_/kTPIEPageSize))!=AMI_ERROR_NO_ERROR) {
	  cout << "AMI ERROR " << ae << " while transfering block to disk\n";
		FATAL(" An error occured whule trying to write on disk\n");
	}

}


// Moves  a page with universal address paddress,
// to cache with cache page adddress ram_page
TEMPLATE__
inline void MEMORY_MANAGER__::MoveToCache(index_t paddress, 
		                                      index_t ram_page) {
  off_t disk_offset = static_cast<off_t>(paddress * page_size_ 
			                                   / kTPIEPageSize);
	AMI_err ae;
  if (unlikely((ae=disk_->seek(disk_offset))!=AMI_ERROR_NO_ERROR)) {
	  cout << "AMI_ERROR " << ae << "\n";
		if (paddress<0) {
		  FATAL("Null pointer exception!\n");
		}
    FATAL("Unable to seek to %llu \n", (unsigned long long)disk_offset);
	}  
  char *ptr = cache_+ ram_page * page_size_;
	off_t num_of_pages_to_read =page_size_/kTPIEPageSize;
  if ((ae=disk_->read_array((Page *)ptr, 
					                  &num_of_pages_to_read))!=AMI_ERROR_NO_ERROR) {
	  cout << "AMI ERROR " << ae << " while transfering block to disk\n";
		FATAL(" An error occured while trying to read from disk\n");
	}

}

// Returns the object address of a pointer that is in the cache. If
// it is beyond the cache limits it fails.
TEMPLATE__ 
inline index_t MEMORY_MANAGER__::GetObjectAddress(void *pointer) {
  if (unlikely(pointer <cache_ || pointer >= cache_ + alloc_size_)) {
  	FATAL( "Pointer %p is out of cache_limits %p -- %p\n",
  	       pointer, cache_, cache_+cache_size_);
  }
  index_t oaddress;
	
  oaddress = CachePageToPage((ptrdiff_t)((char*)pointer - cache_) /page_size_)
            * page_size_ + (ptrdiff_t)((char*)pointer - cache_) % page_size_;
  return oaddress;
}

// Maps the universal page address paddress to cache address cache_ram_address
TEMPLATE__
inline void MEMORY_MANAGER__::MapNewAddress(index_t paddress, 
		                                     index_t cache_ram_address) {
  page_address_ptr_[paddress] = cache_ + cache_ram_address * page_size_;
	page_address_[paddress]=cache_ram_address;
  cache_to_page_[cache_ram_address] = paddress; 
}

// UnMaps the address of a page with cache address cache_address
TEMPLATE__
inline void MEMORY_MANAGER__::UnMapAddress(index_t cache_address) {
  index_t paddress = CachePageToPage(cache_address);
  page_address_ptr_[paddress] = NULL; 
  page_address_[paddress]=-1;
	cache_to_page_[cache_address] = -2; 
}

// Clears the modification status of a page with cache_page address
TEMPLATE__
inline void MEMORY_MANAGER__::ClearPageStatus(index_t cache_page) {
  page_modified_[cache_page] = false;
}

// Sets the modification status of a page with cache_page address
TEMPLATE__
inline void MEMORY_MANAGER__::SetPageModified(index_t cache_page) {
  page_modified_[cache_page] = true;
  // need to set the appropriate system page unprotected
   ProtectSysPagesAffected(cache_page, PROT_READ | PROT_WRITE);
}

// Unprotects all the system pages that include the requested cache_page
TEMPLATE__
inline void MEMORY_MANAGER__::ProtectSysPagesAffected(index_t cache_page, 
		                                                  int permission) {
  char *addr = cache_ + cache_page * page_size_;

  if (unlikely(mprotect(addr, page_size_, permission) !=0)) {
   FATAL("Error %s while trying to "
				 "change the protection\n"
				 "Page Address %p\n", strerror(errno), addr);
  }
	
}

TEMPLATE__
inline bool MEMORY_MANAGER__::IsPageModified(index_t cache_page){
  return page_modified_[cache_page];
}

TEMPLATE__
inline pair<index_t, index_t> *MEMORY_MANAGER__::PagesAffectedBySEGV(long system_page) {
  long system_page_start = system_page * system_page_size_;
  long system_page_end = system_page_start + system_page_size_ - 1;
  pair<index_t, index_t> *p;
  p = new pair<index_t, index_t>((system_page_start - (ptrdiff_t)cache_) /page_size_,
                       std::min((index_t)((system_page_end - (ptrdiff_t)cache_)/page_size_), num_of_pages_) );
  return p;
}

TEMPLATE__
inline index_t MEMORY_MANAGER__::LeastNeededPage() {
  index_t least_recently_used = -1;
  index_t least_recently_used_time = numeric_limits<index_t>::max(); 
  for(index_t i=0; i< num_of_pages_; i++) {
	  if (unlikely(page_locks_[i]>0)) {
		  continue;
		}
  	if (page_timestamp_[i] < least_recently_used_time) {
  	  least_recently_used = i;
  	  least_recently_used_time = page_timestamp_[i];
    }
  }
  if (unlikely(least_recently_used==-1)) {
		FATAL("All paged are locked  and cache is stalled. Try to unlock");
	}
  return least_recently_used ;
//  return rand() % num_of_pages_;
}

TEMPLATE__
inline void MEMORY_MANAGER__::HandlePageFault(index_t page_requested) {
  index_t least_needed_page = LeastNeededPage();
  index_t paddress_least_needed_page = CachePageToPage(least_needed_page);
  if (IsPageModified(least_needed_page)) {
    MoveToDisk(paddress_least_needed_page);
  }
  UnMapAddress(least_needed_page);
  // mark this page as unmodified
  ClearPageStatus(least_needed_page);
  MoveToCache(page_requested, least_needed_page);
  ProtectSysPagesAffected(least_needed_page, PROT_READ);
  MapNewAddress(page_requested, least_needed_page);
  // Age the page that you are trying to access 
  page_timestamp_[least_needed_page] = page_timer_;  
} 	

TEMPLATE__
inline index_t MEMORY_MANAGER__::PageToCachePage(index_t paddress) {

	DEBUG_ASSERT_MSG(page_address_ptr_[paddress] != NULL,
			             "You are trying to access a page that "
			             "is not in Cache\n Page number %i\n",
			              paddress);
   return page_address_[paddress];
	// return  ((ptrdiff_t)(page_address_ptr_[paddress] - cache_)) / page_size_;
}

TEMPLATE__
inline index_t MEMORY_MANAGER__::CachePageToPage(index_t cache_page) {
  return cache_to_page_[cache_page];
}

TEMPLATE__
inline bool MEMORY_MANAGER__::get_page_modified(index_t cache_page) {
  return page_modified_[cache_page];
}



TEMPLATE__
inline index_t MEMORY_MANAGER__::GetLastObjectAddress(char *ptr) {
  index_t oaddress;
  oaddress = current_page_ * page_size_ +
             (ptrdiff_t)(ptr - page_address_ptr_[current_page_]);
  return oaddress;
}

TEMPLATE__
index_t MEMORY_MANAGER__::AlignedAlloc(index_t size) {
  size_t stride;
	stride = StrideOf<double>();	
  // check if it fits in the current page
  if (unlikely(!FitsInPage(size, stride))) {
  	NextPage();
  } 
  if (unlikely(!FitsInPage(size, stride))) {
		FATAL("The object cannot fit in one block (page) "
  	      "of memory, increase page_size and try again...\n");
  }
  index_t stride_offset = Align(current_ptr_, stride); 
  char *new_ptr = current_ptr_ + stride_offset;
	index_t new_offset=current_offset_+ stride_offset;
  current_ptr_ = new_ptr + size;
  current_offset_ = current_offset_ + stride_offset + size;
  return new_offset;

}

TEMPLATE__
template <typename T>
index_t MEMORY_MANAGER__::Alloc(index_t size=1) {
  size_t stride;
	stride = StrideOf<T>();
	size=size*sizeof(T);	
  // check if it fits in the current page
  if (unlikely(!FitsInPage(size, stride))) {
  	NextPage();
  } 
  if (unlikely(!FitsInPage(size, stride))) {
		FATAL("The object cannot fit in one block (page) "
  	      "of memory, increase page_size and try again...\n");
  }
  index_t stride_offset = Align(current_ptr_, stride); 
  char *new_ptr = current_ptr_ + stride_offset;
	index_t new_offset=current_offset_+ stride_offset;
  current_ptr_ = new_ptr + size;
  current_offset_ = current_offset_ + stride_offset + size;
 	return new_offset;
} 

TEMPLATE__
void MEMORY_MANAGER__::DefaultInitializations() {
  num_of_pages_= cache_size_ / page_size_;
  cache_size_ = num_of_pages_ * page_size_;
  alloc_size_ = ( 1+(cache_size_ / system_page_size_))*system_page_size_;
  total_num_of_page_faults_ = 0;

}

TEMPLATE__
void MEMORY_MANAGER__::CreateNewPageOnDisk() {
  char *ptr=new char[page_size_]; 
	memset(ptr, 0, page_size_);
  AMI_err ae;
	off_t disk_offset = disk_->stream_len();
	if (unlikely((ae=disk_->seek(disk_offset))!=AMI_ERROR_NO_ERROR)) {
	  cout << "AMI_ERROR " << ae << "\n";
    FATAL("Unable to seek to %llu \n", (unsigned long long)disk_offset);
	}	
	if ((ae=disk_->write_array((Page *)(ptr),
	  	 page_size_/kTPIEPageSize))!=AMI_ERROR_NO_ERROR) {
		  cout << "AMI_ERROR " << ae << " during disk_.write_item()\n";
			exit(1);
	}
	delete []ptr;
}

};

#undef TEMPLATE__
#undef MEMORY_MANAGER__

