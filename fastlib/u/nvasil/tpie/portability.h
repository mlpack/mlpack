//
// File: portability.h
// Created: 2002/10/30
// Authors: Joerg Rotthowe, Jan Vahrenhold, Markus Vogel
//
// $Id: portability.h,v 1.32 2005/08/24 19:34:34 adanner Exp $
//
// This header-file offers macros for independent use on Win and Unix systems.

#ifndef _PORTABILITY_H
#define _PORTABILITY_H

//  The following wil cause TPIE_OS_SIZE_T to be a 32-bit integer!
#define _TPIE_SMALL_MAIN_MEMORY

#ifdef _WIN32
#ifndef __MINGW32__
#pragma warning (disable : 4018) // signed/unsigned comparison mismatch
#pragma warning (disable : 4786) // debug identifier truncated to 255 chars.
#endif
#endif
// overview of this file:				//
//////////////////////////////////////////
// includes								//
// typedef, enum, etc.					//
// functions							//
//		non tpie specific				//
//		tpie specific					//
//			open functions				//
//			working with open files		//
//			close file functions		//
// warnings								//
// others								//



//////////////////////////////////////////////
// includes                                 //
//////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
// For time()
#include <time.h>



// Get random functions //
#include <stdlib.h>
#include <assert.h>

#ifdef _WIN32
#if (_MSC_VER < 1300) && !defined(__MINGW32__)
#include <fstream.h>
#else
#include <fstream>
using namespace std;
#endif
#else
#include <fstream>
#endif

// for reading command line parameter //
#if defined(_WIN32) && !defined(__MINGW32__)
#include <sstream>
#else
#if (__GNUC__ == 2)
#include <strstream>
#else
#include <sstream>
#endif
#endif

#include <stack>
#include <functional>
#include <algorithm>
#include <vector>
#include <queue>
#include <list>

#ifdef _WIN32
#if (_MSC_VER < 1300) && !defined(__MINGW32__)
using std::ostringstream;
using std::istringstream;
using std::stack;
using std::pair;
using std::less;
using std::lower_bound;
using std::upper_bound;
using std::unique;
using std::sort;
using std::vector;
using std::list;
using std::queue;
using std::priority_queue;
#endif
#else
using namespace std;
#endif

// Get class tms or time_t //
#ifdef _WIN32
#include <time.h>
#else
#include <sys/times.h>
#endif

#ifdef _WIN32 
#include <io.h>
#else
#define DO_NOTHING  
#endif 


#ifdef _WIN32
#define DO_NOTHING  
#else
#include <unistd.h>					
#endif	


#ifdef _WIN32
#  define DO_NOTHING  
#else
#  if USE_LIBAIO
#    include <sys/asynch.h>
#  endif
#endif

#ifdef _WIN32
#define DO_NOTHING
#else
#include <sys/time.h>				  
#endif									


#ifdef _WIN32					
#include <time.h>				
#else
#include <sys/times.h>				
#endif


#ifdef _WIN32
#define DO_NOTHING
#else
#include <sys/resource.h>	
#endif

#include <bte_err.h>

//		Get functions for mapping			//
#ifdef _WIN32
#include <windows.h>
#else
//extern "C" {
#include <sys/mman.h>
//}
#if !HAVE_PROTOTYPE_MMAP
//extern "C" mmap(caddr_t addr, size_t len, int prot, int flags,
//		int filedes, off_t off);
#endif	

#if !HAVE_PROTOTYPE_MUNMAP
//extern "C" int munmap(caddr_t addr, int len);
//extern "C" int madvise(caddr_t addr, int len, int advice);
#endif

#if !HAVE_PROTOTYPE_FTRUNCATE
//extern "C" int ftruncate(int fd, off_t length);
#endif
#endif





//////////////////////////////////////////////
// typedefs, enum etc.						//
//////////////////////////////////////////////

#ifdef _WIN32
#ifdef _WIN64
typedef __time64_t TPIE_OS_TIME_T;
inline TPIE_OS_TIME_T TPIE_OS_TIME(TPIE_OS_TIME_T* timep) {
	return _time64(timep);
}
#else
typedef time_t TPIE_OS_TIME_T;
inline TPIE_OS_TIME_T TPIE_OS_TIME(TPIE_OS_TIME_T* timep) {
	return time(timep);
}
#endif
typedef time_t TPIE_OS_TMS;
#else
typedef time_t TPIE_OS_TIME_T;
typedef tms TPIE_OS_TMS;
inline TPIE_OS_TIME_T TPIE_OS_TIME(TPIE_OS_TIME_T* timep) {
	return time(timep);
}
#endif

#ifdef _WIN32
typedef LONGLONG TPIE_OS_OFFSET;
#else
typedef off_t TPIE_OS_OFFSET;
#endif	

#ifdef _WIN32
//windows doesn't have a default way
//of printing 64 bit integers
//printf doesn't work either with %d, use %I64d in Win32
#if (_MSC_VER < 1300) && !defined(__MINGW32__)
extern ostream& operator<<(ostream& s, const TPIE_OS_OFFSET x);
#endif
#endif

#if defined (_WIN32) && !defined(__MINGW32__)
typedef long TPIE_OS_LONG;
typedef __int64 TPIE_OS_LONGLONG;
typedef unsigned __int64 TPIE_OS_ULONGLONG;
#else
typedef long TPIE_OS_LONG;
typedef long long int TPIE_OS_LONGLONG;
typedef unsigned long long int TPIE_OS_ULONGLONG;
#endif	

#if defined (_WIN32) && !defined(__MINGW32__)
typedef SSIZE_T TPIE_OS_SSIZE_T;
#ifdef _TPIE_SMALL_MAIN_MEMORY
typedef unsigned __int32 TPIE_OS_SIZE_T;
#define TPIE_OS_OUTPUT_SIZE_T TPIE_OS_SIZE_T
#else
#define TPIE_OS_OUTPUT_SIZE_T TPIE_OS_OFFSET
typedef size_t TPIE_OS_SIZE_T;
#endif
#else
typedef ssize_t TPIE_OS_SSIZE_T;
typedef size_t TPIE_OS_SIZE_T;
#define TPIE_OS_OUTPUT_SIZE_T TPIE_OS_OFFSET
#endif

#ifdef _WIN32
enum TPIE_OS_FLAG {
    TPIE_OS_FLAG_SEEK_SET = FILE_BEGIN,
    TPIE_OS_FLAG_SEEK_CUR = FILE_CURRENT,
    TPIE_OS_FLAG_SEEK_END = FILE_END,
    TPIE_OS_FLAG_PROT_READ= FILE_MAP_READ,
    TPIE_OS_FLAG_PROT_WRITE=FILE_MAP_WRITE,
    TPIE_OS_FLAG_MAP_SHARED,
    TPIE_OS_FLAG_MS_SYNC,
    TPIE_OS_FLAG_MS_ASYNC,
    TPIE_OS_FLAG_MAP_FIXED = 0
};
#else
enum TPIE_OS_FLAG {
    TPIE_OS_FLAG_SEEK_SET = SEEK_SET,
    TPIE_OS_FLAG_SEEK_CUR = SEEK_CUR,
    TPIE_OS_FLAG_SEEK_END = SEEK_END,
    TPIE_OS_FLAG_PROT_READ= PROT_READ,
    TPIE_OS_FLAG_PROT_WRITE=PROT_WRITE,
    TPIE_OS_FLAG_MAP_SHARED = MAP_SHARED,
    TPIE_OS_FLAG_MS_SYNC = MS_SYNC,
    TPIE_OS_FLAG_MS_ASYNC = MS_ASYNC,
    TPIE_OS_FLAG_MAP_FIXED = MAP_FIXED
};
#endif

#ifdef _WIN32
const int TPIE_OS_PERSIST_READ_ONCE = 0;
#else
const int TPIE_OS_PERSIST_READ_ONCE = 0;
#endif

enum TPIE_OS_MAPPING_FLAG {
	TPIE_OS_FLAG_USE_MAPPING_FALSE,
	TPIE_OS_FLAG_USE_MAPPING_TRUE	
};

#ifdef _WIN32
typedef struct {
    HANDLE FileHandle,
	mapFileHandle;	
    BOOL RDWR;
    TPIE_OS_MAPPING_FLAG useFileMapping;
} TPIE_OS_FILE_DESCRIPTOR;
#else
typedef int TPIE_OS_FILE_DESCRIPTOR;
#endif


// Default block id type
typedef TPIE_OS_OFFSET TPIE_BLOCK_ID_TYPE;


//////////////////////////////////////////////
// macros				    //
//////////////////////////////////////////////


#ifdef _WIN32
#define	 TMP_DIR ".\\"
#define TPLOGDIR ".\\"
#define TPIE_OS_TEMPNAMESTR "%s\\%s_XXXXXX"
#else
#define	TMP_DIR	"/var/tmp/"
#define TPLOGDIR "/tmp/"
#define TPIE_OS_TEMPNAMESTR "%s/%s_XXXXXX"
#endif

 
#ifdef _WIN32					
#define TPIE_OS_STL_STACK stack				
#else							
#define TPIE_OS_STL_STACK stack					
#endif							


#ifdef _WIN32					
#define TPIE_OS_STL_PAIR pair				
#else 			
#define TPIE_OS_STL_PAIR pair					
#endif	

#ifdef _WIN32	
#define TPIE_OS_SET_LIMITS_BODY	\
    return 512;
#else
#define TPIE_OS_SET_LIMITS_BODY						\
    struct rlimit limits;						\
    if(getrlimit(RLIMIT_NOFILE,&limits) == -1)   {	\
        limits.rlim_cur = 255;					       \
    }								       \
    return limits.rlim_cur;
#endif


#ifdef _WIN32
#define TPIE_OS_SET_CLOCK_TICK				\
		clock_tick = CLOCKS_PER_SEC			
#else
#define TPIE_OS_SET_CLOCK_TICK clock_tick = sysconf(_SC_CLK_TCK); elapsed.tms_utime = 0; elapsed.tms_stime = 0; elapsed.tms_cutime = 0; elapsed.tms_cstime = 0;
#endif
		

#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_SET_ELAPSED_TIME(current)
#else
#define TPIE_OS_UNIX_ONLY_SET_ELAPSED_TIME(current) elapsed.tms_utime += (current).tms_utime - last_sync.tms_utime; elapsed.tms_stime += (current).tms_stime - last_sync.tms_stime; elapsed.tms_cutime += (current).tms_cutime - last_sync.tms_cutime; elapsed.tms_cstime += (current).tms_cstime - last_sync.tms_cstime;
#endif


#ifdef _WIN32
#define TPIE_OS_SET_CURRENT_TIME(current) time(& current ); current_real = clock();
#else
#define TPIE_OS_SET_CURRENT_TIME(current) current_real = times(& current);
#endif


#ifdef _WIN32
#define TPIE_OS_LAST_SYNC_REAL_DECLARATION last_sync_real = clock();
#else
#define TPIE_OS_LAST_SYNC_REAL_DECLARATION last_sync_real = times(&last_sync);	
#endif


#ifdef _WIN32
#define TPIE_OS_USER_TIME_BODY return double(elapsed_real) / double(clock_tick)
#else
#define TPIE_OS_USER_TIME_BODY return double(elapsed.tms_utime) / double(clock_tick)
#endif


#ifdef _WIN32	
#define TPIE_OS_OPERATOR_OVERLOAD \
 return s << double(wt.elapsed_real) / double(wt.clock_tick); 
#else
#define TPIE_OS_OPERATOR_OVERLOAD return s << double(wt.elapsed.tms_utime) / double(wt.clock_tick) << "u " << double(wt.elapsed.tms_stime) / double(wt.clock_tick) << "s " << double(wt.elapsed_real) / double(wt.clock_tick);	
#endif

// for ANSI conform arrays.
#include <vararray.h>

//////////////////////////////////////////////
// functions				    //
//////////////////////////////////////////////


//////////////////////////////////////////////
//    non-tpie specific functions	    //

#ifdef _WIN32
// Generate 31 random bits using rand(), which normally generates only
// 15 random bits.
inline int TPIE_OS_RANDOM() {
  return rand() % 0x8000 + (rand() % 0x8000 << 15) + (rand() % 0x2 << 30);
}
#else
inline int TPIE_OS_RANDOM() {
    //adanner: rand and srand are ANSI standards
    //random and srandom are from old BSD systems
    //use the standard unless we run into problems
    //http://www.gnu.org/software/libc/manual/html_node/Pseudo_002dRandom-Numbers.html
    return rand();
}
#endif

#ifdef _WIN32
inline void TPIE_OS_SRANDOM(unsigned int seed) {
    srand(seed);
}
#else
inline void TPIE_OS_SRANDOM(unsigned int seed) {
    srand(seed);
}
#endif

#ifdef _WIN32
// Win32 File Seeks use high/low order offsets 
// Getting Highorder 32 Bit OFFSET
inline LONG getHighOrderOff(TPIE_OS_OFFSET off) {
    //Be careful with sign bits. 
    return (LONG)((ULONGLONG)(off)>>32);
}
 
     
// Getting Loworder 32 Bit OFFSET
inline LONG getLowOrderOff(TPIE_OS_OFFSET off) {
    return (LONG)((ULONGLONG)(off) % 0x000100000000ULL);
}
#endif



//////////////////////////////////////////////
//       tpie specific functions            //

#ifdef _WIN32
inline TPIE_OS_SIZE_T  TPIE_OS_PAGESIZE() {
    SYSTEM_INFO systemInfos;
    GetSystemInfo(&systemInfos);
    return (TPIE_OS_SIZE_T )systemInfos.dwPageSize;
}
#else
#ifdef _SC_PAGE_SIZE
inline TPIE_OS_SIZE_T  TPIE_OS_PAGESIZE() {
    return sysconf (_SC_PAGE_SIZE);
}
#else
inline TPIE_OS_SIZE_T  TPIE_OS_PAGESIZE() {
    return getpagesize();
}
#endif
#endif


#ifdef _WIN32
inline TPIE_OS_SIZE_T  TPIE_OS_BLOCKSIZE() {
    SYSTEM_INFO systemInfos;
    GetSystemInfo(&systemInfos);
    return systemInfos.dwAllocationGranularity;
}
#else
#ifdef _SC_PAGE_SIZE
inline TPIE_OS_SIZE_T TPIE_OS_BLOCKSIZE() {
    return sysconf (_SC_PAGE_SIZE);
}
#else
inline TPIE_OS_SIZE_T  TPIE_OS_BLOCKSIZE() {
    return getpagesize();
}
#endif
#endif



//////////////////////////////////////////////
//	   open functions		    //


//there is no difference between the systemcalls
//but for later adaptation to other systems it maybe useful
#ifdef _WIN32
inline FILE* TPIE_OS_FOPEN(const char* filename,
			   const char* mode) {
    return fopen(filename,mode);
}
#else
inline FILE* TPIE_OS_FOPEN(const char* filename,
			   const char* mode) {
    return fopen(filename,mode);
}
#endif

//there is no difference between the systemcalls
//but for later adaptation to other systems it maybe useful
#ifdef _WIN32
inline int TPIE_OS_FSEEK(FILE* file, TPIE_OS_OFFSET offset, int whence) {
	//  Please note that the second parameter should be TPIE_OS_OFFSET
	//  instead of int. This is due to the fact that VS2003 does not
	//  support large files with fopen/fseek etc.
    return fseek(file, static_cast<int>(offset), whence);
}
#else
inline int TPIE_OS_FSEEK(FILE* file, TPIE_OS_OFFSET offset, int whence) {
    return fseek(file, offset, whence);
}
#endif

//there is no difference between the systemcalls
//but for later adaptation to other systems it maybe useful
#ifdef _WIN32
inline TPIE_OS_LONG TPIE_OS_FTELL(FILE* file) {
    return ftell(file);
}
#else
inline TPIE_OS_LONG TPIE_OS_FTELL(FILE* file) {
    return ftell(file);
}
#endif

//there is no difference between the systemcalls
//but for later adaptation to other systems it maybe useful
#ifdef _WIN32
inline size_t TPIE_OS_FREAD(void* buffer, size_t size, size_t nitems, FILE* stream) {
    return fread(buffer, size, nitems, stream);
}
#else
inline size_t TPIE_OS_FREAD(void* buffer, size_t size, size_t nitems, FILE* stream) {
    return fread(buffer, size, nitems, stream);
}
#endif

//there is no difference between the systemcalls
//but for later adaptation to other systems it maybe useful
#ifdef _WIN32
inline size_t TPIE_OS_FWRITE(const void* buffer, size_t size, size_t nitems, FILE* stream) {
    return fwrite(buffer, size, nitems, stream);
}
#else
inline size_t TPIE_OS_FWRITE(const void* buffer, size_t size, size_t nitems, FILE* stream) {
    return fwrite(buffer, size, nitems, stream);
}
#endif

//there is no difference between the systemcalls
//but for later adaptation to other systems it maybe useful
#ifdef _WIN32
inline int TPIE_OS_FCLOSE(FILE* file) {
    return fclose(file);
}
#else
inline int TPIE_OS_FCLOSE(FILE* file) {
    return fclose(file);
}
#endif


// internal help-function for translating Unix open into a Windows CreateFile + CreateFileMapping
#ifdef _WIN32
#include <FCNTL.h>
inline TPIE_OS_FILE_DESCRIPTOR portabilityInternalOpen(LPCTSTR name, int flag, TPIE_OS_MAPPING_FLAG mappingFlag) {
    TPIE_OS_FILE_DESCRIPTOR internalHandle;	
    switch(flag) {
    case _O_RDONLY: 
	internalHandle.RDWR = false;
	internalHandle.FileHandle =	CreateFile(	
	    name,
	    GENERIC_READ, 
		FILE_SHARE_READ | FILE_SHARE_WRITE,
		0,
	    OPEN_EXISTING, 
	    0, 0);
	break;
    case _O_EXCL:	
	internalHandle.RDWR = true;
	internalHandle.FileHandle =	CreateFile(	
	    name,
	    GENERIC_READ | GENERIC_WRITE, 
		FILE_SHARE_READ | FILE_SHARE_WRITE,
		0,
	    CREATE_NEW, 
	    0, 0);
	break;
    case _O_RDWR:	
	internalHandle.RDWR = true;
	internalHandle.FileHandle =	CreateFile(	
	    name,
	    GENERIC_READ | GENERIC_WRITE, 
        FILE_SHARE_READ | FILE_SHARE_WRITE,
		0,
	    OPEN_EXISTING, 
	    0, 0);
	break;
    default :		
	internalHandle.RDWR = false;
	internalHandle.FileHandle =	CreateFile(	
	    name,
	    GENERIC_READ, 
		FILE_SHARE_READ | FILE_SHARE_WRITE,
		0,
	    OPEN_EXISTING, 
	    0, 0);
    };
    internalHandle.useFileMapping = mappingFlag;
    DWORD dwFileSize = GetFileSize(internalHandle.FileHandle,NULL);
    if (dwFileSize == 0) {
	SetFilePointer(internalHandle.FileHandle,
		       static_cast<LONG>(TPIE_OS_BLOCKSIZE()),0,FILE_BEGIN);
	SetEndOfFile(internalHandle.FileHandle);
    };
    if (internalHandle.useFileMapping == TPIE_OS_FLAG_USE_MAPPING_TRUE) {
	internalHandle.mapFileHandle = 
	    CreateFileMapping( 
		internalHandle.FileHandle,
		0,  
		(internalHandle.RDWR ? PAGE_READWRITE : PAGE_READONLY),	
		0, 0,
		NULL);
    }
    else {
	internalHandle.mapFileHandle = (void*)1;
    }
    return internalHandle;
}
#endif


#ifdef _WIN32
inline TPIE_OS_FILE_DESCRIPTOR TPIE_OS_OPEN_ORDONLY(const char* name,TPIE_OS_MAPPING_FLAG mappingFlag = TPIE_OS_FLAG_USE_MAPPING_FALSE) {
    return portabilityInternalOpen(name, _O_RDONLY,mappingFlag);
}
#else
inline TPIE_OS_FILE_DESCRIPTOR TPIE_OS_OPEN_ORDONLY(const char* name,TPIE_OS_MAPPING_FLAG mappingFlag = TPIE_OS_FLAG_USE_MAPPING_FALSE) {
    return ::open(name, O_RDONLY);
}
#endif


#ifdef _WIN32
inline TPIE_OS_FILE_DESCRIPTOR TPIE_OS_OPEN_OEXCL(const char* name, TPIE_OS_MAPPING_FLAG mappingFlag = TPIE_OS_FLAG_USE_MAPPING_FALSE) {
    return portabilityInternalOpen(name, _O_EXCL, mappingFlag);
}
#else
inline TPIE_OS_FILE_DESCRIPTOR TPIE_OS_OPEN_OEXCL(const char* name, TPIE_OS_MAPPING_FLAG mappingFlag = TPIE_OS_FLAG_USE_MAPPING_FALSE) {
    return ::open(name, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
}
#endif


#ifdef _WIN32
inline TPIE_OS_FILE_DESCRIPTOR TPIE_OS_OPEN_ORDWR(const char* name, TPIE_OS_MAPPING_FLAG mappingFlag = TPIE_OS_FLAG_USE_MAPPING_FALSE) {
    return portabilityInternalOpen(name, _O_RDWR, mappingFlag);
}
#else
inline TPIE_OS_FILE_DESCRIPTOR TPIE_OS_OPEN_ORDWR(const char* name, TPIE_OS_MAPPING_FLAG mappingFlag = TPIE_OS_FLAG_USE_MAPPING_FALSE) {
    return ::open(name, O_RDWR);
}
#endif


//////////////////////////////////////////////
//	    working with open files         //

#ifdef _WIN32
inline BOOL TPIE_OS_IS_VALID_FILE_DESCRIPTOR(TPIE_OS_FILE_DESCRIPTOR& fd) {
    BOOL x;
    if (fd.FileHandle == INVALID_HANDLE_VALUE || fd.mapFileHandle == NULL) {
	x = false;	
    } else {
	x = true;
    };
    return x;
}
#else
inline bool TPIE_OS_IS_VALID_FILE_DESCRIPTOR(TPIE_OS_FILE_DESCRIPTOR& fd) {
    return (fd == -1 ? false : true);
}
#endif


// for working with HANDLEs under Windows we have to use SetFilePointer instead of _lseek //
#ifdef _WIN32
inline TPIE_OS_OFFSET TPIE_OS_LSEEK(TPIE_OS_FILE_DESCRIPTOR &fd,TPIE_OS_OFFSET offset,TPIE_OS_FLAG origin) {	
    LONG highOrderOff = getHighOrderOff(offset);	
    DWORD x = SetFilePointer(fd.FileHandle,getLowOrderOff(offset),&highOrderOff,origin);

	if( x==0xFFFFFFFF && (GetLastError() != NO_ERROR) ){
	  //Error
	  return -1;
	}
	else{
      return TPIE_OS_OUTPUT_SIZE_T((((ULONGLONG) highOrderOff)<<32)+(ULONGLONG) x);
	}
}
#else
inline TPIE_OS_OFFSET TPIE_OS_LSEEK(TPIE_OS_FILE_DESCRIPTOR &fd,TPIE_OS_OFFSET offset,TPIE_OS_FLAG origin) {
    return ::lseek(fd, offset, origin);
}
#endif

    
#ifdef _WIN32
inline TPIE_OS_SSIZE_T TPIE_OS_WRITE(TPIE_OS_FILE_DESCRIPTOR fd, const void* buffer, TPIE_OS_SIZE_T count) {
    DWORD bytesWritten = 0;
	::WriteFile(fd.FileHandle, buffer, (DWORD)count, &bytesWritten, 0);
    return (TPIE_OS_SSIZE_T)(bytesWritten > 0 ? bytesWritten : -1);
}
#else
inline TPIE_OS_SSIZE_T TPIE_OS_WRITE(TPIE_OS_FILE_DESCRIPTOR fd, const void* buffer, size_t count) {
    return ::write(fd,buffer,count);
}
#endif

#ifdef _WIN32
inline TPIE_OS_SSIZE_T TPIE_OS_READ(TPIE_OS_FILE_DESCRIPTOR fd, void* buffer, TPIE_OS_SIZE_T count) {
    DWORD bytesRead = 0;
    ReadFile(fd.FileHandle, buffer, (DWORD)count, &bytesRead, 0);
    return (bytesRead > 0 ? bytesRead : -1);
}
#else
inline TPIE_OS_SSIZE_T TPIE_OS_READ(TPIE_OS_FILE_DESCRIPTOR fd, void* buffer, size_t count) {
    return ::read(fd,buffer,count);
}
#endif

#ifdef _WIN32
// The suggested starting address of the mmap call has to be
// a multiple of the systems granularity (else the mapping fails)
// Hence, the parameter addr is not used at present.
inline LPVOID TPIE_OS_MMAP(LPVOID addr, 
			   size_t len, 
			   int     prot, 
			   int     flags, 
			   TPIE_OS_FILE_DESCRIPTOR fildes, 
			   TPIE_OS_OFFSET off) {
    return MapViewOfFileEx( fildes.mapFileHandle,
			    prot,
			    getHighOrderOff(off),		       
			    getLowOrderOff(off),
			    len, NULL);
}
#else	
inline void* TPIE_OS_MMAP(void* addr, size_t len, int prot, int flags, TPIE_OS_FILE_DESCRIPTOR fildes, TPIE_OS_OFFSET off) {
    return mmap((caddr_t)addr, len, prot, flags, fildes, off);
}
#endif


#ifdef _WIN32
inline int TPIE_OS_MUNMAP(LPVOID addr, size_t len) {
    return (UnmapViewOfFile(addr) == 0 ? -1 : 0);
}
#else							
inline int TPIE_OS_MUNMAP(void* addr, size_t len) {
    return munmap((caddr_t)addr, len);
}
#endif


#ifdef _WIN32
inline int TPIE_OS_MSYNC(LPVOID addr, size_t len, int flags=0) {
    return (FlushViewOfFile(addr,len) ? 0 : -1);
}
#else							
inline int TPIE_OS_MSYNC(char* addr, size_t len,int flags) {
    return msync(addr, len, flags);
}
#endif


#ifdef _WIN32

// Force the use of truncate to lengthen a collection under WIN32, due
// to mapping issues.
#ifdef BTE_COLLECTION_USE_FTRUNCATE
#undef BTE_COLLECTION_USE_FTRUNCATE
#endif
#define BTE_COLLECTION_USE_FTRUNCATE 1

inline int TPIE_OS_FTRUNCATE(TPIE_OS_FILE_DESCRIPTOR& fd, TPIE_OS_OFFSET length) {
  // Save the offset
  TPIE_OS_OFFSET so = TPIE_OS_LSEEK(fd, 0, TPIE_OS_FLAG_SEEK_CUR);
  if (fd.useFileMapping == TPIE_OS_FLAG_USE_MAPPING_TRUE) {
    CloseHandle(fd.mapFileHandle);	
  }
  LONG highOrderOff = getHighOrderOff(length);
  int x = ((((fd).RDWR == false) 	|| 
	    (SetFilePointer((fd).FileHandle,getLowOrderOff(length),&highOrderOff,FILE_BEGIN) == 0xFFFFFFFF)	|| 
	    (SetEndOfFile((fd).FileHandle) == 0)) ? -1 : 0);

  if (fd.useFileMapping == TPIE_OS_FLAG_USE_MAPPING_TRUE) {
    fd.mapFileHandle= CreateFileMapping( (fd).FileHandle,
					 0,
					 ((fd).RDWR ? PAGE_READWRITE : PAGE_READONLY),	
					 0, 0,
					 NULL);
  }
  // Restore the offset, mimicking the ftruncate() behavior.
  TPIE_OS_LSEEK(fd, (so < length ? so : length), TPIE_OS_FLAG_SEEK_SET);
  return x;
}
#else							
inline int TPIE_OS_FTRUNCATE(TPIE_OS_FILE_DESCRIPTOR& fd, TPIE_OS_OFFSET length) {
    return ftruncate(fd, length);
}
#endif



//////////////////////////////////////////////	
//				tpie close file functions	//


#ifdef _WIN32
inline int TPIE_OS_CLOSE(TPIE_OS_FILE_DESCRIPTOR fd) {	
	if (fd.useFileMapping == TPIE_OS_FLAG_USE_MAPPING_TRUE) {
		return ((  (CloseHandle(fd.mapFileHandle) != 0) &&	
			   (CloseHandle(fd.FileHandle) != 0)) ? 0 : -1);
	} else {
		return ((CloseHandle(fd.FileHandle) != 0) ? 0 : -1);
	}    
}
#else
inline int TPIE_OS_CLOSE(TPIE_OS_FILE_DESCRIPTOR fd) {
    return ::close(fd);
}
#endif


#if defined(_WIN32) && !defined(__MINGW32__)
inline int TPIE_OS_UNLINK(const char* filename) {
    return _unlink(filename);
}
#else
inline int TPIE_OS_UNLINK(const char* filename) {
    return ::unlink(filename);
}
#endif





//////////////////////////////////////////////	
// warnings									//
//////////////////////////////////////////////	

#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_AMI_IMP_SINGLE 			
#else
#define TPIE_OS_UNIX_ONLY_WARNING_AMI_IMP_SINGLE \
#warning The AMI_IMP_SINGLE flag is obsolete. \
#warning Please use AMI_STREAM_IMP_SINGLE.\
#warning Implicitly defining AMI_STREAM_IMP_SINGLE.	
#endif							


#ifdef _WIN32	
#define TPIE_OS_UNIX_ONLY_WARNING_BTE_COLLECTION_IMP_MMB		
#else
#define TPIE_OS_UNIX_ONLY_WARNING_BTE_COLLECTION_IMP_MMB \
#warning The BTE_COLLECTION_IMP_MMB flag is obsolete.\
#warning Please use BTE_COLLECTION_IMP_MMAP. \
#warning Implicitly defining BTE_COLLECTION_IMP_MMAP.	
#endif


#ifdef _WIN32 				
#define TPIE_OS_UNIX_ONLY_WARNING_MULTIPLE_BTE_COLLECTION_IMP_DEFINED	
#else
#define TPIE_OS_UNIX_ONLY_WARNING_MULTIPLE_BTE_COLLECTION_IMP_DEFINED \
#warning Multiple BTE_COLLECTION_IMP_* defined.	\
#warning Undetermined default implementation. \
#warning Implicitly defining BTE_COLLECTION_IMP_MMAP. 
#endif 

		
// To avoid this warning, define one of: BTE_COLLECTION_IMP_MMAP.	//
// BTE_COLLECTION_IMP_UFS, BTE_COLLECTION_USER_DEFINED			//
// in your app_config.h file.                                   //
#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_NO_DEFAULT_BTE_COLLECTION 
#else
#define	TPIE_OS_UNIX_ONLY_WARNING_NO_DEFAULT_BTE_COLLECTION \
#warning No default BTE_COLLECTION implementation defined, using BTE_COLLECTION_IMP_MMAP by default.	
#endif


#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_UFS
#else
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_UFS \
#warning The BTE_IMP_UFS flag is obsolete. Please use BTE_STREAM_IMP_UFS. \
#warning Implicitly defining BTE_STREAM_IMP_UFS. 
#endif						


#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_MMAP
#else
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_MMAP \
#warning The BTE_IMP_MMB flag is obsolete. Please use BTE_STREAM_IMP_MMAP. \
#warning Implicitly defining BTE_STREAM_IMP_MMAP. 
#endif


#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_STDIO
#else
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_STDIO  \
#warning The BTE_IMP_STDIO flag is obsolete. Please use BTE_STREAM_IMP_STDIO.\
#warning Implicitly defining BTE_STREAM_IMP_STDIO.
#endif


#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_USER_DEFINED
#else
#define TPIE_OS_UNIX_ONLY_WARNING_USE_BTE_STREAM_IMP_USER_DEFINED  \
#warning The BTE_IMP_USER_DEFINED flag is obsolete. Please use BTE_STREAM_IMP_USER_DEFINED.\
#warning Implicitly defining BTE_STREAM_IMP_USER_DEFINED. 
#endif


#ifdef _WIN32
#define TPIE_OS_UNIX_ONLY_WARNING_MULTIPLE_BTE_STREAM_IMP_DEFINED
#else
#define TPIE_OS_UNIX_ONLY_WARNING_MULTIPLE_BTE_STREAM_IMP_DEFINED \
#warning Multiple BTE_STREAM_IMP_* defined, but BTE_STREAM_IMP_MULTI_IMP undefined.\
#warning Implicitly defining BTE_STREAM_IMP_MULTI_IMP.			
#endif


#ifdef _WIN32
#define	TPIE_OS_UNIX_ONLY_WARNING_NO_IMPLEMENTATION_USING_BTE_STREAM_IMP_UFS
#else
#define	TPIE_OS_UNIX_ONLY_WARNING_NO_IMPLEMENTATION_USING_BTE_STREAM_IMP_UFS \
#warning No implementation defined.  Using BTE_STREAM_IMP_UFS by default.
#endif

#ifdef _WIN32
inline int TPIE_OS_TRUNCATE(FILE* file, const char* path, TPIE_OS_OFFSET offset) {
//    TPIE_OS_LONG highOrderOff = getHighOrderOff(offset);	
//    DWORD x = SetFilePointer(fd.FileHandle,getLowOrderOff(offset),&highOrderOff, FILE_BEGIN);
//	return SetEndOfFile(fd.FileHandle);
	// Check for  64-bit file length! (jv)
	return _chsize(file->_file, (LONG)offset);
}
#else
inline int TPIE_OS_TRUNCATE(FILE* file, const char* path, TPIE_OS_OFFSET offset) {
	return ::truncate(path, offset);
}
#endif

#ifdef _WIN32
#define TPIE_OS_TRUNCATE_STREAM_TEMPLATE_CLASS_BODY \
LOG_FATAL_ID("_WIN32 does not support truncate() for "); \
LOG_FATAL_ID(path); \
return BTE_ERROR_OS_ERROR 
#else
#define TPIE_OS_TRUNCATE_STREAM_TEMPLATE_CLASS_BODY off_t file_position; if (substream_level) { return BTE_ERROR_STREAM_IS_SUBSTREAM; } if (offset < 0) {   return BTE_ERROR_OFFSET_OUT_OF_RANGE; } file_position = offset * sizeof (T) + os_block_size_; if (::truncate (path, file_position)) {   os_errno = errno;  TP_LOG_FATAL_ID("Failed to truncate() to the new end of file:");  TP_LOG_FATAL_ID(path);  TP_LOG_FATAL_ID(strerror (os_errno));   return BTE_ERROR_OS_ERROR; } if (fseek (file, file_position, SEEK_SET)) { LOG_FATAL("fseek failed to go to position " << file_position << " of \"" << "\"\n"); LOG_FLUSH_LOG;  return BTE_ERROR_OS_ERROR; } f_offset = file_position; f_eof = file_position; return BTE_ERROR_NO_ERROR
#endif 

/*
#ifdef _WIN32	
#define TPIE_OS_WIN_ONLY_TEMPLATE_MERGE_HEAP_ELEMENT_COMPILER_FOOLER template<> class merge_heap_element<int>{};	
#else
#define TPIE_OS_WIN_ONLY_TEMPLATE_MERGE_HEAP_ELEMENT_COMPILER_FOOLER	
#endif
*/




//////////////////////////////////////////////	
// others									//
//////////////////////////////////////////////	

// config.h needs in line 35:				 //
// For WIN32, do not include <unistd.h>		 //
// Where is unistd.h?						 //


// #ifndef HAVE_UNISTD_H
// #define HAVE_UNISTD_H 0
// #endif
// #ifndef HAVE_SYS_UNISTD_H
// #define HAVE_SYS_UNISTD_H 0
// #endif

#if defined(_WIN32) && !defined(__MINGW32__)
#define HAVE_UNISTD_H 0
#define HAVE_SYS_UNISTD_H 0
#endif


//  WIN32 does not support data type "long long", but does support "LONGLONG".//
#ifdef _WIN32
#define TPIE_OS_DECLARE_LOGSTREAM_LONGLONG _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const LONGLONG); 
#else
#define TPIE_OS_DECLARE_LOGSTREAM_LONGLONG _DECLARE_LOGSTREAM_OUTPUT_OPERATOR(const long long);
#endif

#ifdef _WIN32
#define TPIE_OS_DEFINE_LOGSTREAM_LONGLONG  \
logstream& logstream::operator<<(const LONGLONG x)\
{\
    char buf[30];\
    sprintf(buf,"%I64d",x);\
    if (priority <= threshold) {\
          ofstream::operator<<(buf);\
    }\
    return *this;\
}

#else
#define TPIE_OS_DEFINE_LOGSTREAM_LONGLONG _DEFINE_LOGSTREAM_OUTPUT_OPERATOR(long long);					
#endif


#ifdef _WIN32	
#define VERSION(name,id) static char __ ## name[] = id;      
#else
#define VERSION(name,id) 
// static char __ ## name[] = id;
//#define VERSION(name,id) static char __ ## name[] = ## id; static struct __ ## name ## _compiler_fooler {	char *pc; __ ## name ## _compiler_fooler *next; } the__ ## name ## _compiler_fooler = { __ ## name, & the__ ## name ## _compiler_fooler};
#endif


    //		//**
    //		*void * operator new() - Get a block of memory from the debug heap
    //		*
    //		*Purpose:
    //		*       Allocate of block of memory of at least size bytes from the heap and
    //		*       return a pointer to it.
    //		*
    //		*       Allocates any type of supported memory block.
    //		*
    //		*Entry:
    //		*       unsigned int    cb          - count of bytes requested
    //		*       int             nBlockUse   - block type
    //		*       char *          szFileName  - file name
    //		*       int             nLine       - line number
    //		*
    //		*Exit:
    //		*       Success:  Pointer to memory block
    //		*       Failure:  NULL (or some error value)
    //		*
    //		*Exceptions:
    //  	*
    //		******************************************************************************* /

#ifdef _WIN32	
#ifndef NDEBUG
#define TPIE_OS_SPACE_OVERHEAD_BODY \
void * __cdecl _nh_malloc_dbg ( size_t, int, int, const char *,	int );\
void * operator new(\
		    unsigned int cb,\
		    int nBlockUse,\
		    const char * szFileName,\
		    int nLine\
		    )\
{\
  void *p;\
  if ((MM_manager.register_new != MM_IGNORE_MEMORY_EXCEEDED)\
      && (MM_manager.register_allocation (cb + SIZE_SPACE) !=	MM_ERROR_NO_ERROR)) {\
    switch(MM_manager.register_new) {\
    case MM_ABORT_ON_MEMORY_EXCEEDED:\
       TP_LOG_FATAL_ID("In operator new() - allocation request ");\
       TP_LOG_FATAL((TPIE_OS_LONG)(cb + SIZE_SPACE));\
       TP_LOG_FATAL(" plus previous allocation ");\
       TP_LOG_FATAL((TPIE_OS_LONG)(MM_manager.memory_used() - (cb + SIZE_SPACE)));\
       TP_LOG_FATAL(" exceeds user-defined limit ");\
       TP_LOG_FATAL((TPIE_OS_LONG)(MM_manager.memory_limit()));\
       TP_LOG_FATAL(" ");\
        cerr << "memory manager: memory allocation limit " << (TPIE_OS_LONG)MM_manager.memory_limit() << " exceeded while allocating " << (TPIE_OS_LONG)cb << " bytes" << endl;\
        exit(1);\
        break;\
    case MM_WARN_ON_MEMORY_EXCEEDED: \
      TP_LOG_WARNING_ID("In operator new() - allocation request \"");\
      TP_LOG_WARNING((TPIE_OS_LONG)(cb + SIZE_SPACE));\
      TP_LOG_WARNING("\" plus previous allocation \"");\
      TP_LOG_WARNING((TPIE_OS_LONG)(MM_manager.memory_used () - (cb + SIZE_SPACE)));\
      TP_LOG_WARNING("\" exceeds user-defined limit \"");\
      TP_LOG_WARNING((TPIE_OS_LONG)(MM_manager.memory_limit ()));\
      TP_LOG_WARNING("\" \n");\
      TP_LOG_FLUSH_LOG;\
       cerr << "memory manager: memory allocation limit " << (TPIE_OS_LONG)MM_manager.memory_limit () << " exceeded " << "while allocating " << (TPIE_OS_LONG)cb << " bytes" << endl;\
       break;\
   case MM_IGNORE_MEMORY_EXCEEDED:\
       break;\
   }\
   }\
   p = malloc (cb+SIZE_SPACE);\
   if (!p) {\
      TP_LOG_FATAL_ID("Out of memory. Cannot continue.");\
      TP_LOG_FLUSH_LOG;\
       cerr << "out of memory while allocating " << (TPIE_OS_LONG)cb << " bytes" << endl;\
       perror ("mm_base::new malloc");\
       assert(0);\
       exit (1);\
   }\
   *((size_t *) p) = cb;\
   return ((char *) p) + SIZE_SPACE;\
};
#endif
#else							
#define TPIE_OS_SPACE_OVERHEAD_BODY //
#endif

#endif 

// _portability_H  //
