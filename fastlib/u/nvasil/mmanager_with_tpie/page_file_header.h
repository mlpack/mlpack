/*
 * =====================================================================================
 * 
 *       Filename:  page_file_header.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/09/2007 06:37:38 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#include "fastlib/fastlib.h"
// The header of the file contains the following information
// int          : version
// unsigned long: total number of number of ram_pages
// CacheSizeNum : disk page size  (this is the cache_size_)
// RamPageNum : ram page size
class PageFileHeader {
 public:
  int32 version_;
  uint64 total_pages_;
  uint32 cache_size_;
  uint32 page_size_;
  int32  last_offset_;
  PageFileHeader(int32 version,
                 uint64 total_pages,
                 uint32  cache_size,
                 int32 last_offset) {
    version_ = version;
    total_pages_ = total_pages;
    cache_size_ = cache_size;
    last_offset_ = last_offset;
  };
  PageFileHeader() {
    version_ = 0;
    total_pages_ = 0;
    cache_size_ = 0;
    last_offset_ = 0;
  }
};


