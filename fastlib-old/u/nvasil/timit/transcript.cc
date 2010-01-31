/*
 * =====================================================================================
 *
 *       Filename:  transcript.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/05/2007 05:45:39 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include "base/basic_types.h"
#include "transcript.h"

Transcript *OpenBinaryTranscriptFile(string binary_transcript_file) {
	Transcript *ptr=NULL;
  int fd=open(binary_transcript_file.c_str(), O_RDWR);
	if (fd<0) {
	  fprintf(stderr, "Can not open %s, error %s\n",
				    binary_transcript_file.c_str(),
						strerror(errno));
		assert(false);
	}
	struct stat info;
  if (stat(binary_transcript_file.c_str(), &info)!=0) {
	  fprintf(stderr, "Cannot open file %s, error: %s\n",
				    binary_transcript_file.c_str(), strerror(errno));
	  assert(false);	
	}
	ptr=(Transcript *)mmap(NULL, info.st_size, PROT_READ | PROT_WRITE,
			                   MAP_SHARED, fd, 0);
	if (ptr==MAP_FAILED) {
	  fprintf(stderr, "Cannot memory map %s, error %s\n",
				    binary_transcript_file.c_str(), strerror(errno));
		assert(false);
	}
	close(fd);
	return ptr;
}

void CloseBinaryTranscriptFile(Transcript *ptr, 
		                           string binary_transcript_file) {
	struct stat info;
  if (stat(binary_transcript_file.c_str(), &info)!=0) {
	  fprintf(stderr, "Cannot open file %s, error: %s\n",
				    binary_transcript_file.c_str(), strerror(errno));
	  assert(false);	
	}
	munmap(ptr, info.st_size);
}
