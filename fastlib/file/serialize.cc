/**
 * @file serialize.cc
 *
 * Definitions for serialization methods.
 */

#include "serialize.h"

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

file::magic_t file::CreateMagic(const char *str) {
  magic_t magic = 0x31415926;
  
  for (; *str != '\0'; str++) {
    magic = (magic << 9) + (magic >> 11) + *str;
  }
  
  return magic;
}

success_t NativeArraySerializer::WriteFile(const char *fname) {
  FILE *f = fopen(fname, "wb");
  
  if (unlikely(!f)) {
    return SUCCESS_FAIL;
  }
  
  if (unlikely(index_t(fwrite(ptr(), 1, size(), f)) != index_t(size()))) {
    (void) fclose(f);
    return SUCCESS_FAIL;
  }
  
  return SUCCESS_FROM_INT(fclose(f));
}

success_t NativeFileDeserializer::Init(const char *fname) {
  struct stat info;
  FILE *f;
  size_t file_size;
  char *data;
  
  DEBUG_MSG(1.0, "Opening a native file deserializer.");
  
  if (unlikely(stat(fname, &info) < 0)) {
    NONFATAL("File [%s] does not exist.", fname);
    data = NULL;
  } else {
    f = fopen(fname, "rb");
    
    if (unlikely(!f)) {
      data = NULL;
      NONFATAL("Cannot read [%s], but it exists.", fname);
    } else {
      file_size = size_t(info.st_size);
      data = mem::Alloc<char>(file_size);
      
      if (unlikely(fread(data, 1, file_size, f) != file_size)) {
        mem::Free(data);
        data = NULL;
        NONFATAL("Error reading contents of [%s].", fname);
      }
      
      (void) fclose(f);
    }
  }
  
  if (data == NULL) {
    // We have to initialize this into a valid state, or else the program
    // will segfault.
    file_size = 0;
  }
  
  NativeArrayDeserializer::Init(data, file_size);
  
  return (data != NULL) ? SUCCESS_PASS : SUCCESS_FAIL;
}
