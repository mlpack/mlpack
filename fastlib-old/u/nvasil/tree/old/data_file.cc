#include <sys/mman.h>
#include <sys/unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <assert.h>
#include "base/basic_types.h"
#include "data_file.h"

using namespace std;
// each data point has D float32 (where D is the dimension)
// per point there is a uint32 id
// datafile header structure
// num_of_data: uint32  number of data
// dimension:   int32   dimensionality of data
// map_size:    uint32  size of bytes to memory map
// So after the header the rest bytes up to 64K are padded with zeros

void  OpenDataFile(string name, 
                   int32 *dimension, 
                   uint64 *num_of_data,
                   void** data,
                   uint64 *map_size) {
  int fd = open(name.c_str(), O_RDWR);
  if (fd < 0 ) {
  	fprintf(stderr, "couldn't open file %s\n", name.c_str());
  	assert(false);
  }  	
  read(fd, num_of_data, sizeof(uint64));
  read(fd, dimension, sizeof(int32));
  read(fd, map_size, sizeof(uint64));
  lseek(fd, 0, SEEK_SET);
 	*data = mmap(NULL, *map_size, 
               PROT_WRITE, MAP_SHARED, 
               fd, 65536);
   close(fd);
} 

void CloseDataFile(void *data, uint64 map_size) {
  munmap(data, map_size);
}
