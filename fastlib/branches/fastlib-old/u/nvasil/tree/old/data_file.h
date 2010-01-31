#ifndef DATA_FILE_H_
#define DATA_FILE_H_
#include <string>
#include "base/basic_types.h"

using namespace std;
void   OpenDataFile(string name,
                    int32 *dimension, 
                    uint64 *num_of_data,
                    void** data,
                    uint64 *map_size);
void   CloseDataFile(void *data, uint64 map_size);                   
                   

#endif /*DATA_FILE_H_*/
