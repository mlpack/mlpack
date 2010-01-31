#ifndef DATA_READER_H_
#define DATA_READER_H_
#include <string.h>
#include "base/basic_types.h"

template<typename DATAPRECISION=float32, typename IDPRECISION=uint64>
class DataReader {
 public:
  DataReader(void *source, uint64 data_length) {
  	source_ = source;
  	data_length_ = data_length;
  }
  ~DataReader() { 
  }
  void Assign(DataReader<DATAPRECISION, IDPRECISION> &d, uint64 index) {
  	 source_ = d.At(index);
  }
  void Swap(uint64 ind1, uint64 ind2) {
  	char temp[DataSize()];
  	memcpy(temp, (char*)source_+TranslateIndex(ind1), DataSize());
  	memcpy((char*)source_+TranslateIndex(ind1), (char*)source_+TranslateIndex(ind2),
  	       DataSize());
    memcpy((char*)source_+TranslateIndex(ind2), temp, DataSize());
  }	         
  DATAPRECISION *At(uint64 index) {
  	return (DATAPRECISION *)((char*)source_ + TranslateIndex(index)); 
  }
  IDPRECISION GetId(uint64 index) {
  	return *((IDPRECISION*)((char *)source_ + 
  	    index * DataSize() + data_length_ * sizeof(DATAPRECISION)));
  }
  uint64 DataSize() {
  	return data_length_*sizeof(DATAPRECISION)+sizeof(IDPRECISION);
  }
	void *get_source() {
		return source_;
	}
 private:
  uint64 TranslateIndex(uint64 index) {
  	return index *DataSize();
  }
  void *source_;
  uint64 data_length_;
};
  
#endif /*DATA_READER_H_*/
