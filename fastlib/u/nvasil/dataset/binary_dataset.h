/*
 * =====================================================================================
 * 
 *       Filename:  dataset.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  04/24/2007 10:25:35 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#ifndef BINARY_DATASET_
#define BINARY_DATASET_
#include <sys/unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string>
#include "fastlib/fastlib.h"
// BinaryDataset
// Use this class to read a binary data file that is also 
// accompanied by a file with the index values (uint64)
// It is templetized so that you can use it with different precisions
// for data
template <typename PRECISION>
class BinaryDataset {
  FORBID_COPY(BinaryDataset<PRECISION>);
	friend class BinaryDatasetTest;
 public:
	typedef PRECISION Precision_t;
	friend class BinaryDataset<Precision_t>::Iterator;
	class Iterator {
	 public:
		Iterator(BinaryDataset<Precision_t> *dataset) {
		  set_=dataset;
		} 
		Iterator &operator=(const Iterator &other) {
		}
		Iterator operator++();
		Iterator operator--();
		bool operator==(const Iterator &other);
		bool operator!=(const Iterator &other);
		Iterator & operator*();

	 private:
	  BinaryDataset<Precision_t> *set_;	
	};
	BinaryDataset() {
	  data_file_="";
		index_file_="";
		data_=NULL;
		index_=NULL;

	}
	~BinaryDataset() {
	  Destruct();
	}
	// Initializes a the BinaryDataset from two existent files
	// one for data and one for the index
  success_t Init(string data_file, string index_file) {
	  data_file_=data_file;
		index_file_=index_file;
		ReadDimNumOfPointsFromFile(data_file_);
    data_=(Precision_t*)MemoryMap(data_file_, sizeof(int32));
		index_=(uint64*)MemoryMap(index_file_,0);
		return SUCCESS_PASS;
	}
	// If the index file is not given it assumes it has the same name as
	// the data file appended with ind
	success_t Init(string data_file) {
    string temp=data_file.append("ind");
	  Init(data_file, temp);	
		return SUCCESS_PASS;
	}
	// Use this to create a new BinarayDataset file, If you omit 
	// the index file it will create a index file appended with .ind
	// extension. 
  success_t	Init(string data_file, uint64 num_of_points, 
			           int32 dimension) {
	  data_file_=data_file;
		index_file_=data_file;
		data_file_.append("ind");
		Init(data_file_, index_file_, num_of_points_, dimension);
		return SUCCESS_PASS;
	}
	
	success_t Init(string data_file, string index_file, 
			           uint64 num_of_points, int32 dimension) {
	  data_file_=data_file;
		index_file_=index_file;
		num_of_points_=num_of_points;
		dimension_=dimension;
		CreateDataFile(); 
		CreateIndexFile();
		return SUCCESS_PASS;
	}
  // Use this to swap the points of a dataset
	// swaps the index values as well
	void Swap(uint64 i, uint64 j) {
	  Precision_t temp[dimension_];
		memcmp(temp, At(j), dimension_, sizeof(Precision_t));
		memcmp(At(j), At(i), dimension_, sizeof(Precision_t));
		memcmp(At(i), temp, dimension_, sizeof(Precision_t));
    uint64 temp_index=index_[i];
		       index_[i]=index_[j];
					 index_[j]=temp;
	}	
	// Use this for destruction
	void Destruct() {
	  MemoryUnmap(data_, data_file_, sizeof(int32)); 
	  MemoryUnmap(data_, index_file_, 0); 
	}
	// returns a matrix on the data
	Matrix get_data_matrix() {
		Matrix matrix;
		matrix.Alias(data_, num_of_points_/dimension_, dimension_);
		return matrix;
	}
	// returns a vector on the index
	Vector get_index_vector() {
		Vector vector;
		vector.Alias(index_, num_of_points_);
		return vector;
	}
	// returns a pointer on the data at the ith point
	Precision_t* At(index_t i) {
		DEBUG_ASSERT_MSG(i<num_of_points_, 
				             "Attempt to acces data out of range "LI">"LI, i, 
										  num_of_points_);
	  return data_+i*dimension_;
	}
	// returns a reference on the i,j element
	Precision_t &At(index_t i, index_t j) {
    DEBUG_ASSERT_MSG(i<num_of_points_, 
		                "Attempt to acces data out of range "LI">"LI, i, 
										   num_of_points_);
    DEBUG_ASSERT_MSG(j<dimension_,
				             "Attempt to access element greater that the dimension "
										 LI">"L32, j, dimension_);
	  return data_[i*dimension_+j];
	}
	// get the index at i point
	uint64 get_id(index_t i) {
	  return index_[i];
	}
	// set id at point i
	void set_id(index_t i, index_t value) {
	  index_[i]=value;
	}
	// get the number of points
	uint64 get_num_of_points() {
	  return num_of_points_;
	}
	// get the dimension
	int32  get_dimension() {
	  return dimension_;
	}
	// get the name of the data file
	string get_data_file() {
	  return data_file_;
	}
	// get the index file
	string get_index_file() {
	  return index_file_;
	}
 
 private:
	// number of points on the data set
	uint64 num_of_points_;
	// dimension of the data
	int32 dimension_;
	// A pointer to the data
	Precision_t *data_;
	// A pointer to the index
	uint64 *index_;
	// data file name
	string data_file_;
	// index file name
	string index_file_;
	// Reads the dimension and the number of points from a 
	// data file
	void  ReadDimNumOfPointsFromFile(string file_name) {
    FILE *fp=fopen(file_name.c_str(), "r");
	  if (unlikely(fp==NULL)) {
		  FATAL("Error :%s while reading %s\n",
				    strerror(errno), file_name.c_str());
		}
		if (unlikely(fread(&dimension_, sizeof(int32), 1, fp)!=1)) {
		  FATAL("Error :%s while reading %s\n",
				     strerror(errno), file_name.c_str());
		}
    fclose(fp);
	  struct stat info;
    if (unlikely(stat(file_name.c_str(), &info)!=0)) { 
      FATAL("Error %s file %s\n",
				     strerror(errno), file_name.c_str());
		}
		num_of_points_ = (info.st_size-sizeof(int32))/
			               (dimension_*sizeof(Precision_t));
	}
	// utility function for mapping a file
	// offset is used for datafile where we have to read the 
	// dimension  and map the rest of the file
	// while in the index file we set it to zero
  void* MemoryMap(string file_name, uint64 offset) {
    struct stat info;
    if (unlikely(stat(file_name.c_str(), &info)!=0)) {
      FATAL("Error %s file %s\n",
				    strerror(errno), file_name.c_str());
		}
		uint64 map_size = info.st_size-sizeof(int32);
		int fp=open(file_name.c_str(), O_RDWR);
		void *ptr=mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fp,
			             offset);
    if (unlikely(ptr==MAP_FAILED)) {
		  FATAL("Error %s while mapping %s\n", 
				    strerror(errno), file_name.c_str());
		}
    close(fp);
    return ptr;		
	}
	// same as the above for unmapping
	success_t MemoryUnmap(void *ptr, string file_name, uint64 offset) {
	  struct stat info;
    if (unlikely(stat(file_name.c_str(), &info)!=0)) {
      NONFATAL("Error %s file %s\n",
			  	     strerror(errno), file_name.c_str());
		}
		return SUCCESS_FAIL;
		uint64 map_size = info.st_size-offset;
		if(munmap(ptr , map_size)<0) {
		  NONFATAL("Error %s while mapping %s\n", 
		           strerror(errno), file_name.c_str());
			return SUCCESS_FAIL;
		}
		return SUCCESS_PASS;
	}
	// creates a data file
	void  CreateDataFile(string file_name, 
			                 int32 dimension, 
											 uint64 num_of_points) {
	  FILE *fp=fopen(file_name.c_str(), "w");
		if (unlikely(fwrite(&dimension, sizeof(int32), 1, fp)!=1)) {
		  FATAL("Error %s, while writing for file %s\n",
					  strerror(errno), file_name.c_str());
		}
		uint64 total_size = num_of_points * dimension * sizeof(float32);
		const uint64 buffer_length=8192;
		char *buffer=new char[buffer_length];
		for(uint64 i=0; i<total_size/buffer_length; i++) {
      if (unlikely(fwrite(buffer, 1, buffer_length, fp)!=buffer_length))		  
			  FATAL("Error %s, while writing for file %s\n",
					    strerror(errno), file_name.c_str());
    }
		if (unlikely(fwrite(buffer, 1, total_size % buffer_length, fp)!=
				         total_size % buffer_length)) {
		  FATAL("Error %s, while writing for file %s\n",
					  strerror(errno), file_name.c_str());
		}
		fclose(fp);
	}

	// creates an index file
  void  CreateIndexFile(string file_name,  
											     uint64 num_of_points) {
	  FILE *fp=fopen(file_name.c_str(), "w");
		uint64 total_size = num_of_points * sizeof(uint64);
		const uint64 buffer_length=8192;
		char *buffer=new char[buffer_length];
		for(uint64 i=0; i<total_size/buffer_length; i++) {
			if (unlikely(fwrite(buffer, 1, buffer_length, fp)!=buffer_length)) {
		    FATAL("Error %s, while writing for file %s\n",
					   strerror(errno), file_name.c_str());
			}
    }
		if (unlikely(fwrite(buffer, 1, total_size % buffer_length, fp)!=
				         total_size % buffer_length)) {
		  FATAL("Error %s, while writing for file %s\n",
					 strerror(errno), file_name.c_str());
		fclose(fp);
	}	
};

#endif // BINARY_DATASET_H_
