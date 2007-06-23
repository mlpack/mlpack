/*
 * =====================================================================================
 * 
 *       Filename:  sparse_matrix.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  06/20/2007 02:40:16 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef SPARSE_MATRIX_H_
#define SPARSE_MATRIX_H_

#include <errno.h>
#include <algorithm>
#include <string>
#include <sys/mman.h>
#include "fastlib/fastlib.h"

using namespace std;

namespace sparse {
template<typename T>
class Matrix {
 public:
	typedef pair<index_t, T> NonZeroElement_t;
	template<typename > friend class Matrix;
	Matrix(){
	  allocation_flag_=false;
	}
	~Matrix(){};
	void Init(index_t dimension, 
			      int32 max_non_zero) {
		dimension_=dimension;
		max_non_zero_=max_non_zero;
		index_t alloc_size = dimension_ * 
			                   max_non_zero_ * sizeof(NonZeroElement_t);
		ptr_rows_ = (NonZeroElement_t *)mmap(NULL, alloc_size, 
			          PROT_READ | PROT_WRITE, 
								MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	  if (ptr_rows_==MAP_FAILED) {
	    FATAL("Couldn't allocate memory for the ptr_rows_, error: %s\n",
			  	  strerror(errno));
	  }
    allocation_flag_=true;
		row_elements_=(int32 *)mmap(NULL, dimension_*sizeof(int32) , 
			             PROT_READ | PROT_WRITE, 
								   MAP_SHARED | MAP_ANONYMOUS, -1, 0);

		Fill();
	}
	void Init(NonZeroElement_t *ptr_rows,
			      index_t dimension,
			      int32 max_non_zero) {
	  ptr_rows_=ptr_rows;
		dimension_=dimension;
		max_non_zero_=max_non_zero;
    row_elements_=(int32 *)mmap(NULL, dimension_*sizeof(int32) , 
			             PROT_READ | PROT_WRITE, 
								   MAP_SHARED | MAP_ANONYMOUS, -1, 0);

		Fill();
	}
	void Init(string file) {
	  FILE *fp=fopen(file.c_str(), "r");
		if (fp==NULL) {
		  FATAL("Error %s while trying to open %s\n", 
					  strerror(errno), file.c_str());
		}
		fscanf(fp, "%i %i\n", dimension_, max_non_zero_);
    index_t alloc_size = dimension_ * 
			                   max_non_zero_ * sizeof(NonZeroElement_t);
		ptr_rows_ = (NonZeroElement_t *)mmap(NULL, alloc_size, 
			          PROT_READ | PROT_WRITE, 
								MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	  if (ptr_rows_==MAP_FAILED) {
	    FATAL("Couldn't allocate memory for the ptr_rows_, error: %s\n",
			  	  strerror(errno));
	  }
    allocation_flag_=true;
		row_elements_=(int32 *)mmap(NULL, dimension_*sizeof(int32) , 
			             PROT_READ | PROT_WRITE, 
								   MAP_SHARED | MAP_ANONYMOUS, -1, 0);

		Fill();
		long long i,j;
		double val;
		while (feof(fp)==false) {
		  fscanf(fp,"%lli %lli %lg\n", &i, &j, &val);
		  this->set(i, j, val);
		}
		fclose(fp);
	}
	void Destruct() {
	  if (allocation_flag_==true) {
			index_t alloc_size = dimension_ * 
			                   max_non_zero_ * sizeof(NonZeroElement_t);
		  munmap(ptr_rows_, alloc_size);
		}
		munmap(row_elements_, dimension_*sizeof(int32));
  }
  
	inline T get(index_t  i, index_t j) {
		index_t row = i*max_non_zero_;
		DEBUG_ASSERT_MSG(i<dimension_ && i>=0,
				            "Tried to access row %lli > %lli\n",
			              (signed long long)i,
										(signed long long)dimension_);
		for(int32 k=0; k<row_elements_[i]; k++) {
			if (ptr_rows_[row+k].first ==j) {
			  return ptr_rows_[row+k].second;
			} 
		}
    FATAL("Tried to access %lli, %lli "
					"element which is zero", 
					(signed long long)i,
					(signed long long)j);		
	}

	inline void set(index_t  i, index_t j, T value) {
		index_t row = i*max_non_zero_;
		DEBUG_ASSERT_MSG(i<dimension_ && i>=0,
				            "Tried to access row %lli > %lli\n",
			              (signed long long)i,
										(signed long long)dimension_);
    DEBUG_ASSERT_MSG(row_elements_[i]<max_non_zero_,
	      "The matrix is overfull, increase sparsity\n");
    int32 k=row_elements_[i];
		row_elements_[i]++;
		ptr_rows_[row+k].first=j;
		ptr_rows_[row+k].second=value;
	}			
	void Advise() {
    if (madvise(ptr_rows_, 
				    dimension_*max_non_zero_*sizeof(NonZeroElement_t),
						MADV_SEQUENTIAL)==-1) {
		  NONFATAL("Advising failed error %s\n", strerror(errno));
		}
	}
	void UnAdvise() {
	  if (madvise(ptr_rows_, 
				    dimension_*max_non_zero_*sizeof(NonZeroElement_t),
						MADV_NORMAL)==-1) {
		  NONFATAL("Advising failed error %s\n", strerror(errno));
		}
	}
  void Multiply(T* vec, T* result) {
	  for(index_t i=0; i<dimension_; i++) {
		  result[i]=0;
			index_t row=i*max_non_zero_;
			for(index_t k=0; k<row_elements_[i]; k++) {
			  result[i]+=ptr_rows_[row+k].second * 
					         vec[ptr_rows_[row+k].first];
			}
		}
	}
  void MakeFast() {
	  for(index_t i=0; i<dimension_; i++) {
			std::sort(ptr_rows_+i*(max_non_zero_), 
					      ptr_rows_+i*(max_non_zero_)+row_elements_[i]);
		}
	}
 
  inline T FastGet(index_t i, index_t j) {
	  index_t ind = BinarySearch(i*max_non_zero_,
															 i*max_non_zero_+row_elements_[i],
															 j);
		FATAL("You are trying to access %lli , %lli "
				  "which is nonzero\n",
				  (signed long long)i,
				  (signed long long)j);
		return ptr_rows_[ind].second;
	}

	inline bool IsZero(index_t i, index_t j) {
	  index_t ind = BinarySearch(i*max_non_zero_,
															 i*max_non_zero_+row_elements_[i]-1,
															 j);
		return ind==-1;
	}

	index_t get_dimension() {
	  return dimension_;
	}
	index_t get_non_zeros_elements() {
	  index_t total=0;
		for(index_t i=0; i<dimension_; i++) {
		  total+=row_elements_[i];
		}
		return total;
	}
 private:
  NonZeroElement_t *ptr_rows_;
	index_t dimension_;
	int32 max_non_zero_;
	bool allocation_flag_;
	int32 *row_elements_;
	void Fill() {
	  for(index_t i=0; i<dimension_; i++) {
			for(int32 j=0; j<max_non_zero_; j++){
		    ptr_rows_[i*max_non_zero_+j].first=-1;
			}
		  row_elements_[i]=0;
		}
	}
	inline index_t BinarySearch(index_t low, 
			                 index_t high, 
											 index_t value) {
	  while(low <= high) {
		  index_t mid=(low+high)/2;
			if (ptr_rows_[mid].first > value) {
			  high=mid-1;
			} else {
			  if (ptr_rows_[mid].first < value) {
				  low=mid+1;
				} else {
			    return mid;
			  }
		  }
		}
		return -1;
	}
};

// solves the problem A*x=b
// where A is sparse semipositive definite 
template<typename T> 
void ConjugateGradient(Matrix<T> &A, T *b, T* x, T tolerance);
template<typename T>
inline T VectorDotProduct(T* a, T* b, index_t size) {
  T result=0;
	for(index_t i=0; i<size; i++) {
	  result+=a[i]*b[i];
	}
	return result;
}
template<typename T>
inline void VectorPlus(T* a, T* b, index_t size, T* c) {
  for(index_t i=0; i<size; i++) {
	  c[i]=a[i]+b[i];
	}
}
template<typename T>
inline void VectorPlusTimes(T* a, T scalar, T* b, index_t size, T* c) {
  for(index_t i=0; i<size; i++) {
	  c[i]=a[i]+scalar*b[i];
	}
}
template<typename T>
inline void VectorMinusTimes(T* a, T scalar, T* b, index_t size, T* c) {
  for(index_t i=0; i<size; i++) {
	  c[i]=a[i]-scalar*b[i];
	}
}

template<typename T>
inline void VectorMinus(T* a, T* b, index_t size, T* c) {
  for(index_t i=0; i<size; i++) {
	  c[i]=a[i]-b[i];
	}
}
template<typename T>
inline void VectorMultiplyScalar(T* vector_in, T scalar, index_t size, 
		                      T* vector_out) {
  for(index_t i=0; i<size; i++) {
	  vector_out[i]=vector_in[i]*scalar;
	}
}
template<typename T>
inline T *NewVector(index_t size) {
  T* out = (T*)mmap(NULL, size*sizeof(T), 
	         PROT_READ | PROT_WRITE, 
				   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (madvise(out, size*sizeof(T),
						MADV_SEQUENTIAL)==-1) {
	  NONFATAL("Advising failed error %s\n", strerror(errno));
	}
  return out;
}

template<typename T>
inline void DeleteVector(T* vector, index_t size) {
  if (munmap(vector, size*sizeof(T))<0) {
	  NONFATAL("Failed to unmap memory, error:%s\n", strerror(errno));
	}
}

template<typename T>
T *ReadVectorFromFile(string filename) {
	FILE *fp;
	fp=fopen(filename.c_str(), "r");
  if (fp==NULL) {
	  FATAL("Error %s while trying to open %s\n", strerror(errno), 
				   filename.c_str());
	}
	long long size;
	fscanf(fp, "%lli\n", &size);
  T *ptr = NewVector<T>(size);
	index_t i=0;
	while (!feof(fp)) {
		double value;
	  fscanf(fp, "%lg", &value);
		ptr[i]=(T)value;
		i++;
	}
  fclose(fp);
	return ptr;
}
};

#include "u/nvasil/sparse/conjugate_gradient_impl.h"
#endif // SPARSE_MATRIX_H_
