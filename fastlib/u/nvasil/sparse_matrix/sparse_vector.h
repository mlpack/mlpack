/*
 * =====================================================================================
 * 
 *       Filename:  sparse_vector.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  12/03/2007 12:44:34 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef SPARSE_VECTOR_H_
#define SPARSE_VECTOR_H_
#ifndef HAVE_CONFIG_H
#define HAVE_CONFIG_H
#endif
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include "fastlib/fastlib.h"
#include "la/matrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"

class SparseVectorTest;
class sparse;

class SparseVector {
 public:
	friend class SparseVectorTest;
	friend class sparse;
	SparseVector() {
	}
	SparseVector(std::vector<index_t> &indices, 
			         Vector &values, 
							 index_t dimension);
	SparseVector(std::map<index_t, double> &data, 
			         index_t dimension);
	SparseVector(index_t estimated_non_zero_elements, 
			         index_t dimension);
	SparseVector(Epetra_CrsMatrix *one_dim_matrix, 
			         index_t dimension);
	void Destruct();
	SparseVector(const SparseVector &other);
	~SparseVector() {
	  Destruct();
	}
	void Init();
	void Init(std::vector<index_t> &indices, Vector &values, 
			      index_t dimension);
	void Init(std::vector<index_t> &indices, std::vector<double> &values, 
			      index_t dimension);
	void Init(index_t *indices, double *values, index_t len, index_t dimension);
	void Init(std::map<index_t, double> data, index_t dimension);
	void Init(index_t estimated_non_zero_elements, index_t dimension);
	void Init(Epetra_CrsMatrix *one_dim_matrix, index_t dimension);
	void Copy(const SparseVector &other);
  void MakeSubvector(index_t start_index, 
			               index_t len, 
										 SparseVector* dest);
  double get(index_t i);
	void   set(index_t i, double  value);
	void   set_start(index_t i);
	void   set_end(index_t i);
	void   Lock();
	  
		
 private:
	Epetra_CrsMatrix *vector_;
	Epetra_SerialComm comm_;
	Epetra_Map *map_;
  index_t *myglobal_elements_;
	index_t dimension_;
	index_t start_;
	index_t end_;
	bool own_;

};

class sparse {
 public:
  static inline void AddVectors(SparseVector &v1, SparseVector &v2, SparseVector* sum) {
	  if (unlikely(v1.dimension_ != v2.dimension_)) {
		  FATAL("Sparse Vectors have different dimensions %i != %i", v1.dimension_, v2.dimension_);
		}
		index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		v1.vector_->ExtractGlobalRowView(0, num1, values1, indices1);
  	v2.vector_->ExtractGlobalRowView(0, num2, values2, indices2);
		std::vector<double>  values3;
		std::vector<index_t> indices3;
		index_t i=0;
		index_t j=0;
		while (likely(i<num1 && j<num2)) {
			while (indices1[i] < indices2[j]) {
			 	values3.push_back(values1[i]);
				indices3.push_back(indices1[i]);
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if ( likely(i<num1) && indices1[i] == indices2[j]) {
			  values3.push_back(values1[i] + values2[j]);
				indices3.push_back(indices1[i]);
			} else {
			  values3.push_back(values2[j]);
			  indices3.push_back(indices2[j]);	
			}
			j++;
		}
		if (i<num1) {
		  values3.insert(values3.end(), values1+i, values1+num1);
			indices3.insert(indices3.end(), indices1+i, indices1+num1);
		}
		if (j<num2) {
		  values3.insert(values3.end(), values2+j, values2+num2);
			indices3.insert(indices3.end(), indices2+i, indices2+num2);
		}
    sum->Init(indices3, values3, v1.dimension_);
	}
	
	static inline void Subtract(SparseVector &v1, SparseVector &v2, SparseVector *diff) {
	  if (unlikely(v1.dimension_ != v2.dimension_)) {
		  FATAL("Sparse Vectors have different dimensions %i != %i", v1.dimension_, v2.dimension_);
		}
		index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		v1.vector_->ExtractGlobalRowView(0, num1, values1, indices1);
  	v2.vector_->ExtractGlobalRowView(0, num2, values2, indices2);
		std::vector<double>  values3;
		std::vector<index_t> indices3;
		index_t i=0;
		index_t j=0;
		while (likely(i<num1 && j<num2)) {
			while (indices1[i] < indices2[j]) {
			 	values3.push_back(values1[i]);
				indices3.push_back(indices1[i]);
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if ( likely(i<num1) && indices1[i] == indices2[j]) {
			  values3.push_back(values1[i] - values2[j]);
				indices3.push_back(indices1[i]);
			} else {
			  values3.push_back(-values2[j]);
			  indices3.push_back(indices2[j]);	
			}
			j++;
		}
		if (i<num1) {
		  values3.insert(values3.end(), values1+i, values1+num1);
			indices3.insert(indices3.end(), indices1+i, indices1+num1);
		}
		if (j<num2) {
			for(index_t jj=i; jj<num2; jj++) {
		    values3.push_back(-values2[jj]);
			  indices3.push_back(indices2[jj]);
			}
		}
    diff->Init(indices3, values3, v1.dimension_);
	}
	
	static inline void PointProduct(SparseVector &v1, SparseVector &v2, SparseVector *point_prod) {
	  if (unlikely(v1.dimension_ != v2.dimension_)) {
		  FATAL("Sparse Vectors have different dimensions %i != %i", v1.dimension_, v2.dimension_);
		}
		index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		v1.vector_->ExtractGlobalRowView(0, num1, values1, indices1);
  	v2.vector_->ExtractGlobalRowView(0, num2, values2, indices2);
		std::vector<double>  values3;
		std::vector<index_t> indices3;
		index_t i=0;
		index_t j=0;
		while (likely(i<num1 && j<num2)) {
			while (indices1[i] < indices2[j]) {
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if ( likely(i<num1) && indices1[i] == indices2[j]) {
			  values3.push_back(values1[i] * values2[j]);
				indices3.push_back(indices1[i]);
			} 
			j++;
		}
		point_prod->Init(indices3, values3, v1.dimension_);
	}

	static inline void DotProduct(SparseVector &v1, SparseVector &v2, double *dot_product) {
	  if (unlikely(v1.dimension_ != v2.dimension_)) {
		  FATAL("Sparse Vectors have different dimensions %i != %i", v1.dimension_, v2.dimension_);
		}
		index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		v1.vector_->ExtractGlobalRowView(0, num1, values1, indices1);
  	v2.vector_->ExtractGlobalRowView(0, num2, values2, indices2);
		index_t i=0;
		index_t j=0;
		*dot_product=0;
		while (likely(i<num1 && j<num2)) {
			while (indices1[i] < indices2[j]) {
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if ( likely(i<num1) && indices1[i] == indices2[j]) {
			  *dot_product += values1[i] * values2[j];
			} 
			j++;
		}
	}
  
	static inline void DistanceSqEuclidean(SparseVector &v1, SparseVector &v2, double *dist) {
	  if (unlikely(v1.dimension_ != v2.dimension_)) {
		  FATAL("Sparse Vectors have different dimensions %i != %i", v1.dimension_, v2.dimension_);
		}
		index_t num1, num2;
		double *values1, *values2;
		index_t *indices1, *indices2;
		v1.vector_->ExtractGlobalRowView(0, num1, values1, indices1);
  	v2.vector_->ExtractGlobalRowView(0, num2, values2, indices2);
		std::vector<double>  values3;
		std::vector<index_t> indices3;
		index_t i=0;
		index_t j=0;
		*dist=0;
		while (likely(i<num1 && j<num2)) {
			while (indices1[i] < indices2[j]) {
			 	values3.push_back(values1[i]);
				indices3.push_back(indices1[i]);
			  i++;	
        if unlikely((i>=num1)) {
				  break;
				}
		  }
			if ( likely(i<num1) && indices1[i] == indices2[j]) {
			  *dist += (values1[i] - values2[j]) * (values1[i] - values2[j]);
			} else {
			  *dist += values2[j] * values2[j];
			}
			j++;
		}
		if (i<num1) {
			for(index_t ii=i; i<num1; i++) {
		    *dist += values1[ii] * values1[ii];
			}
		}
		if (j<num2) {
			for(index_t jj=j; jj<num2; jj++) {
		    *dist +=  values2[jj] * values2[jj];
			  
			}
		}
	}
  
	template<int t_pow>
  inline void RawLMetric(SparseVector &v1, SparseVector &v2, double *dist);
};

#include "sparse_vector_impl.h"
#endif
