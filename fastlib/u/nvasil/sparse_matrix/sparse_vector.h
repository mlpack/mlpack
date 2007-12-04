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
#include "fastlib/la/vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"

class SparseVector {
 public:
	SparseVector() {
		Init()
	}
	SparseVector(std::vector<index_t> &indices, 
			         Vector &values, 
							 index_t dimension);
	SparseVector(std::map<index_t, double> data, 
			         index_t dimension);
	SparseVector(index_t estimated_non_zero_elements, 
			         index_t dimension);
	SparseVector(Epetra_CrsMatrix *one_dim_matrix, 
			         index_t dimension);
	SparceVector(const SparseVector &other);
	~SparseVector() {
	  Destruct();
	}
	void Init();
	void Init(std::vector<index_t> &indices, Vector &values, 
			      index_t dimension);
	void Init(std::map<index_t, double> data, index_t dimension);
	void Init(index_t estimated_non_zero_elements, index_t dimension);
	void Init(Epetra_CrsMatrix *one_dim_matrix, index_t dimension);
	void Copy(const SparseVector &other)
	void Destruct();
  void MakeSubvector(index_t start_index, 
			               index_t len, 
										 SparseVector* dest)
  double get(index_t i);
	void   set(index_t i, double  value);
		
 private:
	Epetra_CrsMatrix *vector_;
	Epetra_SerialComm comm_;
	Epetra_Map map_;
  index_t *myglobal_elements_;
	index_t dimension_;
	index_t start_;
	index_t end_;
	bool own_;

};

namespace sparse {
	inline void AddVectors(SparseVector &v1, SparseVector &v2, SparseVector *sum);
	inline void Subtract(SparseVector &v1, SparseVector &v2, SparseVector *diff);
	inline void PointProduct(SparseVector &v1, SparseVector &v2, SparseVector *point_prod);
	inline void DotProduct(SparseVector &v1, SparseVector &v2, double *dot_product);
  inline void DistanceSqEuclidean(SparseVector &v1, SparseVector &v2, double *dist);
  template<int t_pow>
  inline void RawLMetric(SparseVector &v1, SparseVector &v2, double *dist);
  template<int t_pow>
  inline void RawLMetric(SparseVector &v1, SparseVector &v2, double *dist);
};

#include "sparse_vector_impl.h"
#endif
