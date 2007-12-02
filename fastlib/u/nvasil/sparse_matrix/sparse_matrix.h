/*
 * =====================================================================================
 * 
 *       Filename:  sparse_matrix.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  12/01/2007 04:12:00 PM EST
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

class SparseMatrix {
 public:
  SparseMatrix();
	SparseMatrix(index_t dimension);
  SparseMatrix(const SparseMatrix);
	~SparseMatrix();
	void Destruct();
  void Init(index_t dimension, index_t nnz_per_row);
  void Init(index_t dimension, index_t *nnz_per_row);
	void Init(const std::vector<index_t> &indices, const Vector values, 
			      index_t nnz_per_row, 
			      index_t dimension);
  template<typename T>
	void Init(const index_t *indices, const T *values, index_t num_of_elemets);
	void Init(std::string textfile);
	void InitDiagonal(const Vector &vec); 
	void StartLoading();
  void LoadRow();
  void EndLoading();
  void MakeSymetric();	
  void SetAll(double d);
  void SetZero();
  void SetDiagonal(const Vector &vector); 
  void Copy(const SparseMatrix &other);
  void Alias(const Matrix& other);
  void SwapValues(SparseMatrix* other);
  void CopyValues(const SparseMatrix& other);
  double get(index_t r, index_t c);
  void set(index_t r, index_t c, double v);
	void set_non_zero_to_non_zero(index_t r, index_t c, double v);
	void set_zero_to_non_zero(index_t r, index_t c, double v);
  void set_non_zero_to_zero(index_t r, index_t c);
  double &ref(index_t r, index_t c);
  index_t get_dimension();
	index_t get_nnz();
 
 private:
	index_t dimension_;
	Epetra_SerialComm comm_;
	Epetra_Map *map_;
  Epetra_CrsMatrix *matrix_;
  index_t *MyglobalElements_;
  void AllRowsLoad(Vector &rows, Vector &columns);
			

};

#include "u/nvasil/sparse_matrix/sparse_matrix_impl.h"
#endif
