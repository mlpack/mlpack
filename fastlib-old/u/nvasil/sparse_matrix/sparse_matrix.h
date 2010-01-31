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
#ifndef USE_TRILINOS
#define USE_TRILINOS
#endif
#include <stdio.h>
#include <errno.h>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <algorithm>
#include "fastlib/fastlib.h"
#include "la/matrix.h"
// you need this because trillinos redifines it. It's ok
// if you don't have it, but you will get an annoying warning
#ifdef F77_FUNC
#undef F77_FUNC
#endif
#include "trilinos/include/Epetra_CrsMatrix.h"
#include "trilinos/include/Epetra_SerialComm.h"
#include "trilinos/include/Epetra_Map.h"
#include "trilinos/include/Epetra_Vector.h"
#include "trilinos/include/Epetra_MultiVector.h"
#include "trilinos/include/AnasaziBasicEigenproblem.hpp"
#include "trilinos/include/AnasaziEpetraAdapter.hpp"
#include "trilinos/include/AnasaziBlockKrylovSchurSolMgr.hpp"
#include "trilinos/include/AztecOO.h"

/* class SparseMatrix created by Nick
 * This is a sparse matrix wrapper for trilinos Epetra_CrsMatrix
 * It is much simpler than Epetra_CrsMatrix. At this time 
 * it supports eigenvalues (Krylov method) and linear system solution
 * I have added matrix addition/subtraction multiplication 
 * I am also trying to add the submatrices
 * Note: There is a restriction on these matrices, the number of rows is
 * always greater or equal to the number of columns. The number of rows
 * is also called dimension. We pose this restriction because trilinos supports
 * square matrices only. In sparse matrices though this is not the problem since
 * an mxn matrix where  m>n can is equivalent to an mxm matrix where all the
 * elements with n<j<m are zero
 */

class Sparsem;
class SparseMatrix {
 public:
	 friend class Sparsem;
 // Some typedefs for oft-used data types
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<double, Epetra_MultiVector> MVT;

  SparseMatrix() ;
	// Constructor 
	// num_of_rows: number of rows
	// num_of_cols: number of columns
	// nnz_per_row: an estimate of the non zero elements per row
	//              This doesn't need to be accurate. If you need
	//              more it will automatically resize. Try to be as accurate
	//              as you can because resizing costs. It is better if your
	//              estimete if greater than the true non zero elements. So
	//              it is better to overestimate than underestimate
	SparseMatrix(const index_t num_of_rows, 
			         const index_t num_of_cols, 
			         const index_t nnz_per_row);
	// Copy constructor
  SparseMatrix(const SparseMatrix &other);
	SparseMatrix(std::string textfile) {
	  Init(textfile);
	}
	~SparseMatrix()  {
	  Destruct();
	}
	void Destruct();
	// Use this initializer like the Constructor
  void Init(const index_t num_of_rows, 
			      const index_t num_of_columns, 
			      const index_t nnz_per_row);
	// This Initializer is like the previous one with the main difference that
	// for every row we give a seperate estimate for the non-zero elements.
  void Init(index_t num_of_rows, index_t num_of_columns, index_t *nnz_per_row);
	// This Initializer fills the sparse matrix with data.
	// row_indices: row indices for non-zero elements
	// col_indices: column indices for non-zero elements
	// values     : values of non-zeros elements
	// If the dimension (number of rows)and the expected (nnz elements per row)
	// are set to a negative value, the function will automatically detect it 
	void Init(const std::vector<index_t> &row_indices,
		       	const std::vector<index_t> &col_indices,
			      const Vector &values, 
			      index_t nnz_per_row, 
			      index_t dimension);
	// The same as above but we use STL vector for values
  void Init(const std::vector<index_t> &row_indices,
		       	const std::vector<index_t> &col_indices,
			      const std::vector<double> &values, 
			      index_t nnz_per_row, 
			      index_t dimension);

	// Initialize from a text file in the following format
	// row column value \n
	void Init(std::string textfile);
	// Initialize the diagonal 
	void InitDiagonal(const Vector &vec); 
	// Initialize the diagonal with a constant
	void InitDiagonal(const double value);
	// It is recomended that you load the matrix row-wise, Before
	// you do that call StartLoadingRows()
	void StartLoadingRows();
	// All these functions load Rows, with the data in different format
  void LoadRow(index_t row, std::vector<index_t> &columns, Vector &values);
  void LoadRow(index_t row, index_t *columns, Vector &values);
  void LoadRow(index_t row, index_t num, index_t *columns, double  *values);
  void LoadRow(index_t row, std::vector<index_t> &columns, std::vector<double>  &values);
	// When you are done call this it does some optimization in the storage, no
	// further asignment 
	void EndLoading();
	// It makes the matrix symmetric. It scans the rows of the matrix and for every (i,j)
	// element (j,i) equal to (j,i)
  void MakeSymmetric();	
	// if you know that the matrix is symmetric set the flag
	void set_symmetric(bool val) {
	  issymmetric_ = val;
	}
	// Not implemented yet
  void SetDiagonal(const Vector &vector); 
	// Copy function, used also by copy constructor
  void Copy(const SparseMatrix &other);
	// Not implemented yet
  void Alias(const Matrix& other);
	// Not Implemented yet
  void SwapValues(SparseMatrix* other);
  // Access values, It will fail if EndLoading() has been called
	double get(index_t r, index_t c) const;
	// Set Values
  void   set(index_t r, index_t c, double v);
	// For debug purposes you can call it to print the matrix
	std::string Print() {
		std::ostringstream s1;
		matrix_->Print(s1);
		return s1.str();
	}
	// Get the number of rows
  index_t get_num_of_rows() {
		return num_of_rows_;
	}
	// Get the number of columns
  index_t get_num_of_columns() {
	  return num_of_columns_;
	}
	// Dimension should be equal to the number of rows
	index_t get_dimension() {
	  return dimension_;
	}
	// The number of non zero elements
	index_t get_nnz() {
	 return matrix_->NumGlobalNonzeros();
  }
	// Computes the eignvalues with the Krylov Method
	void Eig(index_t num_of_eigvalues, // number of eigenvalues to compute
			     std::string eigtype, // Choose which eigenvalues to compute
                                // Choices are:
                                // LM - target the largest magnitude  
                                // SM - target the smallest magnitude 
                                // LR - target the largest real 
                                // SR - target the smallest real 
                                // LI - target the largest imaginary
                                // SI - target the smallest imaginary
		       Matrix *eigvectors,  // The eigenvectors computed must not be initialized
					 std::vector<double> *real_eigvalues, // real part of the eigenvalues
					                                      // must be initialized, but should not
																								// allocate space. The eigenvalues
																								// returned might actually be less 
																								// than the ones requested
																								// for example when the matrix has 
																								// rank n< eigenvalues requested
					 std::vector<double> *imag_eigvalues  // imaginary part of the eigenvalues
					                                      // must be initialized. If the
																								// problem is symmetric there is 
																								// no need to initialize. 
																								// The same as real_eigvalues
																								// hold for the space allocated
																								// in the non-symmetric case    
					 );
  // Linear System solution, Call Endloading First.
	void LinSolve(Vector &b, // must be initialized (space allocated)
			          Vector *x, // must be initialized (space allocated)
								double tolerance,
								index_t iterations
						) {
		if (matrix_->Filled()==false && matrix_->StorageOptimized()==false){
		  FATAL("You should call EndLoading() first\n");
		}
	  Epetra_Vector tempb(View, *map_, b.ptr());
    Epetra_Vector tempx(View, *map_, x->ptr());
    // create linear problem
 	  Epetra_LinearProblem problem(matrix_.get(), &tempx, &tempb);
 		// create the AztecOO instance
		AztecOO solver(problem);
		solver.SetAztecOption( AZ_precond, AZ_Jacobi);
		solver.Iterate(iterations, tolerance);
		NONFATAL("Solver performed %i iterations, true residual %lg",
				     solver.NumIters(), solver.TrueResidual());
	}

	// Use this for the general case
  void LinSolve(Vector &b, Vector *x) {
	  LinSolve(b, x, 1E-9, 1000);
	}
  	
	// scales the matrix with a scalar;
	void Scale(double scalar) {
	  matrix_->Scale(scalar);
	}
	// The matrix will be scaled such that A(i,j) = x(j)*A(i,j) 
	// where i denotes the global row number of A and j denotes the  column number 
	void ColumnScale(const Vector &vec) {
    Epetra_Vector temp(View, *map_, (double*)vec.ptr());
    matrix_->RightScale(temp);
	}
	// The  matrix will be scaled such that A(i,j) = x(i)*A(i,j) 
	// where i denotes the row number of A and j denotes the column number of A.
	void RowScale(const Vector &vec) {
    Epetra_Vector temp(View, *map_, (double *)vec.ptr());
    matrix_->LeftScale(temp);
	}
	// computes the L1 norm
	double L1Norm() {
	  return matrix_->NormOne();
	}
	// L infinity norm
	double LInfNorm() {
		return matrix_->NormInf();
	}
	// Computes the inverse of the sum of absolute values of the rows 
	// of the matrix 
	void InvRowsSums(Vector *result) {
	  Epetra_Vector temp(View, *map_, result->ptr());
		matrix_->InvRowSums(temp);
	}
	// Computes the inv of max of absolute values of the rows of the matrix, 
	void InvRowMaxs(Vector *result) {
	  Epetra_Vector temp(View, *map_, result->ptr());
    matrix_->InvRowMaxs(temp);
	} 
	// Computes the inverse of the sum of absolute values of the columns of the
	// matrix
	void InvColSums(Vector *result) {
    Epetra_Vector temp(View, *map_, result->ptr());
    matrix_->InvColSums(temp);
	}
  // Computes the inv of max of absolute values of the columns of the matrix, 
	void InvColMaxs(Vector *result) {
	  Epetra_Vector temp(View, *map_, result->ptr());
    matrix_->InvColMaxs(temp);
	}

 private:
	index_t dimension_;
	index_t num_of_rows_;
	index_t num_of_columns_;
	Epetra_SerialComm comm_;
	bool issymmetric_;
	Epetra_Map *map_;
  Teuchos::RCP<Epetra_CrsMatrix> matrix_;
  index_t *my_global_elements_;
  void Load(const std::vector<index_t> &rows, 
			      const std::vector<index_t> &columns, 
						const Vector &values);
	void AllRowsLoad(Vector &rows, Vector &columns);

};

class Sparsem {
 public:
  static inline void Add(const SparseMatrix &a, 
			                   const SparseMatrix &b, 
												 SparseMatrix *result) {
		DEBUG_ASSERT(a.num_of_rows_==b.num_of_rows_);
		DEBUG_ASSERT(a.num_of_columns_==b.num_of_columns_);
    DEBUG_ASSERT(a.num_of_rows_==result->num_of_rows_);
		DEBUG_ASSERT(a.num_of_columns_==result->num_of_columns_);
		result->StartLoadingRows();
    for(index_t r=0; r<a.num_of_rows_; r++) {
		  index_t num1, num2;
		  double *values1, *values2;
		  index_t *indices1, *indices2;
		  a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r], num1, values1, indices1);
  	  b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r], num2, values2, indices2);
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
			  indices3.insert(indices3.end(), indices2+j, indices2+num2);
		  }
      result->LoadRow(r, indices3, values3);
		}
	}
	static inline void Subtract(const SparseMatrix &a,
			                        const SparseMatrix &b,
														  SparseMatrix *result) {
    DEBUG_ASSERT(a.num_of_rows_==b.num_of_rows_);
		DEBUG_ASSERT(a.num_of_columns_==b.num_of_columns_);
		DEBUG_ASSERT(a.num_of_rows_==result->num_of_rows_);
		DEBUG_ASSERT(a.num_of_columns_==result->num_of_columns_);
		// If you try assigning the results to an already initialized matrix
		// you might get unexpected results. The following assertions
		// prevent you partially from that 
		DEBUG_ASSERT(&a != result); 
		DEBUG_ASSERT(&b != result);
		result->StartLoadingRows();
    for(index_t r=0; r<a.num_of_rows_; r++) {
		  index_t num1, num2;
		  double *values1, *values2;
		  index_t *indices1, *indices2;
		  a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r], num1, values1, indices1);
  	  b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r], num2, values2, indices2);
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
			  if (likely(i<num1) && indices1[i] == indices2[j]) {
					double diff=values1[i] - values2[j];
					if (diff!=0) {
			      values3.push_back(diff);
				    indices3.push_back(indices1[i]);
					}
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
				for(index_t k=j; k<num2; k++) {
		      values3.push_back(-values2[k]);
			    indices3.push_back(indices2[k]);
				}
		  }
      result->LoadRow(r, indices3, values3);
		}
	}
	/* Multiplication of two matrices A*B in matlab notation 
	 * If B is symmetric then it is much faster, because we can
	 * multiply rows. Otherwise we have to compute the transpose
	 * As an advise multiplication of two sparse matrices might 
	 * lead to a dense one, so please be carefull
	 */
	static inline void Multiply(const SparseMatrix &a,
			                        const SparseMatrix &b,
															SparseMatrix *result) {
	  DEBUG_ASSERT(a.num_of_columns_ == b.num_of_rows_);
	  DEBUG_ASSERT(a.num_of_rows_ == result->num_of_rows_);
	  DEBUG_ASSERT(b.num_of_columns_ == result->num_of_columns_);
	  // If you try assigning the results to an already initialized matrix
		// you might get unexpected results. The following assertions
		// prevent you partially from that 
    DEBUG_ASSERT(&a != result); 
		DEBUG_ASSERT(&b != result);

	  if (b.issymmetric_ == true) {
	    for(index_t r1=0; r1<a.num_of_rows_; r1++) {
				std::vector<index_t> indices3;
				std::vector<double>  values3;
				index_t num1;
				double  *values1;
				index_t *indices1;
        a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r1],
						                            num1, values1, indices1);
		    for(index_t r2=0; r2<b.num_of_rows_; r2++) {
			    index_t num2;
		      double  *values2;
					index_t *indices2;
		     	b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r2], 
							                            num2, values2, indices2);
		      index_t i=0;
		      index_t j=0;
		      double dot_product=0;
		      while (likely(i<num1 && j<num2)) {
			      while (indices1[i] < indices2[j]) {
			        i++;	
              if unlikely((i>=num1)) {
				        break;
				      }
		        }
			      if (likely(i<num1) && indices1[i] == indices2[j]) {
			        dot_product += values1[i] * values2[j];
			      } 
			      j++;
		      }
					if (dot_product!=0) {
					  indices3.push_back(r2);
						values3.push_back(dot_product);
					}
				}
				result->LoadRow(r1, indices3, values3);
			  indices3.clear();
				values3.clear();
		  }
	  } else {
	    for(index_t r1=0; r1<a.num_of_rows_; r1++) {
        index_t num1;
				double  *values1;
				index_t *indices1;
        a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r1],
						                            num1, values1, indices1);
				double dot_product=0;
				std::vector<index_t> indices3;
				std::vector<double>  values3;
				for(index_t r2=0; r2<b.num_of_columns_; r2++) {
					for(index_t k=0; k< num1; k++) {
				    dot_product += values1[k]*b.get(indices1[k], r2);
					}
					if (dot_product!=0){
					  indices3.push_back(r2);
						values3.push_back(dot_product);
					}
					dot_product=0;
				}
				if (!indices3.empty()) {
				  result->LoadRow(r1, indices3, values3);
				}
        indices3.clear();
				values3.clear();
			}
		}
	}
	/* The transpose flag should be set to true if 
	 * we want to use the transpose of mat, otherwise
	 * set it to false.
	 * */
	static inline void Multiply(const SparseMatrix &mat,
			                        const Vector &vec,
															Vector *result,
															bool transpose_flag) {
   Epetra_Vector temp_in(View, *(mat.map_), (double *)vec.ptr());
	 Epetra_Vector temp_out(View, *(mat.map_), (double *)result->ptr());
	 mat.matrix_->Multiply(transpose_flag, temp_in, temp_out);
	}
	/* Multiply the matrix with a scalar
	 */
	static inline void Multiply(const SparseMatrix &mat,
			                        const double scalar,
															SparseMatrix *result) {
	  result->Copy(mat);
		result->Scale(scalar);
	}
	/* element wise multiplication of the matrices
	 * A.*B in matlab notation
	 */
	static inline void DotMultiply(const SparseMatrix &a,
		                             const SparseMatrix &b,
		                             SparseMatrix *result) {
	  DEBUG_ASSERT(a.num_of_columns_ == b.num_of_rows_);
	  DEBUG_ASSERT(a.num_of_rows_ == result->num_of_rows_);
	  DEBUG_ASSERT(b.num_of_columns_ == result->num_of_columns_);
	  // If you try assigning the results to an already initialized matrix
		// you might get unexpected results. The following assertions
		// prevent you partially from that 
    DEBUG_ASSERT(&a != result); 
		DEBUG_ASSERT(&b != result);
		for(index_t r=0; r<a.num_of_rows_; r++) {
	    std::vector<index_t> indices3;
			std::vector<double>	 values3;
			indices3.clear();
			values3.clear();
			index_t num1, num2;
			double *values1, *values2;
			index_t *indices1, *indices2;
		  a.matrix_->ExtractGlobalRowView(a.my_global_elements_[r], num1, values1, indices1);
  	  b.matrix_->ExtractGlobalRowView(b.my_global_elements_[r], num2, values2, indices2);
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
		  result->LoadRow(r, indices3, values3);
	  }
	}
}; 
#include "u/nvasil/sparse_matrix/sparse_matrix_impl.h"
#endif
