/**
 * @file sparse_matrix.h
 * Wrappers on the trilinos sparse solver
 * It also has functionality for adding, subtracting and multiplying
 * sparse matrices
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
#include <sstream>
#include <algorithm>
#include "fastlib/fastlib.h"
#include "fastlib/la/matrix.h"
// you need this because trillinos redifines it. It's ok
// if you don't have it, but you will get an annoying warning
#ifdef F77_FUNC
#undef F77_FUNC
#endif
#include "trilinos/include/Tpetra_CrsMatrix.hpp"
#include "trilinos/include/Tpetra_SerialPlatform.hpp"
#include "trilinos/include/Tpetra_Map.hpp"
#include "trilinos/include/Tpetra_Vector.hpp"
#include "trilinos/include/Tpetra_MultiVector.hpp"
#include "trilinos/include/AnasaziBasicEigenproblem.hpp"
#include "trilinos/include/AnasaziEpetraAdapter.hpp"
#include "trilinos/include/AnasaziBlockKrylovSchurSolMgr.hpp"
#include "trilinos/include/AztecOO.hpp"
#include "trilinos/include/Ifpack_CrsIct.hpp"


namespace la {
/** class SparseMatrix created by Nick
 *  This is a sparse matrix wrapper for trilinos Epetra_CrsMatrix
 *  It is much simpler than Epetra_CrsMatrix. At this time 
 *  it supports eigenvalues (Krylov method) and linear system solution
 *  I have added matrix addition/subtraction multiplication 
 *  I am also trying to add the submatrices
 *  Note: There is a restriction on these matrices, the number of rows is
 *  always greater or equal to the number of columns. The number of rows
 *  is also called dimension. We pose this restriction because trilinos supports
 *  square matrices only. In sparse matrices though this is not the problem since
 *  an mxn matrix where  m>n can is equivalent to an mxm matrix where all the
 *  elements with n<j<m are zero
 */

template<typename IndexPrecision, ValuePrecision>
class SparseMatrix {
 public:
  // Some typedefs for oft-used data types
  typedef Tpetra::MultiVector<IndexPrecision, ValuePrecision> MV;
  typedef Tpetra::Operator OP;
  typedef Anasazi::MultiVecTraits<ValuePrecision, MV> MVT;
  typedef SparseMatrix<IndexPrecision, ValuePrecision> SparseMatrixT;
  SparseMatrix() ;
  /** 
   * Constructor 
   * num_of_rows: number of rows
   * num_of_cols: number of columns
   * nnz_per_row: an estimate of the non zero elements per row
   *               This doesn't need to be accurate. If you need
   *               more it will automatically resize. Try to be as accurate
   *               as you can because resizing costs. It is better if your
   *               estimete if greater than the true non zero elements. So
   *               it is better to overestimate than underestimate 
   */
  SparseMatrix(const IndexPrecision num_of_rows, 
               const IndexPrecision num_of_cols, 
               const IndexPrecision nnz_per_row);
  /** Copy constructor */
  SparseMatrix(const SparseMatrixT &other);
  SparseMatrix(std::string textfile); 
  ~SparseMatrix(); 
  /** Use this initializer like the Constructor */
  void Init(const IndexPrecision num_of_rows, 
            const IndexPrecision num_of_columns, 
            const IndexPrecision nnz_per_row);
  /** This Initializer is like the previous one with the main difference that
   * for every row we give a seperate estimate for the non-zero elements.
   */
  void Init(IndexPrecision num_of_rows, IndexPrecision num_of_columns, IndexPrecision *nnz_per_row);
  /** This Initializer fills the sparse matrix with data.
   *   row_indices: row indices for non-zero elements
   *   col_indices: column indices for non-zero elements
   *   values     : values of non-zeros elements
   *   If the dimension (number of rows)and the expected (nnz elements per row)
   *   are set to a negative value, the function will automatically detect it 
  */
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer>
  void Init(const IndexContainer<IndexPrecision> &row_indices,
            const IndexContainer<IndexPrecision> &col_indices,
            const ValueContainer<ValuePrecision>  &values, 
            IndexPrecision nnz_per_row, 
            IndexPrecision dimension);
  /** 
   * The same as above but we use STL vector for values
   */
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer>
  void Init(const IndexContainer<IndexPrecision> &row_indices,
            const IndexContainer<IndexPrecision> &col_indices,
            const ValueContainer<ValuePrecision> &values, 
            IndexPrecision nnz_per_row, 
            IndexPrecision dimension);

  /** Initialize from a text file in the following format
   *   row column value \n
   *   NOTE !!!!!!!
   *   the text file must be sorted according to the rows
   *   meaning that the rows should be in increasing order
   *   you can do that easily in unix with the sort -n command
   *   if it is not sorted it will still work but it will load 
   *   much slower
   */
  void Init(std::string textfile);
  /** 
   * Copy function, used also by copy constructor
  */
  void Copy(const SparseMatrixT &other);
  /** 
   * Not implemented yet
   */
  void Alias(const GenMatrix<ValuePrecision>& other);

  void Destruct();
  /** 
   * Initialize the diagonal 
   */
  template<template<typename> ValueContainer>
  void InitDiagonal(const ValueContainer<ValuePrecision> &vec); 
  /** 
   * Initialize the diagonal with a constant
  */
  
  void InitDiagonal(const ValuePrecision value);
  /** It is recomended that you load the matrix row-wise, Before
   *   you do that call StartLoadingRows()
   */
  void StartLoadingRows();
  /** 
   * All these functions load Rows, with the data in different format
   * WARNING!!! there should not be duplicate indices 
   * If you load the same row twice or the row has duplicate columns
   * then you will get unexpected results
   */
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer>
  void LoadRow(IndexContainer<IndexPrecision> row, 
               IndexContainer<IndexPrecision> &columns, 
               ValueContainer<ValuesPrecision> &values);
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer>
  void LoadRow(IndexContainer<IndexPrecision> row, 
               IndexContainer<IndexPrecision> *columns, 
               ValueContainer<ValuePrecision> &values);
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer>
  void LoadRow(IndexContainer<IndexPrecision> row, 
               IndexContainer<IndexPrecision> num, 
               IndexPrecision *columns, 
               ValueContainer<ValuePrecision>  *values);
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer> 
  void LoadRow(IndexPrecision row, 
               IndexContainer<IndexPrecision> &columns, 
               ValueContainer<ValuePrecision>  &values);
  /**
   * Sort Indices. This is a precondition for running any operation in Sparsem
   * addition, subdraction, multiplication, between matrices. EndLoading also does 
   * that, but it also does other things. SortIndices() is a lighter operation
   * as it still allows insertions. Note also that if you call EndLoading gets
   * are slower, so prefer SortIndices if you don't need EndLoading. But if you 
   * need EndLoading do not call SortIndices, as EndLoading will do the sorting too.
   * It is just redundant
   */
   void SortIndices(); 
   /**
    * Explicitly set if indices are sorted
    */
   void set_indices_sorted(bool indices_sorted) {
     indices_sorted_=indices_sorted;
   }
  /** When you are done call this it does some optimization in the storage, no
   *  further asignment 
   *  !!!WARNING !!!!
   *  if there are empty rows it is going to eliminate them, 
   *  so it might change the dimensions of the matrix
   */
  void EndLoading();
  /** 
   * It makes the matrix symmetric. It scans the rows of the matrix and for every (i,j)
   * element (j,i) equal to (j,i)
   */
  void MakeSymmetric();  
  /** 
   * if you know that the matrix is symmetric set the flag
   */
  void set_symmetric(bool val) {
    issymmetric_ = val;
  }
  /** 
   * Sets the diagonal with the values of the vector
   */
  template<template<typename> ValueContainer> 
  void SetDiagonal(const ValueContainer<ValuePrecision> &vector); 
  /** 
   * Sets the diagonal withe a scalar
  */
  void SetDiagonal(const ValuePrecision scalar); 
  /** 
   * Not Implemented yet
   */
  void SwapValues(SparseMatrixT* other);
  /**
   * Returns a copy of a row. It allocates memory for *columns
   * and values. Make sure that you do delete []*columns and
   * delete []*values after you use them
   */
  void get_row_copy(IndexPrecision r, IndexPrecision *num, 
                  IndexPrecision **columns, 
                  ValuePrecision **values) const ; 
  /** Access values, It will fail if EndLoading() has been called
  */
  double get(IndexPrecision r, IndexPrecision c) const;
  /** 
   * Set Values
   */
  void  set(IndexPrecision r, IndexPrecision c, ValuePrecision v);
  /**
   * Returns the transpose of the matrix. 
   * if it is symmetric it just returns a copy of the same matrix
   */
  void Transpose(SparseMatrixT *transpose);
  /** 
   * scales the matrix with a scalar
   */
  void Scale(ValuePrecision scalar) {
    matrix_->Scale(scalar);
  }
  /** 
   * negate the matrix get -A
   */
  void Negate();
  /** The matrix will be scaled such that A(i,j) = x(j)*A(i,j) 
   *  where i denotes the global row number of A and j denotes the  column number 
   *  You must have called EndLoading()
   */
  template<template<typename> ValueContainer>
  void ColumnScale(const ValueContainer<ValuePrecision> &vec) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You should call EndLoading first...\n");
    }
    Tpetra::Vector<IndexPrecision, ValuePrecision> 
      temp(View, *map_, (ValuePrecision*)vec.ptr());
    matrix_->RightScale(temp);
  }
  /** The  matrix will be scaled such that A(i,j) = x(i)*A(i,j) 
   *   where i denotes the row number of A and j denotes the column number of A.
   *   You must have called EndLoading()
   */
  template<template<typename> ValueVector>
  void RowScale(const ValueVector<ValuePrecision> &vec) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You should call EndLoading first...\n");
    }
    Tpetra::Vector<IndexPrecision, ValuePrecision> 
      temp(View, *map_, (ValuePrecision *)vec.ptr());
    matrix_->LeftScale(temp);
  }
  /** 
   * computes the L1 norm
   * You must have called EndLoading()
   */
  long double L1Norm() {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You should call EndLoading first...\n");
    }
    return matrix_->NormOne();
  }
  /** 
   * L infinity norm
   * You must have called EndLoading()
   */
  Precision LInfNorm() {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You should call EndLoading first...\n");
    }
    return matrix_->NormInf();
  }
  /** 
   * Computes the inverse of the sum of absolute values of the rows 
   *   of the matrix 
   *   You must have called EndLoading()
   */
  template<template<typename> ValueVector>
  void InvRowSums(ValueVector<ValuePrecision> *result) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You have to call EndLoading first...\n");
    }
    result->Init(dimension_);
    Tpetra::Vector<IndexPrecision, ValuePrecision> 
      temp(View, *map_, result->ptr());
    matrix_->InvRowSums(temp);
  }
  /** 
   * Computes the  the sum of absolute values of the rows 
   * of the matrix 
   * You must have called EndLoading()
  */
  typename<template<typename> ValueContainer>
  void RowSums(ValueContainer<ValuePrecision> *result) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You have to call EndLoading first...\n");
    }
    result->Init(dimension_);
    Tpetra::Vector temp(View, *map_, result->ptr());
    matrix_->InvRowSums(temp);
    for(IndexPrecision i=0; i<dimension_; i++) {
      (*result)[i]= 1/(*result)[i];
    }
    
  }
  /** 
   * Computes the inv of max of absolute values of the rows of the matrixa
   * You must have called EndLoading()
   */
  template<template<typename> ValueContainer>
  void InvRowMaxs(ValueContainer<ValuePrecision> *result) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You have to call EndLoading first...\n");
    }
    result->Init(dimension_);
    Tpetra::Vector<IndexPrecision, ValuePrecision> temp(View, *map_, result->ptr());
    matrix_->InvRowMaxs(temp);
  } 
  /** Computes the inverse of the sum of absolute values of the columns of the
   *  matrix
   *  You must have called EndLoading()
   */
  template<template<typename> ValueContainer>
  void InvColSums(ValueContainer<ValuePrecision> *result) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You have to call EndLoading first...\n");
    }
    result->Init(dimension_);
    Tpetra::Vector<IndexPrecision, ValuePrecision> temp(View, *map_, result->ptr());
    matrix_->InvColSums(temp);
  }
  /**
   *  Computes the inv of max of absolute values of the columns of the matrix
   *  You must have called EndLoading()
   */
  template<template<typename> ValueContainer>
  void InvColMaxs(ValueContainer<ValuePrecision> *result) {
    if (unlikely(!matrix_->Filled())) {
      FATAL("You have to call EndLoading first...\n");
    }
    result->Init(num_of_columns_);  
    Tpetra::Vector<IndexPrecision, ValuePrecision> temp(View, *map_, result->ptr());
    matrix_->InvColMaxs(temp);
  }
  /** 
   * Get the number of rows
   */
  IndexPrecision n_rows() {
    return num_of_rows_;
  }
  /** 
   * Get the number of columns
   */
  IndexPrecision n_cols() {
    return num_of_columns_;
  }
  /** 
   * Dimension should be equal to the number of rows
   */
  IndexPrecision dimension() {
    return dimension_;
  }
  /** 
   * The number of non zero elements
   */
  IndexPrecision nnz() const {
   return matrix_->NumGlobalNonzeros();
  }
  /** Apply a function on every non-zero element, very usefull for kernels
   *  If you have entered a zero element then it will also be applied on it 
   *  as well
   */
  template<typename FUNC>
  void ApplyFunction(FUNC &function);
  /** 
   * For debug purposes you can call it to print the matrix
   */
  std::string Print() {
    std::ostringstream s1;
    matrix_->Print(s1);
    return s1.str();
  }
  void ToFile(std::string file);
  /** 
   * Computes the eignvalues with the Krylov Method
   *  IndexPrecision num_of_eigvalues:   number of eigenvalues to compute
   *       std::string eigtype:   Choose which eigenvalues to compute
   *                              Choices are:
   *                              LM - target the largest magnitude  
   *                              SM - target the smallest magnitude 
   *                              LR - target the largest real 
   *                              SR - target the smallest real 
   *                              LI - target the largest imaginary
   *                              SI - target the smallest imaginary
   *       Matrix *eigvectors:    The eigenvectors computed must not be initialized
   *       Vector              *real_eigvalues:  real part of the eigenvalues
   *                                             must not be initialized. 
   *                                             The eigenvalues
   *                                             returned might actually be less 
   *                                             than the ones requested
   *                                             for example when the matrix has 
   *                                             rank n< eigenvalues requested
   *       Vector              *imag_eigvalues:  imaginary part of the eigenvalues
   *                                             must not be initialized. 
   *                                             The same as real_eigvalues
   *                                             hold for the space allocated
   *                                             in the non-symmetric case    
   */
  void Eig(IndexPrecision num_of_eigvalues, 
           std::string eigtype, 
           GenMatrix<ValuePrecision> *eigvectors,  
           GenVector<ValuePrecision> *real_eigvalues, 
           GenVector<ValuePrecision> *imag_eigvalues);
  /** 
   * Solves the pancil problem:
   *  A*x=lambda *B*x
   *  where pencil_part is the B matrix
   *  You have to call EndLoading() for B first
   */
  void Eig(SparseMatrixT &pencil_part,
           IndexPrecision num_of_eigvalues, 
           std::string eigtype, 
           GenMatrix<ValuePrecision> *eigvectors,  
           GenVector<ValuePrecision> *real_eigvalues, 
           GenVector<ValuePrecision> *imag_eigvalues);

  /** 
   * Linear System solution, Call Endloading First.
   */
  void LinSolve(GenVector<ValuePrecision> &b, // must be initialized (space allocated)
                GenVector<ValuePrecision> *x, // must be initialized (space allocated)
                ValuePrecision tolerance,
                IndexPrecision iterations);

  /** Use this for the general case
  */
  void LinSolve(GenVector<ValuePrecision> &b, GenVector<ValuePrecision> *x) {
    LinSolve(b, x, 1E-9, 1000);
  }
  void IncompleteCholesky(IndexPrecision level_fill,
                          ValuePrecision drop_tol,
                          SparseMatrixT *u, 
                          GenVector<ValuePrecision *d,
                          ValuePrecision *condest);     

 private:
  IndexPrecision dimension_;
  IndexPrecision num_of_rows_;
  IndexPrecision num_of_columns_;
  Tpetra::SerialPlatform<IndexPrecision> comm_;
  bool issymmetric_;
  bool indices_sorted_;
  Tpetra::Map<IndexPrecision> *map_;
  Teuchos::RCP<Tpetra::CrsMatrix<IndexPrecision,  ValuePrecision> matrix_;
  IndexPrecision *my_global_elements_;

  void Init(const Tpetra_CrsMatrix<IndexPrecision, ValuePrecision> &other); 
  template<template<typename> IndexContainer, 
           template<typename> ValueContainer>
  void Load(const IndeContainer<IndexPrecision> &rows, 
            const IndexContainer<IndexPrecision> &columns, 
            const ValueContainer<ValuePrecision> &values);
  void AllRowsLoad(GenVector<ValuePrecision> &rows, 
                   GenVector<ValuePrecision> &columns);

};

/**
 * Sparsem is more like an interface providing basic lagebraic operations
 * addition, subtraction multiplicatiion, for sparse matrices. It should
 * have been a namespace, but I prefered to make it a class with static
 * member functions so that I can declare it as a friend to the SparseMatrix
 * class
 *
 * WARNING !!!! THE RESULT SHOULD NOT BE INITIALIZED !!!
 * WARNING !!!! Before running any of those call SortIndices() or EndLoading() or
 * if you know that you have  loaded the rows with sorted indices then
 * explicitly set the indice_sorted flag to true with  set_indices_sorted(true);
 */
  template<typename IndexPrecision, typename ValuePrecision>
   inline void Add(const SparseMatrix<IndexPrecision, ValuePrecision> &a, 
                   const SparseMatrix<IndexPrecision, ValuePrecision> &b, 
                         SparseMatrix<IndexPrecision, ValuePrecision> *result);

  template<typename IndexPrecision, typename ValuePrecision>
  static inline void Subtract(const SparseMatrix<IndexPrecision, ValuePrecision> &a,
                              const SparseMatrix<IndexPrecision, ValuePrecision> &b,
                              SparseMatrix<IndexPrecision, ValuePrecision> *result);

  /** Multiplication of two matrices A*B in matlab notation 
   *  If B is symmetric then it is much faster, because we can
   *  multiply rows. Otherwise we have to compute the transpose
   *  As an advise multiplication of two sparse matrices might 
   *  lead to a dense one, so please be carefull
   */
  template<typename IndexPrecision, typename ValuePrecision>
  static inline void Multiply(const SparseMatrix<IndexPrecision, ValuePrecision> &a,
                              const SparseMatrix<IndexPrecision, ValuePrecision> &b,
                              SparseMatrix<IndexPrecision, ValuePrecision> *result);
  /* 
   * Computes the result = A * A^T
   */
  template<typename IndexPrecision, typename ValuePrecision>
  inline void MultiplyT(SparseMatrix<IndexPrecision, ValuePrecision> &a,
                        SparseMatrix<IndexPrecision, ValuePrecision> *result);

  /** The transpose flag should be set to true if 
   *  we want to use the transpose of mat, otherwise
   *  set it to false.
   */
  template<typename IndexPrecision, typename ValuePrecision, IsVector=false>
  inline void Multiply(const SparseMatrix<IndexPrecision, ValuePrecision> &mat,
                       const GenMatrix<ValuePrecision, IsVector>  &vec,
                       GenMatric<Precision, IsVector> *result,
                       bool transpose_flag);
  /** 
   * Multiply the matrix with a scalar
   */
  template<typename IndexPrecision, typename ValuePrecision>
  inline void Multiply(const SparseMatrix<IndexPrecision, ValuePrecision> &mat,
                       const ValuePrecision scalar,
                       SparseMatrix<IndexPrecisin, ValuePrecision> *result);
  
  /** 
   * element wise multiplication of the matrices
   * A.*B in matlab notation
  */
  template<typename IndexPrecision, typename ValuePrecision>
  inline void DotMultiply(const SparseMatrix<IndexPrecision, ValuePrecision> &a,
                          const SparseMatrix<IndexPrecision, ValuePrecision> &b,
                          SparseMatrix<IndexPrecision, ValuePrecision> *result);
/***********************************************************/
  template<typename IndexPrecision, typename ValuePrecision, bool IsVector=true>
  inline void ScaleRows(const GenMatrix<ValuePrecision, IsVector> &scales, 
                        SparseMatrix<IndexPrecision, ValuePrecision> *matrix)

  template<typename IndexPrecision, typename ValuePrecision>
  inline ValuePrecision LengthEuclidean(
      SparseMatrix<IndexPrecision, ValuePrecision> &x);


  template<typename IndexPrecision, typename ValuePrecision>
  inline long double Dot(const SparseMatrix<IndexPrecision, ValuePrecision> &x, 
      const SparseMatrix<IndexPrecision, ValuePrecision> &y); 

  template<typename IndexPrecision, typename ValuePrecision>
  inline void Scale(ValuePrecision alpha, 
      SparseMatrix<IndexPrecision, ValuePrecision> *x); 

  template<typename IndexPrecision, typename ValuePrecision, MemoryAlloc M>
  inline void ScaleOverwrite(ValuePrecision alpha, 
      const SparseMatrix<IndexPrecision, ValuePrecision> &x, 
      SparseMatrix<IndexPrecision, ValuePrecision> *y); 
  
  template<typename IndexPrecision, typename ValuePrecision
  inline void AddExpert(ValuePrecision alpha, 
      const SparseMatrix<IndexPrecision, ValuePrecision> &x, 
      SparseMatrix<IndexPrecision, ValuePrecision> *y); 

  template<typename Precision>
  inline void AddTo(index_t length, const Precision *x, Precision *y)

  template<typename IndexPrecision, typename ValuePrecision, MemoryAlloc M>
  inline void Add(const SparseMatrix<IndexPrecision, ValuePrecision> &x, 
      const SparseMatrix<IndexPrecision, ValuePrecision> &y, 
      SparseMatrix<IndexPrecision, ValuePrecisin> *z); 

  template<typename IndexPrecision, typename ValuePrecision>
  inline void SubFrom(const SparseMatrix<IndexPrecision, ValuePrecision> &x, 
                      SparseMatrix<IndexPrecision, ValuePrecision> *y) 

  template<typename IndexPrecision, typename ValuePrecision, MemoryAlloc M>
  inline void Sub(const SparseMatrix<IndexPrecision, ValuePrecision> &x, 
      const SparseMatrix<IndexPrecision, ValuePrecision>  &y, 
      SparseMatrix<IndexPrecision, ValuePrecision> *z); 

  template<typename Precision>
  inline void TransposeSquare(GenMatrix<Precision, false> *X)

  template<typename Precision, TransMode IsTransA, bool IsVector=false>
  inline void MulExpert(
      Precision alpha, 
      const GenMatrix<Precision, false> &A,
      const GenMatrix<Precision, IsVector> &x, 
      Precision beta, GenVector<Precision, IsVector> *y) 

  template<typename Precision, TransMode IsTransA, TransMode IsTransB>
  inline void MulExpert(
      Precision alpha, 
      const GenMatrix<Precision, false> &A,
      const GenMatrix<Precision, false> &B, 
      Precision beta, GenVector<Precision, IsVector> *C) 

  template<typename Precision, TransMode IsTransA, TransMode IsTransB, 
           MemoryAlloc M>
  inline void Mul(const GenMatrix<Precision, false> &A, 
                  const GenMatrix<Precision, false> &B, 
                        GenMatrix<Precision, false> *C) 
  template<typename Precision>
  long double Determinant(const GenMatrix<Precision, false> &A)

  template<typename Precision>
  inline success_t SolveExpert(
      f77_integer *pivots, GenMatrix<Precision, false> *A_in_LU_out,
      index_t k, Precision *B_in_X_out) 

  template<typename Precision>
  success_t EigenExpert(GenMatrix<Precision, false> *A_garbage,
      Precision *w_real, Precision *w_imag, Precision *V_raw) 
#include "fastlib/sparse/sparse_matrix_impl.h"
#endif
