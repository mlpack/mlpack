// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file matrix.h
 *
 * Basic double-precision vector and matrix classes.
 */

#ifndef LA_MATRIX_H
#define LA_MATRIX_H

#include "fastlib/base/base.h"
#ifndef DISABLE_DISK_MATRIX
#include "fastlib/mmanager/memory_manager.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Static assertion for the matrix class
 *        we use them to make sure people 
 *        do not accidently give wrong  template arguments
 *
 */

/**
 * @brief Because we want to be backwards compliant the GenMatrix
 *        can also be used as an old vector. We use the following 
 *        assertion so that people don't accidently call matrix 
 *        member functions that cannot be used for vectors.
 *  @code 
 *   template<bool> struct You_are_trying_to_use_a_vector_as_a_matrix;
     template<> struct You_are_trying_to_use_a_vector_as_a_matrix<false> {};
 *  @endcode 
 */
template<bool> struct You_are_trying_to_use_a_vector_as_a_matrix;
template<> struct You_are_trying_to_use_a_vector_as_a_matrix<false> {};

/**
 * @brief Sometimes we want to Copy a GenMatrix of floats
 *        to a GenMatrix of doubles, which is a valid thing to do
 *        The following static assertions are trying to prevent the 
 *        user from doing invalid operations such as copy a 
 *        GenMatrix of doubles to a GenMatrix of floats, where precision
 *        is lost
 *        So you can always copy to larger precision
 *        And you can always copy int to float double and long double
 *
 */
template<typename Precision1, typename Precision2> struct
You_have_a_precision_conflict;
template<typename Precision> struct
You_have_a_precision_conflict<Precision, Precision> {};
template<> struct
You_have_a_precision_conflict<double, float> {};
template<> struct
You_have_a_precision_conflict<long double, float> {};
template<> struct
You_have_a_precision_conflict<long double, double> {};
template<> struct
You_have_a_precision_conflict<float, int> {};
template<> struct
You_have_a_precision_conflict<double, int> {};
template<> struct
You_have_a_precision_conflict<long double, int> {};


/**
 * @brief  We use this static assertion so that we don't accidently
 *         try to copy a matrix on a vector
 *
 */
template<bool, bool> struct You_are_assigning_a_matrix_on_a_vector {};
template<true,false> struct You_are_assigning_a_matrix_on_a_vector;


/**
 * General-Precision column-major matrix for use with LAPACK.
 * This class is backwords compatible, so that it can represent
 * the old fastlib GenVector.
 *
 * @code
 *  template<typename Precision, bool IsVector=false>
 *   class GenMatrix;
 * @endcode
 *
 * @param Precision In general you can choose any precision you want, but 
 *        our current LAPACK implementation suppors floats and doubles.
 *        It is not difficult thought to support complex numbers. You can 
 *        use integers too,  but LAPACK will not work 
 *
 * Your code can have huge performance hits if you fail to realize this
 * is column major.  For datasets, your columns should be individual points
 * and your rows should be features.
 *
 * TODO: If it's not entirely obvious or well documented how to use this
 * class please let the FASTlib people know.
 */

template<typename Precision, bool IsVector=false>
class GenMatrix {
 private:
  /** Linearized matrix (column-major). */
  Precision *ptr_;
  /** Number of rows. */
  index_t n_rows_;
  /** Number of columns. */
  index_t n_cols_;
  /** Number of elements for faster access */
  index_t n_elements_; 
  /** Whether I am a strong copy (not an alias). */
  bool should_free_;
  
  /**
   *  @brief this is a helper trait to make sure that the PrintDebug
   *         function works correctly
   */
  template<typename>
  class PrintTrait {
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const;  
  };
  
  template<>
  class PrintTrait<float> {
   public:
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3f ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
  };

  class PrintTrait<double> {
   public:
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const {
      PrintTrait<float>::Print(mat, name, stream);
    } 
  };

  class PrintTrait<long double> {
   public:
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const {
      PrintTrait<float>::Print(mat, name, stream);
  };

  class PrintTrait<int> {
   public:
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const {
      fprintf(stream, "----- MATRIX %s ------\n", name);
      for (index_t r = 0; r < mat.n_rows(); r++) {
        for (index_t c = 0; c < mat.n_cols(); c++) {
          fprintf(stream, "%+3.3i ", mat.get(r, c));
        }
        fprintf(stream, "\n");
      }
    }
  };

  class PrintTrait<long int> {
   public:
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const {
      PrintTrait<int>::Print(mat, name, stream);

  };

  class PrintTrait<long long int> {
   public:
    static void Print(GenMatrix<Precision, IsVector> &mat,
                      const char *name = "", FILE *stream = stderr) const {
      PrintTrait<int>::Print(mat, name, stream);

  };

 public:
  /**
   * Creates a Matrix with uninitialized elements of the specified size.
   */
  GenMatrix(index_t in_rows) {
    DEBUG_ONLY(Uninitialize_());
    Init(in_rows);  
  }
  GenMatrix(index_t in_rows, index_t in_cols) {
    if (IsVector == true) {
      if (in_cols!=1) {
        FATAL("You are trying to initialize a vector with more than one column");
      }
    }
    DEBUG_ONLY(Uninitialize_());
    Init(in_rows, in_cols);
  }

  /**
   * Copy constructor -- for use in collections.
   */
  GenMatrix(const GenMatrix<Precision, IsVector>& other) {
    DEBUG_ONLY(Uninitialize_());
    Copy(other);
  }

  /**
   * Creates a matrix that can be initialized.
   */
  GenMatrix() {
    DEBUG_ONLY(Uninitialize_());
    should_free_ = false;
  }

  /**
   * Empty destructor.
   */
  ~GenMatrix() {
    Destruct();
  }
  
  /**
   * Destructs this, so that it is suitable for you to call an initializer
   * on this again.
   */
  void Destruct() {
    DEBUG_ASSERT_MSG(ptr_ != BIG_BAD_POINTER(Precision),
       "You forgot to initialize a Matrix before it got automatically freed.");
    if (unlikely(should_free_)) {
      mem::DebugPoison(ptr_, n_rows_ * n_cols_);
      mem::Free(ptr_);
      DEBUG_ONLY(Uninitialize_());
    }
    DEBUG_POISON_PTR(ptr_);
    DEBUG_ONLY(n_rows_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_cols_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_elements_ = BIG_BAD_NUMBER);
  }

  /**
   * Creates a Matrix with uninitialized elements of the specified size.
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  void Init(index_t in_rows, index_t in_cols) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mem::Alloc<Precision>(in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = in_cols;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = true;
  }

  /**
   * You can use it to initialize vectors and matrices too. For matrices
   * it will assume that it is an one column
   */
  void Init(index_t in_rows) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mem::Alloc<Precision>(in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = 1;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = true;
  }

  /**
   * Creates a diagonal matrix.
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  template<template<typename> SerialContainer>
  void InitDiagonal(const SerialContainer<Precision>& v) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    Init(v.size(), v.size());
    SetZero();
    SetDiagonal(v);
  }

  /**
   * Creates a diagonal matrix.
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  void InitDiagonal(const index_t dimension, const Precision value) {
   You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
   Init(dimension, dimension);
    SetZero();
    for(index_t i=0; i<dimension; i++) {
      this->set(i, i, value);
    }
  }

  /**
   * Creates a Matrix with uninitialized elements of the specified
   * size statically. This matrix is not freed!
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  void StaticInit(index_t in_rows, index_t in_cols) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<Precision>
      (in_rows * in_cols);
    n_rows_ = in_rows;
    n_cols_ = in_cols;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
#else
    Init(in_rows, in_cols);
#endif
  }
  /**
   * Creates a Matrix/Vector with uninitialized elements of the specified
   * size statically. This matrix is not freed!
   * For matrices it assumes column number is one
   */
  void StaticInit(index_t in_rows) {
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<Precision>
      (in_rows );
    n_rows_ = in_rows;
    n_cols_ = 1;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
#else
    Init(in_rows);
#endif
  }
 
  /**
   * Creates a diagonal matrix.
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  template<template<typename> SerialContainer, typename OtherPrecision>
  void StaticInitDiagonal(const SerialContainer<OtherPrecision>& v) {
   You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
   You_have_a_precision_conflict<Precision, OtherPrecision>(); 
   StaticInit(v.size(), v.size());
   SetDiagonal(v);
  }
 
  /**
   * Creates a diagonal matrix.
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  void StaticInitDiagonal(const index_t dimension, const Precision value) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    StaticInit(dimension, dimension);
    for(index_t i=0; i<dimension; i++) {
      this->set(i, i, value);
    }
  }

  /**
   * Sets the entire matrix to zero.
   */
  void SetAll(Precision d) {
    mem::RepeatConstruct(ptr_, d, n_elements());
  }

  /**
   * Makes this matrix all zeroes.
   */
  void SetZero() {
    // TODO: If IEEE floating point is used, this can just be a memset to
    // zero
    SetAll(0);
  }

  /**
   * Makes this a diagonal matrix whose diagonals are the values in v.
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  template<template<typename> SerialContainer, typename OtherPrecision>
  void SetDiagonal(const SerialContainer<OtherPrecision>& v) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    You_have_a_precision_conflict<Precision, OtherPrecision>(); 
    DEBUG_ASSERT(n_rows() == v.length());
    DEBUG_ASSERT(n_cols() == v.length());
    SetZero();
    index_t n = v.length();
    for (index_t i = 0; i < n; i++) {
      set(i, i, v[i]);
    }
  }

  /**
   * Makes this uninitialized matrix a copy of the other Matrix.
   *
   * @param other the vector to explicitly copy
   */
  template<typename OtherPrecision, bool OtherIsVector>
  void Copy(const GenMatrix<OtherPrecision>, OtherIsVector>& other) {
    You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    Copy(other.ptr(), other.n_rows(), other.n_cols());    
  }

  /**
   * Makes this uninitialized matrix a copy of the other vector.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error. 
   */
  template<typename OtherPrecision>
  void Copy(const OtherPrecision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();   
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mem::AllocCopyValues<Precision, OtherPrecision>(ptr_in, n_rows_in * n_cols_in);
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = true;
  }

  /**
   * Makes this uninitialized Vector a copy of the other vector.
   * If you use this for a matrix then it assumes single column
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
  */
  template<typename OtherPrecision>
  void Copy(const OtherPrecision *ptr_in, index_t n_rows_in) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    DEBUG_ONLY(AssertUninitialized_());
    ptr_ = mem::AllocCopyValues<Precision, OtherPrecision>(ptr_in, n_rows_in * n_cols_in);
    n_rows_ = n_rows_in;
    n_cols_ = 1;
    n_elements_ = n_rows_ * n_cols_;   
    should_free_ = true;
  }

  /**
   * Makes this uninitialized matrix a static copy of the other
   * vector which will not be freed!
   *
   * @param other the vector to explicitly copy
   */
  template<typename OtherPrecision, bool OtherIsVector>   
  void StaticCopy(const GenMatrix<OtherPrecision, OtherIsVector>& other) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
    if (IsVector==false) {
      StaticCopy(other.ptr(), other.n_rows(), other.n_cols());    
    } else {
      StaticCopy(other.ptr(), other.n_rows());        
    }
  }

  /**
   * Makes this uninitialized matrix a static copy of the other
   * vector, which will not be freed!
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
  */
  template<typename OtherPrecision>
  void StaticCopy(const OtherPrecision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());

    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<Precision>
      (n_rows_in * n_cols_in);
    mem::CopyValues<Precision, OtherPrecision>(ptr_, ptr_in, n_rows_in * n_cols_in);

    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
#else
    Copy(ptr_in, n_rows_in, n_cols_in);
#endif
  }
 
  template<typename OtherPrecision>
  void StaticCopy(const OtherPrecision *ptr_in, index_t n_rows_in) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());

    ptr_ = mmapmm::MemoryManager<false>::allocator_->Alloc<Precision>
      (n_rows_in * n_cols_in);
    mem::CopyValues<Precision, OtherPrecision>(ptr_, ptr_in, n_rows_in * n_cols_in);

    n_rows_ = n_rows_in;
    n_cols_ = 1;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
#else
    Copy(ptr_in, n_rows_in);
#endif
  }
 
  /**
   * Makes this uninitialized matrix an alias of another matrix.
   *
   * Changes to one matrix are visible in the other (and vice-versa).
   *
   * @param other the other vector
   */
  template<typename OtherPrecision, bool OtherIsVector>
  void Alias(const GenMatrix<OtherPrecision, OtherIsVector> & other) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
  // we trust in good faith that const-ness won't be abused
    if (IsVector==false) {
      Alias(other.ptr_, other.n_rows(), other.n_cols());
    } else {
      Alias(other.ptr_, other.n_rows());
    }
  }
  
  /**
   * Makes this uninitialized matrix an alias of an existing block of doubles.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   * NOTICE! this should be used only when you are in the matrix mode
   * if you set IsVector template parameter to true, which means you 
   * are using it as a vector then you will get a compile time error.
   */
  void Alias(Precision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
  }
  
  void Alias(Precision *ptr_in, index_t n_rows_in) {
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = 1;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
  }
 
  /**
   * Makes this a 1 row by N column alias of a vector of length N.
   *
   * @param row_vector the vector to alias
   */
  
/*  void AliasRowVector(const GenVector<T>& row_vector) {
    Alias(const_cast<T*>(row_vector.ptr()), 1, row_vector.length());
  }
*/  
  
  /**
   * Makes this an N row by 1 column alias of a vector of length N.
   *
   * @param col_vector the vector to alias
   */
/*  void AliasColVector(const GenVector<T>& col_vector) {
    Alias(const_cast<T*>(col_vector.ptr()), col_vector.length(), 1);
  }
*/

  /**
   * Makes this a weak copy or alias of the other.
   *
   * This is identical to Alias.
   */
  template<bool OherIsVector>
  void WeakCopy(const GenMatrix<Precision, OtherIsVector>& other) {
    Alias(other);
  }
  
  /**
   * Makes this uninitialized matrix the "owning copy" of the other
   * matrix; the other vector becomes an alias and this becomes the
   * standard.
   *
   * The other matrix must be the "owning copy" of its memory.
   *
   * @param other a pointer to the other matrix
   */
  template<bool OtherIsVector>
  void Own(GenMatrix<Precision, OtherIsVector>* other) {
    You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
    if (IsVector==false) { 
      Own(other->ptr(), other->n_rows(), other->n_cols());
    } else {
      Own(other->ptr(), other->n_rows());  
    }
    DEBUG_ASSERT(other->should_free_);
    other->should_free_ = false;
  }
  
  /**
   * Initializes this uninitialized matrix as the "owning copy" of
   * some linearized chunk of RAM allocated with mem::Alloc.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   *        allocated via mem::Alloc
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  void Own(Precision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ONLY(AssertUninitialized_());  
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    n_elements_ = n_rows_ * n_cols_;
   should_free_ = true;
  }
  
  void Own(Precision *ptr_in, index_t n_rows_in) {
    DEBUG_ONLY(AssertUninitialized_());  
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = 1;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = true;
  }

  /**
   * Makes this uninitialized matrix the "owning copy" of the other
   * matrix; the other vector becomes an alias and this becomes the
   * standard statically.
   *
   * The other matrix must have been allocated statically.
   *
   * @param other a pointer to the other matrix
   */
  template<bool OtherIsVector>
  void StaticOwn(GenMatrix<Precision, OtherIsVector>* other) {
    You_are_assigning_a_matrix_on_a_vector<IsVector, OtherIsVector>();
#ifndef DISABLE_DISK_MATRIX
    if (IsVector==false) {
      StaticOwn(other->ptr(), other->n_rows(), other->n_cols());
    } else {
      StaticOwn(other->ptr(), other->n_rows()); 
    }
#else
    Own(other);
#endif
  }
  
  /**
   * Initializes this uninitialized matrix as the "owning copy" of
   * some linearized chunk of RAM allocated statically.
   *
   * @param ptr_in the pointer to a block of column-major doubles
   *        allocated via mem::Alloc
   * @param n_rows_in the number of rows
   * @param n_cols_in the number of columns
   */
  template<bool OtherIsVector>
  void StaticOwn(OtherPrecision *ptr_in, index_t n_rows_in, index_t n_cols_in) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
#ifndef DISABLE_DISK_MATRIX
    DEBUG_ONLY(AssertUninitialized_());
    
    ptr_ = ptr_in;
    n_rows_ = n_rows_in;
    n_cols_ = n_cols_in;
    n_elements_ = n_rows_ * n_cols_;
    should_free_ = false;
#else
    Own(ptr_in, n_rows_in, n_cols_in);
#endif
  }

  /**
   * Make a matrix that is an alias of a particular slice of my columns.
   *
   * @param start_col the first column
   * @param n_cols_new the number of columns in the new matrix
   * @param dest an UNINITIALIZED matrix
   */
  void MakeColumnSlice(index_t start_col, index_t n_cols_new,
      GenMatrix<Precision, IsVector> *dest) const {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_BOUNDS(start_col, n_cols_);
    DEBUG_BOUNDS(start_col + n_cols_new, n_cols_ + 1);
    dest->Alias(ptr_ + start_col * n_rows_,
        n_rows_, n_cols_new);
  }
  
  /**
   * Make an alias of a reshaped version of this matrix (column-major format).
   *
   * For instance, a matrix with 2 rows and 6 columns can be reshaped
   * into a matrix with 12 rows 1 column, 1 row and 12 columns, or a variety
   * of other shapes.  The layout of the new elements correspond exactly
   * to just pretending that the current column-major matrix is laid out
   * as a different column-major matrix.
   *
   * It is required that n_rows_new * n_cols_new is the same as
   * n_rows * n_cols of the original matrix.
   *
   * TODO: Considering using const Matrix& for third-party classes that want
   * to implicitly convert to Matrix.
   *
   * @param n_rows_in new number of rows
   * @param n_cols_in new number of columns
   * @param dest a pointer to an unitialized matrix
   * @return a reshaped matrix backed by the original
   */
  void MakeReshaped(index_t n_rows_in, index_t n_cols_in,
      GenMatrix<Precision, IsVector> *dest) const {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ASSERT(n_rows_in * n_cols_in == n_rows() * n_cols());
    dest->Alias(ptr_, n_rows_in, n_cols_in);
  }
  
  /**
   * Makes an alias of a particular column.
   *
   * @param col the column to alias
   * @param dest a pointer to an uninitialized vector, which will be
   *        initialized as an alias to the particular column
   */
  template<bool OtherIsVector>
  void MakeColumnVector(index_t col, 
      GenMatrix<OtherPrecision, OtherIsVector> *dest) const {
    DEBUG_BOUNDS(col, n_cols_);
    dest->Alias(n_rows_ * col + ptr_, n_rows_);
  }
  
  /**
   * Makes an alias of a subvector of particular column.
   *
   * @param col the column to alias
   * @param start_row the first row to put in the subvector
   * @param n_rows_new the number of rows of the subvector
   * @param dest a pointer to an uninitialized vector, which will be
   *        initialized as an alias to the particular column's subvector
   */
  template<bool OtherIsVector>
  void MakeColumnSubvector(index_t col, index_t start_row, index_t n_rows_new,
      GenMatrix<Precision, OtherIsVector> *dest) const {
    DEBUG_BOUNDS(col, n_cols_);
    DEBUG_BOUNDS(start_row, n_rows_);
    DEBUG_BOUNDS(start_row + n_rows_new, n_rows_ + 1);
    dest->Alias(n_rows_ * col + start_row + ptr_, n_rows_new);
  }
  
  /**
   * Retrieves a pointer to a contiguous array corresponding to a particular
   * column.
   *
   * @param col the column number
   * @return an array where the i'th element is the i'th row of that
   *         particular column
   */

  Precision *GetColumnPtr(index_t col) {
    DEBUG_BOUNDS(col, n_cols_);
    return n_rows_ * col + ptr_;
  }
  
  /**
   * Retrieves a pointer to a contiguous array corresponding to a particular
   * column.
   *
   * @param col the column number
   * @return an array where the i'th element is the i'th row of that
   *         particular column
   */
  const Precision *GetColumnPtr(index_t col) const {
    DEBUG_BOUNDS(col, n_cols_);
    return n_rows_ * col + ptr_;
  }
  
  /**
   * Copies a vector to a matrix column.
   * @param col1 the column number of this matrix
   * @param col2 the column number of the other matrix
   * @param mat the other matrix
   * @return nothing
   */  
   template<typename OtherPrecision, bool OtherIsVector
   void CopyColumnFromMat(index_t col1, index_t col2, 
       GenMatrix<OtherPrecision, OtherIsVector> &mat) {
     You_have_a_precision_conflict<Precision, OtherPrecision>();
     DEBUG_BOUNDS(col1, n_cols_);
     DEBUG_BOUNDS(col2, mat.n_cols());
     DEBUG_ASSERT(n_rows_==mat.n_rows());
     mem::CopyValues<Precision, OtherPrecision>
       (ptr_ + n_rows_ * col1,  mat.GetColumnPtr(col2), n_rows_);
  }
  /**
   * Copies a block of columns to a matrix column.
   * @param col1 the column number of this matrix
   * @param col2 the column number of the other matrixa
   * @param ncols the number of columns
   * @param mat the other matrix
   * @return nothing
   */  
   template<typename OtherPrecision, bool OtherIsVector>
   void CopyColumnFromMat(index_t col1, index_t col2, index_t ncols, 
       GenMatrix<OtherPrecision, OtherIsVector> &mat) {
     You_have_a_precision_conflict<Precision, OtherPrecision>();
     DEBUG_BOUNDS(col1, n_cols_);
     DEBUG_BOUNDS(col2, mat.n_cols());
     DEBUG_BOUNDS(col1+ncols-1, n_cols_);
     DEBUG_BOUNDS(col2+ncols-1, mat.n_cols());
     DEBUG_ASSERT(n_rows_==mat.n_rows());
     mem::CopyValues<Precision, OtherPrecision>(ptr_ + n_rows_ * col1, mat.GetColumnPtr(col2), ncols*n_rows_);
   }

   /**
   * Copies a column of matrix 1  to a column of matrix 2.
   * @param col1 the column number
   * @return nothing
   */  
   template<typename OtherPrecision, bool OtherIsVector>
   void CopyVectorToColumn(index_t col, 
       GenMatrix<OtherPrecision, OtherIsVector> &vec) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    DEBUG_BOUNDS(col, n_cols_);
     //NEED to FIX THIS !!!!!!!!!
     memcpy(ptr_ + n_rows_ * col, vec.ptr(), n_rows_*sizeof(T));
   }
 
  /**
   * Changes the number of columns, but REQUIRES that there are no aliases
   * to this matrix anywhere else.
   *
   * If the size is increased, the remaining space is not initialized.
   *
   * @param new_n_cols the new number of columns
   */
  void ResizeNoalias(index_t new_n_cols) {
    You_are_trying_to_use_a_vector_as_a_matrix<IsVector>();
    DEBUG_ASSERT(should_free_); // the best assert we can do
    n_cols_ = new_n_cols;
    ptr_ = mem::Realloc(ptr_, n_elements());
  }
  
  /**
   * Swaps all values in this matrix with values in the other.
   *
   * This is different from Swap, because Swap will only change what these
   * point to.
   *
   * @param other an identically sized vector to swap values with
   */
  void SwapValues(GenMatrix<Precision, IsVector>* other) {
    DEBUG_ASSERT(n_cols() == other->n_cols());
    DEBUG_ASSERT(n_rows() == other->n_rows());
    mem::Swap(ptr_, other->ptr_, n_elements());
  }
  
  /**
   * Copies the values from another matrix to this matrix.
   *
   * @param other the vector to copy from
   */
  template<typename OtherPrecision, bool OtherIsVector>
  void CopyValues(const GenMatrix& other) {
    You_have_a_precision_conflict<Precision, OtherPrecision>();
    DEBUG_ASSERT(n_rows() == other.n_rows());
    DEBUG_ASSERT(n_cols() == other.n_cols());
    // Fix this it is Wrong !!!!!!!!!
    mem::Copy(ptr_, other.ptr_, n_elements());
  }

  /**
   * Prints to a stream as a debug message.
   *
   * @param name a name that will be printed with the matrix
   * @param stream the stream to print to, defaults to @c stderr
   */
  //We need to templatize this !!!!
  void PrintDebug(const char *name = "", FILE *stream = stderr) const {
    PrintTrait<Precision>::Print(*this, name, stream);
  }
  
 public:
  /**
   * Returns a pointer to the very beginning of the matrix, stored
   * in a column-major format.
   *
   * This is suitable for BLAS and LAPACK calls.
   */
  const Precision *ptr() const {
    return ptr_;
  }
  
  /**
   * Returns a pointer to the very beginning of the matrix, stored
   * in a column-major format.
   *
   * This is suitable for BLAS and LAPACK calls.
   */
  Precision *ptr() {
    return ptr_;
  }
  
  /**
   * Gets a particular double at the specified row and column.
   *
   * @param r the row number
   * @param c the column number
   */
  Precision get(index_t r, index_t c) const {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    return ptr_[c * n_rows_ + r];
  }
  
  Precision get(index_t r) const {
    DEBUG_BOUNDS(r, n_elements_);
    return ptr_[r];
  }
 
  /**
   * Sets the value at the row and column.
   *
   * @param r the row number
   * @param c the column number
   * @param v the value to set
   */ 
  void set(index_t r, index_t c, Precision v) {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    ptr_[c * n_rows_ + r] = v;
  }
 
  void set(index_t r, Precision v) {
    DEBUG_BOUNDS(r, n_elements_);
    ptr_[r] = v;
  }
  
  /**
   * Gets a reference to a particular row and column.
   *
   * It is highly recommended you treat this as a single value rather than
   * part of an array; use ColumnSlice or Reshaped instead to make
   * subsections.
   */
  Precision &ref(index_t r, index_t c) {
    DEBUG_BOUNDS(r, n_rows_);
    DEBUG_BOUNDS(c, n_cols_);
    return ptr_[c * n_rows_ + r];
  }

  Precision &ref(index_t r) {
    DEBUG_BOUNDS(r, n_elements_);
    return ptr_[r];
  }

  Precision &operator[](index_t r) {
    return ptr_[r];
  }
  
  /** Returns the number of columns. */
  index_t n_cols() const {
    return n_cols_;
  }
  
  /** Returns the number of rows. */
  index_t n_rows() const {
    return n_rows_;
  }
  
  /**
   * Returns the total number of elements (power user).
   *
   * This is useful for iterating over all elements of the matrix when the
   * row/column structure is not important.
   */
  size_t n_elements() const {
    // TODO: putting the size_t on the outside may be faster (32-bit
    // versus 64-bit multiplication in cases) but is more likely to result
    // in bugs
    return n_elements_;
  }

  /**
   * It is exaclty the same like n_elements()
   * This definition provided a uniform interface for 
   * blas/lapck type operations
   */   
  size_t length() const {
   return ;
  }

  size_t size() const {
   return ;
  }


 private:
  void AssertUninitialized_() const {
    DEBUG_ASSERT_MSG(n_rows_ == BIG_BAD_NUMBER, "Cannot re-init matrices.");
  }
  
  void Uninitialize_() {
    DEBUG_POISON_PTR(ptr_);
    DEBUG_ONLY(n_rows_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_cols_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_elements__ = BIG_BAD_NUMBER);
  } 

};



/**
 * Low-overhead vector if length is known at compile time.
 */
template<typename Precision, int t_length>
class SmallVector : public GenVector<Precision> {
 private:
  Precision array_[t_length];
  
 public:
  SmallVector() {
    Alias(array_, t_length);
  }
  ~SmallVector() {}
  
 public:
  index_t length() const {
    return t_length;
  }

  Precision *ptr() {
    return array_;
  }
  
  const Precision *ptr() const {
    return array_;
  }
  
  Precision operator [] (index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
  
  Precision &operator [] (index_t i) {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
  
  Precision get(index_t i) const {
    DEBUG_BOUNDS(i, t_length);
    return array_[i];
  }
};

/** @brief A Matrix is a GenMatrix of double's.
 */
typedef GenMatrix<double, false> Matrix;

/** @brief A Vector is a GenMatrix of double's.
 */
typedef GenMatrix<double, true> Vector;

/**
 * Low-overhead matrix if size is known at compile time.
 */
template<typename Precision, int t_rows, int t_cols>
class SmallMatrix : public GenMatrix<Precision, true> {
 private:
  Precision array_[t_cols][t_rows];

 public:
  SmallMatrix() {
    Alias(array_[0], t_rows, t_cols);
  }
  ~SmallMatrix() {}

 public:
  const Precision *ptr() const {
    return array_[0];
  }

  Precision *ptr() {
    return array_[0];
  }

  Precision get(index_t r, index_t c) const {
    DEBUG_BOUNDS(r, t_rows);
    DEBUG_BOUNDS(c, t_cols);
    return array_[c][r];
  }

  void set(index_t r, index_t c, Precision v) {
    DEBUG_BOUNDS(r, t_rows);
    DEBUG_BOUNDS(c, t_cols);
    array_[c][r] = v;
  }

  Precision &ref(index_t r, index_t c) {
    DEBUG_BOUNDS(r, t_rows);
    DEBUG_BOUNDS(c, t_cols);
    return array_[c][r];
  }

  index_t n_cols() const {
    return t_cols;
  }

  index_t n_rows() const {
    return t_rows;
  }

  size_t n_elements() const {
    // TODO: putting the size_t on the outside may be faster (32-bit
    // versus 64-bit multiplication in cases) but is more likely to result
    // in bugs
    return size_t(t_rows) * size_t(t_cols);
  }

  Precision *GetColumnPtr(index_t col) {
    DEBUG_BOUNDS(col, t_cols);
    return array_[col];
  }

  const Precision *GetColumnPtr(index_t col) const {
    DEBUG_BOUNDS(col, t_cols);
    return array_[col];
  }
};

#endif
