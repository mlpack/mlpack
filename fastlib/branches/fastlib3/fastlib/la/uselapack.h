/**
 * @file uselapack.h
 *
 * Integration with BLAS and LAPACK.
 */

#ifndef LA_USELAPACK_H
#define LA_USELAPACK_H

#include "matrix.h"
#include "fastlib/col/arraylist.h"

#define DEBUG_VECSIZE(a, b) \
    DEBUG_SAME_SIZE((a).length(), (b).length())
#define DEBUG_MATSIZE(a, b) \
    DEBUG_SAME_SIZE((a).n_rows(), (b).n_rows());\
    DEBUG_SAME_SIZE((a).n_cols(), (b).n_cols())
#define DEBUG_MATSQUARE(a) \
    DEBUG_SAME_SIZE((a).n_rows(), (a).n_cols())

/*
 * TODO: this could an overhaul; LAPACK failures may mean either input
 * problems or just some linear algebra result (e.g. singular matrix).
 *
 * Perhaps extend success_t?
 */
#define SUCCESS_FROM_LAPACK(info) \
    (likely((info) == 0) ? SUCCESS_PASS : SUCCESS_FAIL)

#ifndef NO_LAPACK
#define USE_LAPACK
#define USE_BLAS_L1
#endif

#if defined(DOXYGEN) && !defined(USE_LAPACK)
/* Use the doxygen from the blas level 1 code. */
#define USE_LAPACK
#endif

#ifdef USE_LAPACK
#include "cppblas.h"
#include "cpplapack.h"
#endif

/**
 * Linear-algebra routines.
 *
 * This encompasses most basic real-valued vector and matrix math.
 * Most functions are written in a similar style, so after using a few it
 * should be clear how others are used.  For instance, input arguments
 * come first, and the output arguments are last.
 *
 * Many functions have several versions:
 *
 * - The Init version.  This stores the result into a Matrix or Vector
 * which has been declared, but hasn't been initialized.
 * This will call the proper Init function for you.  Example:
 *
 * @code
 * void SomeExampleCode(const Matrix& a, const Matrix& b) {
 *   Matrix c;
 *   // c must not be initialized
 *   la::MulInit(a, b, &c);
 * }
 * @endcode
 *
 * - The Overwrite version.
 * The result is stored in an existing matrix or vector that is the
 * correct size already, overwriting its previous contents, if any.
 * Example:
 *
 * @code
 * void SomeExampleCode(const Matrix& a, const Matrix& b) {
 *   Matrix c;
 *   c.Init(a.n_rows(), b.n_cols()); // c must be set to the proper size first
 *   la::MulOverwrite(a, b, &c);
 *   // c can be used over again
 *   la::MulOverwrite(a, b, &c);
 * }
 * @endcode
 *
 * - The Expert version, or power-user version.  We recommend you avoid this
 * version whenever possible.  These are the closest mapping to the LAPACK
 * or BLAS routine.  These will require that memory for destinations is
 * already allocated but the memory itself is not initalized.  These usually
 * take a Matrix or Vector pointer, or sometimes just a pointer to a slab of
 * doubles.  Sometimes, one of the "inputs" is completely destroyed while
 * LAPACK performs its computations.
 *
 *
 * FINAL WARNING:
 * With all of these routines, be very careful about passing in an argument
 * as both a source and destination -- usually this is an error and
 * will result in garbage results, except for simple
 * addition/scaling/subtraction.
 */
namespace la {

  enum MemoryAlloc {Init=0, Overwrite};
  enum TransMode {NoTrans=0, Trans};
  template<MemoryAlloc T>
  class AllocationTrait {
    template<typename Container>
    static inline void Init(index_t length, Container *cont);
    
    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont);
  };

  template<>
  class AllocationTrait<Init> {
    template<typename Container>
    static inline void Init(index_t length, Container *cont) {
      cont->Init(length);
    }
    
    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont) {
      cont->Init(dim1, dim2);
    }
  };

  template<>
  class AllocationTrait<Overwrite> {
    template<typename Container>
    static inline void Init(index_t length, Container *cont) {
      DEBUG_SAME(length, cont->length());
    }
    
    template<typename Container>
    static inline void Init(index_t dim1, index_t dim2, Container *cont) {
      DEBUG_SAME(dim1, cont->n_rows());
      DEBUG_SAME(dim2, cont->n_cols());
    }
  };


  /**
   * Scales the rows of a column-major matrix by a different value for
   * each row.
   */
  template<typename Precision>
  inline void ScaleRows(index_t n_rows, index_t n_cols,
      const Precision *scales, Precision *matrix) {
    do {
      for (index_t i = 0; i < n_rows; i++) {
        matrix[i] *= scales[i];
      }
      matrix += n_rows;
    } while (--n_cols);
  }

#if defined(USE_LAPACK) && defined(USE_BLAS_L1)
  /* --- Wrappers for BLAS level 1 --- */
  /*
   * TODO: Add checking to ensure arrays are non-overlapping.
   */

  /**
   * Finds the square root of the dot product of a vector with itself.
   */
  template<typename Precision>
  inline Precision LengthEuclidean(index_t length, const Precision *x) {
    return CppBlas<Precision>::nrm2(length, x, 1);
  }
  /**
   * Finds the dot-product of two arrays
   * (\f$\vec{x} \cdot \vec{y}\f$).
   */
  template<typename Precision>
  inline long double Dot(index_t length, const Precision *x, const Precision *y) {
    return CppBlas<Precision>::dot(length, x, 1, y, 1);
  }

  /**
   * Scales an array in-place by some factor
   * (\f$\vec{x} \gets \alpha \vec{x}\f$).
   */
  template<typename Precision>
  inline void Scale(index_t length, Precision alpha, Precision *x) {
    CppBlas<Precision>::scal(length, alpha, x, 1);
  }
  /**
   * Sets an array to another scaled by some factor
   * (\f$\vec{y} \gets \alpha \vec{x}\f$).
   */
  template<typename Precision>
  inline void ScaleOverwrite(index_t length,
      Precision alpha, const Precision *x, Precision *y) {
    mem::Copy(y, x, length);
    Scale<Precision>(length, alpha, y);
  }

  /**
   * Adds a scaled array to an existing array
   * (\f$\vec{y} \gets \vec{y} + \alpha \vec{x}\f$).
   */
  template<typename Precision>
  inline void AddExpert(index_t length,
      Precision alpha, const Precision *x, Precision *y) {
    CppBlas<Precision>::axpy(length, alpha, x, 1, y, 1);
  }

  /**
   * Adds an array to an existing array
   * (\f$\vec{y} \gets \vec{y} + \vec{x}\f$).
   */
  template<typename Precision>
  inline void AddTo(index_t length, const Precision *x, Precision *y) {
    AddExpert<Precision>(length, 1.0, x, y);
  }
  /**
   * Sets an array to the sum of two arrays
   * (\f$\vec{z} \gets \vec{y} + \vec{x}\f$).
   */
  template<typename Precision>
  inline void AddOverwrite(index_t length,
      const Precision *x, const Precision *y, Precision *z) {
    mem::Copy(z, y, length);
    AddTo<Precision>(length, x, z);
  }

  /**
   * Subtracts an array from an existing array
   * (\f$\vec{y} \gets \vec{y} - \vec{x}\f$).
   */
  template<typename Precision>
  inline void SubFrom(index_t length, const Precision *x, Precision *y) {
    AddExpert<Precision>(length, -1.0, x, y);
  }
  /**
   * Sets an array to the difference of two arrays
   * (\f$\vec{x} \gets \vec{y} - \vec{x}\f$).
   */
  template<typename Precision>
  inline void SubOverwrite(index_t length,
      const Precision *x, const Precision *y, Precision *z) {
    mem::Copy(z, y, length);
    SubFrom<Precision>(length, x, z);
  }
#else
  /* --- Equivalent to BLAS level 1 --- */
  /* These functions steal their comments from the previous versons */
  template<typename Precision> 
  inline Precision LengthEuclidean(index_t length, const Precision *x) {
    Precision sum = 0;
    do {
      sum += *x * *x;
      x++;
    } while (--length);
    return sqrt(sum);
  }
  template<typename Precision>
  inline Precision Dot(index_t length, const Precision *x, const Precision *y) {
    Precision sum = 0;
    const Precision *end = x + length;
    while (x != end) {
      sum += *x++ * *y++;
    }
    return sum;
  }

  template<typename Precision>
  inline void Scale(index_t length, Precision alpha, Precision *x) {
    const Precision *end = x + length;
    while (x != end) {
      *x++ *= alpha;
    }
  }
  
  template<typename Precision>
  inline void ScaleOverwrite(index_t length,
      Precision alpha, const Precision *x, Precision *y) {
    const Precision *end = x + length;
    while (x != end) {
      *y++ = alpha * *x++;
    }
  }
  template<typename Precision>
  inline void AddExpert(index_t length,
      Precision alpha, const Precision *x, Precision *y) {
    const Precision *end = x + length;
    while (x != end) {
      *y++ += alpha * *x++;
    }
  }
  
  template<typename Precision>
  inline void AddTo(index_t length, const Precision *x, Precision *y) {
    const Precision *end = x + length;
    while (x != end) {
      *y++ += *x++;
    }
  }

  template<typename Precision>
  inline void AddOverwrite(index_t length,
      const Precision *x, const Precision *y, Precision *z) {
    for (index_t i = 0; i < length; i++) {
      z[i] = y[i] + x[i];
    }
  }
  
  template<typename Precision>
  inline void SubFrom(index_t length, const Precision *x, Precision *y) {
    const Precision *end = x + length;
    while (x != end) {
      *y++ -= *x++;
    }
  }

  template<typename Precision>
  inline void SubOverwrite(index_t length,
      const Precision *x, const Precision *y, Precision *z) {
    for (index_t i = 0; i < length; i++) {
      z[i] = y[i] - x[i];
    }
  }
#endif

  /**
   * Finds the square root of the dot product of a vector or matrix with itself
   * (\f$\sqrt{\vec{x} \cdot \vec{x}}\f$).
   */
  template<typename Precision, bool IsVector>
  inline Precision LengthEuclidean(const GenMatrix<Precision, IsVector> &x) {
    return LengthEuclidean<Precision>(x.length(), x.ptr());
  }

  /**
   * Finds the dot product of two vectors
   * (\f$\vec{x} \cdot \vec{y}\f$).
   */
  template<typename Precision, bool IsVector>
  inline Precision Dot(const GenMatrix<Precision, IsVector> &x, 
                       const GenMatrix<Precision, IsVector> &y) {
    DEBUG_SAME_SIZE(x.length(), y.length());
    return Dot<Precision, IsVector>(x.length(), x.ptr(), y.ptr());
  }
  /* --- Matrix/Vector Scaling --- */

  /**
   * Scales a vector in-place by some factor
   * (\f$\vec{x} \gets \alpha \vec{x}\f$).
   */
  template<typename Precision, bool IsVector>
  inline void Scale(Precision alpha, GenMatrix<Precision, IsVector> *x) {
    Scale<Precision, IsVector>(x->length(), alpha, x->ptr());
  }
 /**
   * Scales each row of the matrix to a different scale.
   *
   * X <- diag(d) * X
   * 
   * @param d a length-M vector with each value corresponding
   * @param X the matrix to scale
   */
  template<typename Precision, bool IsVector1, bool IsVector2 >
  inline void ScaleRows(const GenMatrix<Precision, IsVector1>& d, 
                        GenMatrix<Precision, IsVector2> *X) {
    DEBUG_SAME_SIZE(d.n_rows(), 1);
    DEBUG_SAME_SIZE(d.n_rows(), X->n_rows());
    ScaleRows<Precision>(d.n_rows(), X->n_cols(), d.ptr(), X->ptr());
  }
 /**
   * Sets a vector to another scaled by some factor
   * (\f$\vec{y} \gets \alpha \vec{x}\f$).
   */
  template<typename Precision, MemoryAlloc M, bool IsVector=false>
  inline void Scale(Precision alpha, const GenMatrix<Precision, IsVector>
      &x, GenMatrix<Precision, IsVector> *y) {
    DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
    AllocationTrait<M>::Init(x.n_rows(), y.n_cols(), y);
    Scale(x.length(), alpha, x.ptr(), y->ptr());
  }

  /* --- Scaled Matrix/Vector Addition --- */

  /**
   * Adds a scaled vector to an existing vector
   * (\f$\vec{y} \gets \vec{y} + \alpha \vec{x}\f$).
   */
  template<typename Precision, bool IsVector=false>
  inline void AddExpert(Precision alpha, 
                       const GenMatrix<Precision, IsVector> &x, 
                       GenMatrix<Precision, IsVector> *y) {
    DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
    AddExpert(x.length(), alpha, x.ptr(), y->ptr());
  }
  /* --- Matrix/Vector Addition --- */

  /**
   * Adds a vector to an existing vector
   * (\f$\vec{y} \gets \vec{y} + \vec{x}\f$);
   */
  template<typename Precision, bool IsVector=false>
  inline void AddTo(const GenMatrix<Precision, IsVector> &x, 
                    GenMatrix<Precision, IsVector> *y) {
    DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
    AddTo(x.length(), x.ptr(), y->ptr());
  }

  /**
   * Sets a vector to the sum of two vectors
   * (\f$\vec{z} \gets \vec{y} + \vec{x}\f$).
   */
  template<typename Precision, MemoryAlloc M, bool IsVector=false>
  inline void Add(const GenVector<Precision, IsVector> &x, 
                  const GenVector<Precision, IsVector> &y, 
                        GenMatrix<Precision, IsVector> *z) {
    DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
    AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), z);
    Add(x.length(), x.ptr(), y.ptr(), z->ptr());
  }

  /* --- Matrix/Vector Subtraction --- */

  /**
   * Subtracts a vector from an existing vector
   * (\f$\vec{y} \gets \vec{y} - \vec{x}\f$).
   */
  template<typename Precision, bool IsVector=false>
  inline void SubFrom(const GenVector<Precision, IsVector> &x, 
                      GenVector<Precision, IsVector> *y) {
    DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
    SubFrom(x.length(), x.ptr(), y->ptr());
  }

  /**
   * Sets a vector to the difference of two vectors
   * (\f$\vec{z} \gets \vec{y} - \vec{x}\f$).
   */
  template<typename Precision, MemoryAlloc M, bool IsVector=false>
  inline void SubOverwrite(const GenVector<Precision, IsVector> &x, 
                           const GenVector<Precision, IsVector> &y, 
                           GenVector<Precision, IsVector> *z) {
    DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
    AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), z);
    SubOverwrite<Precision>(x.length(), x.ptr(), y.ptr(), z->ptr());
  }
  /* --- Matrix Transpose --- */

  /*
   * TODO: These could be an order of magnitude faster if we use the
   * cache-efficient Morton layout matrix transpose algoritihm.
   */

  /**
   * Computes a square matrix transpose in-place
   * (\f$X \gets X'\f$).
   */
  template<typename Precision>
  inline void TransposeSquare(GenMatrix<Precision, false> *X) {
    DEBUG_MATSQUARE(*X);
    index_t nr = X->n_rows();
    for (index_t r = 1; r < nr; r++) {
      for (index_t c = 0; c < r; c++) {
        Precision temp = X->get(r, c);
        X->set(r, c, X->get(c, r));
        X->set(c, r, temp);        
      }
    }
  }

  /**
   * Sets a matrix to the transpose of another
   * (\f$Y \gets X'\f$).
   */
  template<typename Precision, MemoryAlloc M>
  inline void Transpose(const GenMatrix<Precision, false> &X, 
                              GenMatrix<Precision, false> *Y) {
    AllocationTrait<M>::Init(X.n_rows(), Y.n_rows(), Y);
    index_t nr = X.n_rows();
    index_t nc = X.n_cols();
    for (index_t r = 0; r < nr; r++) {
      for (index_t c = 0; c < nc; c++) {
        Y->set(c, r, X.get(r, c));
      }
    }
  }

#ifdef USE_LAPACK
  /* --- Wrappers for BLAS level 2 and 3--- */

  /**
   * Scaled matrix-vector multiplication
   * (\f$\vec{y} \gets \alpha A \vec{x} + \beta \vec{y}\f$).
   *
   * @param alpha the scaling factor of A * x
   * @param A an M-by-N matrix
   * @param x an N-length vector to right-multiply by
   * @param beta the scaling factor of y; 0.0 to overwrite
   * @param y an M-length vector to store results (must not be x)
   */
  template<typename Precision, TransMode IsTransA, bool IsVector=false>
  inline void MulExpert(
      Precision alpha, 
      const GenMatrix<Precision, false> &A,
      const GenMatrix<Precision, IsVector> &x, 
      Precision beta, GenVector<Precision, IsVector> *y) { 
    DEBUG_ASSERT(x.ptr() != y->ptr());
    DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
    DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
    if (IsVector==true) {
      if (IsTransA==true) {
        DEBUG_SAME_SIZE(A.n_rows(), x.n_rows());
        CppBlas<Precision>::gemv("T", A.n_rows(), A.n_cols(),
            alpha, A.ptr(), A.n_rows(), x.ptr(), 1,
            beta, y->ptr(), 1);
      } else {
        DEBUG_SAME_SIZE(A.n_cols(), x.n_rows());
        CppBlas<Precision>::gemv("N", A.n_rows(), A.n_cols(),
            alpha, A.ptr(), A.n_rows(), x.ptr(), 1,
            beta, y->ptr(), 1);
      }
    } else {
      if (IsTransA==true) {
        DEBUG_SAME_SIZE(A.n_rows(), x.n_rows());    
        CppBlas<Precision>::gemm("T", "N",
            y->n_rows(), y->n_cols(),  A.n_rows(),
            alpha, A.ptr(), A.n_col(), x.ptr(), x.n_rows(),
            beta, y->ptr(), y->n_rows());
      } else {
        DEBUG_SAME_SIZE(A.n_cols(), x.n_rows()); 
        CppBlas<Precision>::gemm("N", "N",
            y->n_rows(), y->n_cols(),  A.n_rows(),
            alpha, A.ptr(), A.n_rows(), x.ptr(), x.n_rows(),
            beta, y->ptr(), y->n_rows());
      }
    }
  }

  template<typename Precision, TransMode IsTransA, TransMode IsTransB>
  inline void MulExpert(
      Precision alpha, 
      const GenMatrix<Precision, false> &A,
      const GenMatrix<Precision, false> &B, 
      Precision beta, GenVector<Precision, IsVector> *C) { 
    DEBUG_ASSERT(B.ptr() != B->ptr());
    if (IsTransB==true) {
      if (IsTransA==true) {
        DEBUG_SAME_SIZE(A.n_rows(), B.n_cols());
        DEBUG_SAME_SIZE(A.n_cols(), C.n_rows()); 
        DEBUG_SAME_SIZE(C.n_cols(), B.n_rows());
        CppBlas<Precision>::gemm("T", "T",
            C->n_rows(), C->n_cols(),  A.n_rows(),
            alpha, A.ptr(), A.n_col(), B.ptr(), B.n_cols(),
            beta, C->ptr(), C->n_rows());
 
     } else {
        DEBUG_SAME_SIZE(A.n_cols(), B.n_rows());
        DEBUG_SAME_SIZE(A.n_rows(), C.n_rows());    
        DEBUG_SAME_SIZE(C.n_cols(), B.n_rows());
        CppBlas<Precision>::gemm("N", "T",
            C->n_rows(), C->n_cols(),  A.n_cols(),
            alpha, A.ptr(), A.n_col(), B.ptr(), B.n_rows(),
            beta, C->ptr(), C->n_rows());
      }
    } else {
      if (IsTransA==true) {
        DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());    
        DEBUG_SAME_SIZE(A.n_cols(), C.n_rows());
        DEBUG_SAME_SIZE(C.n_cols(), B.n_cols());
        CppBlas<Precision>::gemm("T", "N",
            C->n_rows(), C->n_cols(),  A.n_rows(),
            alpha, A.ptr(), A.n_col(), B.ptr(), B.n_rows(),
            beta, C->ptr(), C->n_rows());
      } else {
        DEBUG_SAME_SIZE(A.n_cols(), B.n_rows());
        DEBUG_SAME_SIZE(A.n_rows(), C.n_rows()); 
        DEBUG_SAME_SIZE(C.n_cols(), B.n_cols());
        CppBlas<Precision>::gemm("N", "N",
            C->n_rows(), C->n_cols(),  A.n_cols(),
            alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
            beta, C->ptr(), C->n_rows());
      }
    }
  }

  /**
   * Sets a vector to the results of matrix-vector multiplication
   * (\f$\vec{y} \gets A\vec{x}\f$).
   *
   * @code
   * Matrix A;
   * Vector x;
   * Vector y;
   * ... // assign A to [0 2; 2 0]
   * ... // assign x to [0 1]
   * ... // y must be initialized and size 2
   * la::MulOverwrite(A, x, &y);
   * ... // y is now [2 0]
   * @endcode
   *
   * @param A an M-by-N matrix
   * @param x an N-length vector to right-multiply by
   * @param y an M-length vector to store results (must not be x)
   */
  template<typename Precision, TransMode IsTransA, TransMode IsTransB, 
           MemoryAlloc M>
  inline void Mul(const GenMatrix<Precision, false> &A, 
                  const GenMatrix<Precision, false> &B, 
                        GenMatrix<Precision, false> *C) {
    AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), y);
    MulExpert<Precision, IsTransA, IsTransB, IsVector>(1.0, A, B, 0.0, C);
  }




  /* --- Wrappers for LAPACK --- */
  /**
   * Destructively computes an LU decomposition of a matrix.
   *
   * Stores L and U in the same matrix (the unitary diagonal of L is
   * implicit).
   *
   * @param pivots a size min(M, N) array to store pivotes
   * @param A_in_LU_out an M-by-N matrix to be decomposed; overwritten
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t PLUExpert(f77_integer *pivots, GenMatrix<Precision, false> *A_in_LU_out) {
    f77_integer info;
    CppLapack<Precision>::getrf(A_in_LU_out->n_rows(), 
        A_in_LU_out->n_cols(),
        A_in_LU_out->ptr(), A_in_LU_out->n_rows(), pivots, &info);
    return SUCCESS_FROM_LAPACK(info);
  }
  /**
   * Init matrices to the LU decomposition of a matrix.
   *
   * @param A the matrix to be decomposed
   * @param pivots a fresh array to be initialized to length min(M,N)
   *        and filled with the permutation pivots
   * @param L a fresh matrix to be initialized to size M-by-min(M,N)
   *        and filled with the lower triangular matrix
   * @param U a fresh matrix to be initialized to size min(M,N)-by-N
   *        and filled with the upper triangular matrix
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t PLUInit(const GenMatrix<Precision, false> &A,
      ArrayList<f77_integer> *pivots, GenMatrix<Precision, false> *L, 
      GenMatrix<Precision, false> *U) {
    index_t m = A.n_rows();
    index_t n = A.n_cols();
    success_t success;

    if (m > n) {
      pivots->Init(n);
      L->Copy(A);
      U->Init(n, n);
      success = PLUExpert(pivots->begin(), L);

      if (!PASSED(success)) {
        return success;
      }

      for (index_t j = 0; j < n; j++) {
        Precision *lcol = L->GetColumnPtr(j);
        Precision *ucol = U->GetColumnPtr(j);

        mem::Copy(ucol, lcol, j + 1);
        mem::Zero(ucol + j + 1, n - j - 1);
        mem::Zero(lcol, j);
        lcol[j] = 1.0;
      }
    } else {
      pivots->Init(m);
      L->Init(m, m);
      U->Copy(A);
      success = PLUExpert(pivots->begin(), U);

      if (!PASSED(success)) {
        return success;
      }

      for (index_t j = 0; j < m; j++) {
        Precision *lcol = L->GetColumnPtr(j);
        Precision *ucol = U->GetColumnPtr(j);

        mem::Zero(lcol, j);
        lcol[j] = 1.0;
        mem::Copy(lcol + j + 1, ucol + j + 1, m - j - 1);
        mem::Zero(ucol + j + 1, m - j - 1);
      }
    }

    return success;
  }


  /**
   * Destructively computes an inverse from a PLU decomposition.
   *
   * @param pivots the pivots array from PLU decomposition
   * @param LU_in_B_out the LU decomposition; overwritten with inverse
   * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t InverseExpert(f77_integer *pivots, GenMatrix<Precision, false> *LU_in_B_out) {
    f77_integer info;
    f77_integer n = LU_in_B_out->n_rows();
    f77_integer lwork = CppLapack<Precision>::getri_block_size * n;
    Precision work[lwork];
    DEBUG_MATSQUARE(*LU_in_B_out);
    CppLapack<Precision>::getri(n, LU_in_B_out->ptr(), n, pivots, work, lwork, &info);
    return SUCCESS_FROM_LAPACK(info);
  }
  /**
   * Inverts a matrix in place
   * (\f$A \gets A^{-1}\f$).
   *
   * @code
   * Matrix A;
   * ... assign A to [4 0; 0 0.5]
   * la::Inverse(&A);
   * ... A is now [0.25 0; 0 2.0]
   * @endcode
   *
   * @param A an N-by-N matrix to invert
   * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t Inverse(GenMatrix<Precision, false> *A) {
    f77_integer pivots[A->n_rows()];

    success_t success = PLUExpert(pivots, A);

    if (!PASSED(success)) {
      return success;
    }

    return InverseExpert(pivots, A);
  }

  
  /**
   * Set a matrix to the inverse of another matrix
   * (\f$B \gets A^{-1}\f$).
   *
   * @code
   * Matrix A;
   * Matrix B;
   * ... assign A to [4 0; 0 0.5]
   * ... B must be initialized and size 2-by-2
   * la::InverseOverwrite(A, &B);
   * ... B is now [0.25 0; 0 2.0]
   * @endcode
   *
   * @param A an N-by-N matrix to invert
   * @param B an N-by-N matrix to store the results
   * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t InverseOverwrite(const GenMatrix<Precision, false> &A, 
      GenMatrix<Precision, false> *B) {
    f77_integer pivots[A.n_rows()];

    if (likely(A.ptr() != B->ptr())) {
      B->CopyValues(A);
    }
    success_t success = PLUExpert(pivots, B);

    if (!PASSED(success)) {
      return success;
    }

    return InverseExpert(pivots, B);
  } 
  
  /**
   * Init a matrix to the inverse of another matrix
   * (\f$B \gets A^{-1}\f$).
   *
   * @param A an N-by-N matrix to invert
   * @param B a fresh matrix to be initialized to size N-by-N
   *        and filled with the inverse
   * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t InverseInit(const GenMatrix<Precision, false> &A, 
                               GenMatrix<Precision, false> *B) {
    B->Init(A.n_rows(), A.n_cols());
    return InverseOverwrite(A, B);
  }

  /**
   * Returns the determinant of a matrix
   * (\f$\det A\f$).
   *
   * @code
   * Matrix A;
   * ... assign A to [4 0; 0 0.5]
   * double det = la::Determinant(A);
   * ... det is equal to 2.0
   * @endcode
   *
   * @param A the matrix to find the determinant of
   * @return the determinant; note long double for large exponents
   */
  template<typename Precision>
  long double Determinant(const GenMatrix<Precision, false> &A) {
    DEBUG_MATSQUARE(A);
    int n = A.n_rows();
    f77_integer pivots[n];
    GenMatrix<Precision> LU;

    LU.Copy(A);
    PLUExpert<Precision>(pivots, &LU);

    long double det = 1.0;

    for (index_t i = 0; i < n; i++) {
      if (pivots[i] != i+1) {
        // pivoting occured (note FORTRAN has 1-based indexing)
        det = -det;
      }
      det *= LU.get(i, i);
    }

  return det;
  }
  /**
   * Returns the log-determinant of a matrix
   * (\f$\ln |\det A|\f$).
   *
   * This is effectively log(fabs(Determinant(A))).
   * To compute log base 10, divide the result by Math::LN_10.
   *
   * @param A the matrix to find the determinant of
   * @param sign_out set to -1, 1, or 0; pass NULL to disable
   * @return the log of the determinant or NaN if A is singular
   */
  template<typename Precision>
  Precision DeterminantLog(const GenMatrix<Precision, false> &A, int *sign_out) {
    DEBUG_MATSQUARE(A);
    int n = A.n_rows();
    f77_integer pivots[n];
    GenMatrix<Precision, false> LU;

    LU.Copy(A);
    PLUExpert<Precision>(pivots, &LU);

    Precision log_det = 0.0;
    int sign_det = 1;

    for (index_t i = 0; i < n; i++) {
      if (pivots[i] != i+1) {
        // pivoting occured (note FORTRAN has one-based indexing)
        sign_det = -sign_det;
      }

      Precision value = LU.get(i, i);

      if (value < 0) {
        sign_det = -sign_det;
        value = -value;
      } else if (!(value > 0)) {
        sign_det = 0;
        log_det = std::numeric_limits<Precision>::quiet_NaN();
        break;
      }

      log_det += log(value);
    }

    if (sign_out) {
      *sign_out = sign_det;
    }

    return log_det;
  }

  /**
   * Destructively solves a system of linear equations (X st A * X = B).
   *
   * This computes the PLU factorization in A as a side effect, but
   * you are free to ignore it.
   *
   * @param pivots a size N array to store pivots of LU decomposition
   * @param A_in_LU_out an N-by-N matrix multiplied by x; overwritten
   *        with its LU decomposition
   * @param k the number of columns on the right-hand side
   * @param B_in_X_out an N-by-K matrix ptr of desired products;
   *        overwritten with solutions (must not alias A_in_LU_out)
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t SolveExpert(
      f77_integer *pivots, GenMatrix<Precision, false> *A_in_LU_out,
      index_t k, Precision *B_in_X_out) {
    DEBUG_MATSQUARE(*A_in_LU_out);
    f77_integer info;
    f77_integer n = A_in_LU_out->n_rows();
    CppLapack<Precision>::gesv(n, k, A_in_LU_out->ptr(), n, pivots,
        B_in_X_out, n, &info);
    return SUCCESS_FROM_LAPACK(info);
  }
  /**
   * Inits a matrix to the solution of a system of linear equations
   * (X st A * X = B).
   *
   * @code
   * Matrix A;
   * Matrix B;
   * ... assign A to [1 3; 2 10]
   * ... assign B to [2 3; 8 10]
   * Matrix X; // Not initialized
   * la::SolveInit(A, B, &X);
   * ... X is now [-1 0; 1 1]
   * Matrix C;
   * la::MulInit(A, X, &C);
   * ... B and C should be equal (but for round-off)
   * @endcode
   *
   * @param A an N-by-N matrix multiplied by x
   * @param B a size N-by-K matrix of desired products
   * @param X a fresh matrix to be initialized to size N-by-K
   *        and filled with solutions
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t SolveInit(const GenMatrix<Precision, false> &A, 
                             const GenMatrix<Precision, false> &B, 
                             GenMatrix<Precision, false> *X) {
    DEBUG_MATSQUARE(A);
    DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
    GenMatrix<Precision, false> tmp;
    index_t n = B.n_rows();
    f77_integer pivots[n];
    tmp.Copy(A);
    X->Copy(B);
    return SolveExpert<Precision>(pivots, &tmp, B.n_cols(), X->ptr());
  }
  /**
   * Inits a vector to the solution of a system of linear equations
   * (x st A * x = b).
   *
   * @code
   * Matrix A;
   * Vector b;
   * ... assign A to [1 3; 2 10]
   * ... assign B to [2 8]
   * Vector x; // Not initialized
   * la::SolveInit(A, b, &x);
   * ... x is now [-1 1]
   * Vector c;
   * la::MulInit(A, x, &c);
   * ... b and c should be equal (but for round-off)
   * @endcode
   *
   * @param A an N-by-N matrix multiplied by x
   * @param b an N-length vector of desired products
   * @param x a fresh vector to be initialized to length N
   *        and filled with solutions
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t SolveInit(const GenMatrix<Precision, false> &A, 
                             const GenMatrix<Precision, true>,  &b, 
                             GenMatrix<Precision, true> *x) {
    DEBUG_MATSQUARE(A);
    DEBUG_SAME_SIZE(A.n_rows(), b.length());
    GenMatrix<Precision, false> tmp;
    index_t n = b.length();
    f77_integer pivots[n];
    tmp.Copy(A);
    x->Copy(b);
    return SolveExpert<Precision>(pivots, &tmp, 1, x->ptr());
  }

  /**
   * Destructively performs a QR decomposition (A = Q * R).
   *
   * Factorizes a matrix as a rotation matrix (Q) times a reflection
   * matrix (R); generalized for rectangular matrices.
   *
   * @param A_in_Q_out an M-by-N matrix to factorize; overwritten with
   *        Q, an M-by-min(M,N) matrix (remaining columns are garbage,
   *        but are not removed from the matrix)
   * @param R a min(M,N)-by-N matrix to store results (must not be
   *        A_in_Q_out)
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t QRExpert(GenMatrix<Precision, false> *A_in_Q_out, 
                     GenMatrix<Precision, false>  *R) {
    f77_integer info;
    f77_integer m = A_in_Q_out->n_rows();
    f77_integer n = A_in_Q_out->n_cols();
    f77_integer k = std::min(m, n);
    f77_integer lwork = n * CppLapack<Precision>::geqrf_dorgqr_block_size;
    Precision tau[k + lwork];
    Precision *work = tau + k;

    // Obtain both Q and R in A_in_Q_out
    CppLapack<Precision>::geqrf(m, n, A_in_Q_out->ptr(), m,
       tau, work, lwork, &info);

    if (info != 0) {
      return SUCCESS_FROM_LAPACK(info);
    }

    // Extract R
    for (index_t j = 0; j < n; j++) {
      Precision *r_col = R->GetColumnPtr(j);
      Precision *q_col = A_in_Q_out->GetColumnPtr(j);
      int i = std::min(j + 1, index_t(k));
      mem::Copy(r_col, q_col, i);
      mem::Zero(r_col + i, k - i);
    }

    // Fix Q
    CppLapack<Precision>::orgqr(m, k, k, A_in_Q_out->ptr(), m,
      tau, work, lwork, &info);

    return SUCCESS_FROM_LAPACK(info);
  
  }
  /**
   * Init matrices to a QR decomposition (A = Q * R).
   *
   * Factorizes a matrix as a rotation matrix (Q) times a reflection
   * matrix (R); generalized for rectangular matrices.
   *
   * @code
   * Matrix A;
   * ... assign A to [3 5; 4 12]
   * Matrix Q;
   * Matrix R;
   * la::QRInit(A, &Q, &R);
   * ... Q is now [-0.6 -0.8; -0.8 0.6]
   * ... R is now [-5.0 -12.6; 0.0 3.2]
   * Matrix B;
   * la::MulInit(Q, R, &B)
   * ... A and B should be equal (but for round-off)
   * @endcode
   *
   * @param A an M-by-N matrix to factorize
   * @param Q a fresh matrix to be initialized to size M-by-min(M,N)
            and filled with the rotation matrix
   * @param R a fresh matrix to be initialized to size min(M,N)-by-N
            and filled with the reflection matrix
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t QRInit(const GenMatrix<Precision, false> &A, 
                   GenMatrix<Precision, false> *Q, 
                   GenMatrix<Precision, false> *R) {
    index_t k = std::min(A.n_rows(), A.n_cols());
    Q->Copy(A);
    R->Init(k, A.n_cols());
    success_t success = QRExpert(Q, R);
    Q->ResizeNoalias(k);

    return success;
  } 
  

  /**
   * Destructive Schur decomposition (A = Z * T * Z').
   *
   * This uses DGEES to find a Schur decomposition, but is also the best
   * way to find just eigenvalues.
   *
   * @param A_in_T_out am N-by-N matrix to decompose; overwritten
   *        with the Schur form
   * @param w_real a length-N array to store real eigenvalue components
   * @param w_imag a length-N array to store imaginary components
   * @param Z an N-by-N matrix ptr to store the Schur vectors, or NULL
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t SchurExpert(GenMatrix<Precision, false> *A_in_T_out,
      Precision *w_real, Precision *w_imag, Precision *Z) {
    DEBUG_MATSQUARE(*A_in_T_out);
    f77_integer info;
    f77_integer n = A_in_T_out->n_rows();
    f77_integer sdim;
    const char *job = Z ? "V" : "N";
    Precision d; // for querying optimal work size

    CppLapack<Precision>::gees(job, "N", NULL,
        n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
        Z, n, &d, -1, NULL, &info);
    {
      f77_integer lwork = (f77_integer)d;
      Precision work[lwork];

      CppLapack<Precision>::gees(job, "N", NULL,
          n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
          Z, n, work, lwork, NULL, &info);
    }

    return SUCCESS_FROM_LAPACK(info);
  }
  
  /**
   * Init matrices to a Schur decompoosition (A = Z * T * Z').
   *
   * @param A an N-by-N matrix to decompose
   * @param w_real a fresh vector to be initialized to length N
   *        and filled with the real eigenvalue components
   * @param w_imag a fresh vector to be initialized to length N
   *        and filled with the imaginary components
   * @param T a fresh matrix to be initialized to size N-by-N
   *        and filled with the Schur form
   * @param Z a fresh matrix to be initialized to size N-by-N
   *        and filled with the Schur vectors
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t SchurInit(const GenMatrix<Precision, false> &A,
      GenMatrix<Precision, true> *w_real, GenMatrix<Precision, true> *w_imag, 
      GenMatrix<Precision, false> *T, 
      GenMatrix<Precision, false> *Z) {
    index_t n = A.n_rows();
    T->Copy(A);
    w_real->Init(n);
    w_imag->Init(n);
    Z->Init(n, n);
    return SchurExpert(T, w_real->ptr(), w_imag->ptr(), Z->ptr());
  }

  /**
   * Destructive, unprocessed eigenvalue/vector decomposition.
   *
   * Real eigenvectors are stored in the columns of V, while imaginary
   * eigenvectors occupy adjacent columns of V with conjugate pairs
   * given by V(:,j) + i*V(:,j+1) and V(:,j) - i*V(:,j+1).
   *
   * @param A_garbage an N-by-N matrix to be decomposed; overwritten
   *        with garbage
   * @param w_real a length-N array to store real eigenvalue components
   * @param w_imag a length-N array to store imaginary components
   * @param V_raw an N-by-N matrix ptr to store eigenvectors, or NULL
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t EigenExpert(GenMatrix<Precision, false> *A_garbage,
      Precision *w_real, Precision *w_imag, Precision *V_raw) {
    DEBUG_MATSQUARE(*A_garbage);
    f77_integer info;
    f77_integer n = A_garbage->n_rows();
    const char *job = V_raw ? "V" : "N";
    Precision d; // for querying optimal work size

    CppLapack<Precision>::geev("N", job, n, A_garbage->ptr(), n,
        w_real, w_imag, NULL, 1, V_raw, n, &d, -1, &info);
    {
      f77_integer lwork = (f77_integer)d;
      Precision work[lwork];

      CppLapack<Precision>::geev("N", job, n, A_garbage->ptr(), n,
          w_real, w_imag, NULL, 1, V_raw, n, work, lwork, &info);
    }

    return SUCCESS_FROM_LAPACK(info);
  }

  /**
   * Inits vectors to the eigenvalues of a matrix.
   *
   * @param A an N-by-N matrix to find eigenvalues for
   * @param w_real a fresh vector to be initialized to length N
   *        and filled with the real eigenvalue components
   * @param w_imag a fresh vector to be initialized to length N
   *        and filled with the imaginary components
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   **/
  template<typename Precision>
  inline success_t EigenvaluesInit(const GenMatrix<Precision, false> &A,
      GenMatrix<Precision, true> *w_real, GenMatrix<Precision, true> *w_imag) {
    DEBUG_MATSQUARE(A);
    int n = A.n_rows();
    w_real->Init(n);
    w_imag->Init(n);
    GenMatrix<Precision, false> tmp;
    tmp.Copy(A);
    return SchurExpert<Precision>(&tmp, w_real->ptr(), w_imag->ptr(), NULL);
  }
  /**
   * Inits a vector to the real eigenvalues of a matrix.
   *
   * Complex eigenvalues will be set to NaN.
   *
   * @code
   * Matrix A;
   * ... assign A to [0 1; 1 0]
   * Vector w; // Not initialized
   * la::EigenvaluesInit(A, &w);
   * ... w is now [-1 1]
   * @endcode
   *
   * @param A an N-by-N matrix to find eigenvalues for
   * @param w a fresh vector to be initialized to length N
   *        and filled with the real eigenvalue components
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   **/
  template<typename Precision>
  success_t EigenvaluesInit(const GenMatrix<Precision, false> &A, 
                            GenMatrix<Precision, true> *w) {
    DEBUG_MATSQUARE(A);
    int n = A.n_rows();
    w->Init(n);
    Precision w_imag[n];

    GenMatrix<Precision, false> tmp;
    tmp.Copy(A);
    success_t success = SchurExpert<Precision>(&tmp, w->ptr(), w_imag, NULL);

    if (!PASSED(success)) {
      return success;
    }

    for (index_t j = 0; j < n; j++) {
      if (unlikely(w_imag[j] != 0.0)) {
        (*w)[j] = std::numeric_limits<Precision>::quiet_NaN();
      }  
    }
    return success;
  } 
  

  /**
   * Inits vectors and matrices to the eigenvalues/vectors of a matrix.
   *
   * Complex eigenvalues/vectors are processed into components rather
   * than raw conjugate form.
   *
   * @code
   * Matrix A;
   * ... assign A to [0 1; -1 0]
   * Vector w_real, w_imag; // Not initialized
   * Matrix V_real, V_imag; // Not initialized
   * la::EigenvectorsInit(A, &w_real, &w_imag, &V_real, &V_imag);
   * ... w_real and w_imag are [0 0] + i*[1 -1]
   * ... V_real and V_imag are [0.7 0.7; 0 0] + i*[0 0; 0.7 -0.7]
   * @endcode
   *
   * @param A an N-by-N matrix to be decomposed
   * @param w_real a fresh vector to be initialized to length N
   *        and filled with the real eigenvalue components
   * @param w_imag a fresh vector to be initialized to length N
   *        and filled with the imaginary eigenvalue components
   * @param V_real a fresh matrix to be initialized to size N-by-N
   *        and filled with the real eigenvector components
   * @param V_imag a fresh matrix to be initialized to size N-by-N
   *        and filled with the imaginary eigenvector components
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t EigenvectorsInit(const GenMatrix<Precision, false> &A,
      GenMatrix<Precision, true> *w_real, GenMatrix<Precision, true> *w_imag, 
      GenMatrix<Precision, false> *V_real, GenMatrix<Precision, false> *V_imag) {
    DEBUG_MATSQUARE(A);
    index_t n = A.n_rows();
    w_real->Init(n);
    w_imag->Init(n);
    V_real->Init(n, n);
    V_imag->Init(n, n);

    GenMatrix<Precision, false> tmp;
    tmp.Copy(A);
    success_t success = EigenExpert(&tmp,
        w_real->ptr(), w_imag->ptr(), V_real->ptr());

    if (!PASSED(success)) {
      return success;
    }

    V_imag->SetZero();
    for (index_t j = 0; j < n; j++) {
      if (unlikely(w_imag->get(j) != 0.0)) {
        Precision *r_cur = V_real->GetColumnPtr(j);
        Precision *r_next = V_real->GetColumnPtr(j+1);
        Precision *i_cur = V_imag->GetColumnPtr(j);
        Precision *i_next = V_imag->GetColumnPtr(j+1);

        for (index_t i = 0; i < n; i++) {
          i_next[i] = -(i_cur[i] = r_next[i]);
          r_next[i] = r_cur[i];
        }

        j++; // skip paired column
      }
    }

    return success;
 
  }

  /**
   * Inits a vector and matrix to the real eigenvalues/vectors of a matrix.
   *
   * Complex eigenvalues are set to NaN; their associated vectors are
   * left raw (and are effectively garbage).
   *
   * @param A an N-by-N matrix to be decomposed
   * @param w a fresh vector to be initialized to length N
   *        and filled with the real eigenvalues
   * @param V a fresh matrix to be initialized to size N-by-N
   *        and filled with the real eigenvectors
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t EigenvectorsInit(const GenMatrix<Precision, false> &A, 
                             GenMatrix<Precision, true> *w, 
                             GenMatrix<Precision, false> *V) {
    DEBUG_MATSQUARE(A);
    index_t n = A.n_rows();
    w->Init(n);
    Precision w_imag[n];
    V->Init(n, n);

    GenMatrix<Precision, false> tmp;
    tmp.Copy(A);
    success_t success = EigenExpert(&tmp, w->ptr(), w_imag, V->ptr());

    if (!PASSED(success)) {
      return success;
    }

    for (index_t j = 0; j < n; j++) {
      if (unlikely(w_imag[j] != 0.0)) {
        (*w)[j] = std::numeric_limits<Precision>::quiet_NaN();
      }
    }

    return success;
  
  }

/**
   * Generalized eigenvalue/vector decomposition of two symmetric matrices.
   *
   * Compute all the eigenvalues, and optionally, the eigenvectors
   * of a real generalized eigenproblem, of the form
   * A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x, where
   * A and B are assumed to be symmetric and B is also positive definite.
   *
   * @param itype an integer that specifies the	problem	type to	be solved:
   *        = 1: A*x = (lambda)*B*x
   *        = 2: A*B*x = (lambda)*x
   *        = 3: B*A*x = (lambda)*x
   * @param A_eigenvec an N-by-N matrix to be decomposed; overwritten
   *        with the matrix of eigenvectors
   * @param B_chol an N-by-N matrix to be decomposed; overwritten
   *        with triangular factor U or L from the Cholesky factorization
   *        B=U**T*U or B=L*L**T.
   * @param w a length-N array of eigenvalues in ascending order.
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t GenEigenSymmetric(int itype, GenMatrix<Precision, false> *A_eigenvec, 
                              GenMatrix<Precision, false> *B_chol, Precision *w) {
    DEBUG_MATSQUARE(*A_eigenvec);
    DEBUG_MATSQUARE(*B_chol);
    DEBUG_ASSERT(A_eigenvec->n_rows()==B_chol->n_rows());
    f77_integer itype_f77 = itype;
    f77_integer info;
    f77_integer n = A_eigenvec->n_rows();
    const char *job = "V"; // Compute eigenvalues and eigenvectors.
    Precision d; // for querying optimal work size

    CppLapack<Precision>::sygv(&itype_f77, job, "U", n, A_eigenvec->ptr(), n, B_chol->ptr(), n, w,
        &d, -1, &info);
    {
      f77_integer lwork = (f77_integer)d;
      Precision work[lwork];

      CppLapack<Precision>::sygv(&itype_f77, job, "U", n, A_eigenvec->ptr(), n, B_chol->ptr(), n, w,
          work, lwork, &info);
    }

    return SUCCESS_FROM_LAPACK(info);

  }

  /**
   * Generalized eigenvalue/vector decomposition of two nonsymmetric matrices.
   *
   * Compute all the eigenvalues, and optionally, the eigenvectors
   * of a real generalized eigenproblem, of the form
   * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x
   *
   * (alpha_real(j) + alpha_imag(j)*i)/beta(j), j=1,...,N, will be the 
   * generalized eigenvalues.
   *
   * Note: the quotients alpha_real(j)/beta(j) and alpha_imag(j)/beta(j) may
   * easily over- or underflow, and beta(j) may even be zero.  Thus, the
   * user should avoid naively computing the ratio alpha/beta.  However,
   * alpha_real and alpha_imag will be always less than and usually comparable
   * with norm(A) in magnitude, and beta always less than and usually
   * comparable with norm(B).
   *
   * Real eigenvectors are stored in the columns of V, while imaginary
   * eigenvectors occupy adjacent columns of V with conjugate pairs
   * given by V(:,j) + i*V(:,j+1) and V(:,j) - i*V(:,j+1).
   *
   * @param A_garbage an N-by-N matrix to be decomposed; overwritten
   *        with garbage
   * @param B_garbage an N-by-N matrix to be decomposed; overwritten
   *        with garbage
   * @param alpha_real a length-N array
   * @param alpha_imag a length-N array
   * @param beta a length-N array
   * @param V_raw an N-by-N matrix ptr to store eigenvectors, or NULL
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t GenEigenNonSymmetric(GenMatrix<Precision, false> *A_garbage, 
                                 GenMatrix<Precision, false> *B_garbage,
                                 Precision *alpha_real, Precision *alpha_imag, 
                                 Precision *beta, Precision *V_raw) {
    DEBUG_MATSQUARE(*A_garbage);
    DEBUG_MATSQUARE(*B_garbage);
    DEBUG_ASSERT(A_garbage->n_rows()==B_garbage->n_rows());
    f77_integer info;
    f77_integer n = A_garbage->n_rows();
    const char *job = V_raw ? "V" : "N";
    Precision d; // for querying optimal work size

    CppLapack<Precision>::gegv("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n, 
        alpha_real, alpha_imag, beta, NULL, 1, V_raw, n, &d, -1, &info);
    {
      f77_integer lwork = (f77_integer)d;
      Precision work[lwork];

      CppLapack<Precision>::gegv("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n, 
          alpha_real, alpha_imag, beta, NULL, 1, V_raw, n, work, lwork, &info);
    }

  return SUCCESS_FROM_LAPACK(info);
 
  }

  /**
   * Destructive SVD (A = U * S * VT).
   *
   * Finding U and VT is optional (just pass NULL), but you must solve
   * either both or neither.
   *
   * @param A_garbage an M-by-N matrix to be decomposed; overwritten
   *        with garbage
   * @param s a length-min(M,N) array to store the singluar values
   * @param U an M-by-min(M,N) matrix ptr to store left singular
   *        vectors, or NULL for neither U nor VT
   * @param VT a min(M,N)-by-N matrix ptr to store right singular
   *        vectors, or NULL for neither U nor VT
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t SVDExpert(GenMatrix<Precision, false>* A_garbage, 
                      Precision *s, 
                      Precision *U, 
                      Precision *VT) {
    DEBUG_ASSERT_MSG((U == NULL) == (VT == NULL),
                   "You must fill both U and VT or neither.");
    f77_integer info;
    f77_integer m = A_garbage->n_rows();
    f77_integer n = A_garbage->n_cols();
    f77_integer k = std::min(m, n);
    f77_integer iwork[8 * k];
    const char *job = U ? "S" : "N";
    Precision d; // for querying optimal work size

    CppLapack<Precision>::gesdd(job, m, n, A_garbage->ptr(), m,
        s, U, m, VT, k, &d, -1, iwork, &info);
    {
      f77_integer lwork = (f77_integer)d;
      // work for DGESDD can be large, we really do need to malloc it
      Precision *work = mem::Alloc<Precision>(lwork);

      CppLapack<Precision>::gesdd(job, m, n, A_garbage->ptr(), m,
          s, U, m, VT, k, work, lwork, iwork, &info);

      mem::Free(work);
    }

    return SUCCESS_FROM_LAPACK(info);

  }

  /**
   * Inits a vector to the singular values of a matrix.
   *
   * @param A an N-by-N matrix to find the singular values of
   * @param s a fresh vector to be initialized to length N
   *        and filled with the singular values
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t SVDInit(const GenMatrix<Precision, false> &A, 
                           GenMatrix<Precision, true> *s) {
    s->Init(std::min(A.n_rows(), A.n_cols()));
    GenMatrix<Precision, false> tmp;
    tmp.Copy(A);
    return SVDExpert<Precision>(&tmp, s->ptr(), NULL, NULL);
  }

  /**
   * Inits a vector and matrices to a singular value decomposition
   * (A = U * S * VT).
   *
   * @code
   * Matrix A;
   * ... assign A to [0 1; -1 0]
   * Vector s;     // Not initialized
   * Matrix U, VT; // Not initialized
   * la::SVDInit(A, &s, &U, &VT);
   * ... s is now [1 1]
   * ... U is now [0 1; 1 0]
   * ... V is now [-1 0; 0 1]
   * Matrix S;
   * S.Init(s.length(), s.length());
   * S.SetDiagonal(s);
   * Matrix tmp, result;
   * la::MulInit(U, S, &tmp);
   * la::MulInit(tmp, VT, &result);
   * ... A and result should be equal (but for round-off)
   * @endcode
   *
   * @param A an M-by-N matrix to be decomposed
   * @param s a fresh vector to be initialized to length min(M,N)
   *        and filled with the singular values
   * @param U a fresh matrix to be initialized to size M-by-min(M,N)
   *        and filled with the left singular vectors
   * @param VT a fresh matrix to be initialized to size min(M,N)-by-N
   *        and filled with the right singular vectors
   * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t SVDInit(const GenMatrix<Precision, false> &A, 
                           GenMatrix<Precision, true> *s, 
                           GenMatrix<Precision, false> *U, 
                           GenMatrix<Precision, false> *VT) {
    index_t k = std::min(A.n_rows(), A.n_cols());
    s->Init(k);
    U->Init(A.n_rows(), k);
    VT->Init(k, A.n_cols());
    GenMatrix<Precision, false> tmp;
    tmp.Copy(A);
    return SVDExpert(&tmp, s->ptr(), U->ptr(), VT->ptr());
  }

  /**
   * Destructively computes the Cholesky factorization (A = U' * U).
   *
   * @param A_in_U_out an N-by-N matrix to factorize; overwritten
   *        with result
   * @return SUCCESS_PASS if the matrix is symmetric positive definite,
   *         SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  success_t Cholesky(GenMatrix<Precision, false> *A_in_U_out) {
    DEBUG_MATSQUARE(*A_in_U_out);
    f77_integer info;
    f77_integer n = A_in_U_out->n_rows();

    CppLapack<Precision>::potrf("U", n, A_in_U_out->ptr(), n, &info);

    /* set the garbage part of the matrix to 0. */
    for (f77_integer j = 0; j < n; j++) {
      mem::Zero(A_in_U_out->GetColumnPtr(j) + j + 1, n - j - 1);
    }

    return SUCCESS_FROM_LAPACK(info);
  }
  

  /**
   * Inits a matrix to the Cholesky factorization (A = U' * U).
   *
   * @code
   * Matrix A;
   * ... assign A to [1 1; 1 2]
   * Matrix U; // Not initialized
   * la::CholeskyInit(A, &U);
   * ... U is now [1 1; 0 1]
   * Matrix result;
   * la::MulTransAInit(U, U, result);
   * ... A and result should be equal (but for round-off)
   * @endcode
   *
   * @param A an N-by-N matrix to factorize
   * @param U a fresh matrix to be initialized to size N-by-N
   *        and filled with the factorization
   * @return SUCCESS_PASS if the matrix is symmetric positive definite,
   *         SUCCESS_FAIL otherwise
   */
  template<typename Precision>
  inline success_t CholeskyInit(const GenMatrix<Precision, false> &A, 
                                      GenMatrix<Precision, false> *U) {
    U->Copy(A);
    return Cholesky(U);
  }
#endif

  /*
   * TODO:
   *   symmetric eigenvalue
   *   http://www.netlib.org/lapack/lug/node71.html
   *   dgels for least-squares problems
   *   dgesv for linear equations
   */
};

#endif
