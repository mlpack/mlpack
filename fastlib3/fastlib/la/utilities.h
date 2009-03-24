/**
 * @file utilities.h
 *
 * Matlab like utilities for working with matrics
 *
 * See uselapack.h for more linear algebra routines.
 */

#ifndef FASTLIB_LA_UTILITIES_H
#define FASTLIB_LA_UTILITIES_H

#include "matrix.h"
#include "uselapack.h"
#include "fastlib/math/math_lib.h"
#include <math.h>

namespace la {
  enum MemoryAlloc {Init, Overwrite};

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
   * Entrywise element multiplication of two memory blocks
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ C \gets A .* B\f$).
   */
  template<typename Precision >
  void DotMul(const index_t length,
              const Precision *A, 
              const Precision *B, 
              Precision *C) {
    CppBlas<Precision>::gbmv("N", 
        length, length, 0, 0, 1, A, 1, B, 1, 0, C, 1);
  }
 
  /**
   * Entrywise element multiplication of two matrices
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ C \gets A .* B\f$).
   */
 
  template<typename Precision,  MemoryAlloc MemeAlloc>
  void DotMul(const GenMatrix<Precision> &A, 
              const GenMatrix<Precision> &B, 
              GenMatrix<Precision> *C) {
    DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
    DEBUG_SAME_SIZE(A.n_cols(), B.n_cols());
    AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), &C);
    DotMul<Precision>(A.n_elements(), A.ptr(), B.ptr(), C.ptr());  
  }
 
  /**
   * Entrywise element multiplication of two vectors
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ \vec{C} \gets \vec{A} .* \vec{B}\f$).
   */ 
  template<typename Precision, MemoryAlloc MemAlloc>
  void DotMul(const GenVector<Precision> &A, 
              const GeVector<Precision> &B, 
              GenVector<Precision> *C) {
    DEBUG_SAME_SIZE(A.length(), B.length());
    AllocationTrait<MemAlloc>::Init(A.length(), &C);
    DotMul<Precision>(A.length(), A.ptr(), B.ptr(), C.ptr());  
  
  }

  /**
   * Entrywise element multiplication of two memory blocks
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ A \gets A .* B\f$).
   */
  template<typename Precision>
  void DotMulTo(const index_t length,
              const Precision *A, 
              const Precision *B) {
    
    GenVector<Precision> C;
    C.Init(length);
    CppBlas<Precision>::gbmv("N", 
        length, length, 0, 0, 1, A, 1, B, 1, 0, C.ptr(), 1);
    memcopy(A, C.ptr(), length*sizeof(Precision));
  }
  /**
   * Entrywise element multiplication of two matrices
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ A \gets A .* B\f$).
   */
 
  template<typename Precision, MemoryAlloc MemAlloc>
  void DotMulTo(const GenMatrix<Precision> *A, 
              GenMatrix<Precision> &B) {
    
    DEBUG_SAME_SIZE(A->rows(), B.rows());
    DEBUG_SAME_SIZE(A->cols(), B.cols());
    DotMulTo(A->n_elements(), A->ptr(), B.ptr());
  }
  /**
   * Entrywise element multiplication of two vectors
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ \vec{A} \gets \vec{A} .* \vec{B}\f$).
   */ 
  template<typename Precision, MemoryAlloc MemAlloc>
  void DotMulTo(const GenVector<Precision> *A, 
              GeVector<Precision> &B) {
    DEBUG_SAME_SIZE(A->length(), B.length());
    DotMulTo(A.length(), A->ptr(), B.ptr();) 
  }

  /**
   * Elementwise integer powers of memory blocks
   * (\f$ B \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPow(const index_t length,
              index_t power,
              const Precision *A,
              Precision *B) {
    memcpy(B, A, length*sizeof(Precision))
    for(index_t i=0; i<power; i++) {
      DotMulTo(length, A, B)
    }
  } 
  
  /**
   * Elementwise integer powers of matrices
   * (\f$ B \gets A.\^n \f$)
   */
  template<typename Precision, MemoryAlloc MemAlloc>
  void DotIntPowInit(index_t power,
              const GenMatrix<Precision> &A,
              GenMatrix<Precision> *B) {
    AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), B);
    DotIntPow<Precision>(A.n_elements, A.ptr(), B->ptr());
  } 

  /**
   * Elementwise integer powers of vectors
   * (\f$ \vec{B} \gets \vec{A}.\^n \f$)
   */
  template<typename Precision, M>
  void DotIntPowInit(index_t power,
              const GenVector<Precision> &A,
              GenVector<Precision> *B) {
  
    AllocationTrait<MemAlloc>::Init(A.length(), B);
    DotIntPow<Precision>(A.length(), A.ptr(), B->ptr());
  }


  /**
   * Elementwise integer powers of memory blocks
   * (\f$ A \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowTo(const index_t length,
              index_t power,
              Precision *A) {
    GenVector<Precision> temp;
    temp.Init(length);
    DotIntPow<Precision>(length, power, A, temp);
    memcpy(A, temp, length*sizeof(Precision));
  } 
  
 /**
   * Elementwise integer powers of matrices
   * (\f$ A \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowTo(index_t power,
              GenMatrix<Precision> *A) {
    DotIntPow<Precision>(A->n_elements(), power, A->ptr());
  } 

  /**
   * Elementwise integer powers of vectors
   * (\f$ \vec{A} \gets \vec{A}.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowTo(index_t power,
              GenVector<Precision> *A) {
    DotIntPow<Precision>(A->length(), power, A->ptr());  
  }

  /**
   * In this section we provide some basic object functions for matrix 
   * pricessing
   */
  template<Precision>
  class Sin {
    public: 
     Sin() {
       frequency_=1.0;
     }
     void Init(Precision frequency) {
       frequency_ = frequency;
     } 
     void set(Precision frequency){
       frequency_ = frequency;
     }
     Precision operator()(Precision x) {
       return sin(frequency * x);
     }
    private:
     Precision frequency_;
  }

  template<Precision>
  class Cos {
    public: 
     Cos() {
       frequency_=1.0;
     }
     void Init(Precision frequency) {
       frequency_ = frequency;
     } 
     void set(Precision frequency){
       frequency_ = frequency;
     }
     Precision operator()(Precision x) {
       return cos(frequency_ * x);
     }
    private:
     Precision frequency_;
  }

  template<Precision>
  class Tan {
    public: 
     Tan() {
       frequency_=1.0;
     }
     void Init(Precision frequency) {
       frequency_ = frequency;
     } 
     void set(Precision frequency){
       frequency_ = frequency;
     }
     Precision operator()(Precision x) {
       return tan(frequency_ * x);
     }
    private:
     Precision frequency_;
  }

  template<Precision>
  class Tan {
    public: 
     Tan() {
       frequency_=1.0;
     }
     void Init(Precision frequency) {
       frequency_ = frequency;
     } 
     void set(Precision frequency){
       frequency_ = frequency;
     }
     Precision operator()(Precision x) {
       return tan(frequency_ * x);
     }
    private:
     Precision frequency_;
  }

  template<Precision>
  class Exp {
    public: 
     Exp() {
       alpha_=1.0;
     }
     void Init(Precision alpha) {
       alpha_ = alpha;
     } 
     void set(Precision alpha){
       alpha_ = alpha;
     }
     Precision operator()(Precision x) {
       return exp(alpha_ * x);
     }
    private:
     Precision alpha_;
  }

  template<Precision>
  class Log {
    public: 
     Log() {
       alpha_=1.0;
     }
     void Init(Precision alpha) {
       alpha_ = alpha;
     } 
     void set(Precision alpha){
       alpha_ = alpha;
     }
     Precision operator()(Precision x) {
       DEBUG_ASSERT(x>0);
       return log(alpha_ * x);
     }
    private:
     Precision alpha_;
  }

  template<Precision>
  class Pow {
    public: 
     Pow() {
       alpha_=1.0;
     }
     void Init(Precision alpha) {
       alpha_ = alpha;
     } 
     void set(Precision alpha){
       alpha_ = alpha;
     }
     Precision operator()(Precision x) {
       DEBUG_ASSERT(x>0);
       return pow(alpha_, x);
     }
    private:
     Precision alpha_;
  }

   /**
   * Elementwise fun of a memory block
   * (\f$ B= fun(A) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */

  template<typename Precision, typename Function>
  void DotFun(const index_t length,
              Function &fun,
              const Precision *A, 
              Precision *B) {
  
    for(index_t i=0; i<length; i++) {
      B[i]=fun(A[i]);
    }
  }
  /**
   * Elementwise fun of a matrix
   * (\f$ B= \fun(A) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, MemoryAlloc MemAlloc, typename Function>
  void DotFun(Function &fun
              const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B) {
    AllocatorTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), B);
    DotFun<Precision, Function>(A.n_elements(), fun, A.ptr(), B->ptr());
  }

  /**
   * Elementwise fun of a vector
   * (\f$ \vec{B}= \fun(\vec{A}) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, MemoryAlloc MemAlloc, typename Function>
  void DotFun(Function &fun,              
              const GenVector<Precision> &A, 
              GenVector<Precision> *B) {
    AllocatorTrait<MemAlloc>::Init(A.length(), B);
    DotFun<Precision, Function>(A.length(), fun, A.ptr(), B->ptr());
  }

 /**
   * Elementwise fun of  memory blocks
   * (\f$ C= fun(A, B) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFun(const index_t length,
              Function &fun,
              const Precision *A, 
              const Precision *B, 
              Precision *C) {
    for(index_t i=0; i<length; i++) {
      C[i]=fun(A[i], B[i]);
    }
  }
  /**
   * Elementwise fun of two matrices
   * (\f$ C= \fun(A, B) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, MemoryAlloc MemAlloc, typename Function>
  void DotFun(Function &fun
              const GenMatrix<Precision> &A, 
              const GenMatrix<Precision> &B,
              const GenMatrix<Precision> *C) {
    DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
    DEBUG_SAME_SIZE(A.n_cols(), B.n_cols());
    AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), C);
    DotFun<Precision, Function>(A.n_elements(), 
        fun, A.ptr(), B.ptr(), C->ptr());
  }

  /**
   * Elementwise fun of two vectors
   * (\f$ \vec{C}= \fun(\vec{A}, \vec{B}) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFun(Function &fun,              
              const GenVector<Precision> &A, 
              const GenVector<Precision> &B, 
              GenVector<Precision> *C) {
    DEBUG_SAME_SIZE(A.length(), B.length());
    DotFun(A.length(), fun, A.ptr(), B.ptr(), C->ptr());
  }


  /**
   * Sums the elements of a memory block
   * (\f$ sum(A)\f$)
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision>
  RetPrecision Sum(const index_t length, 
      const Precision *A) {
    RetPrecision sum=0;
    for(index_t i=0; i<length; i++) {
      sum+=A[i];
    }
    return sum;
  }

  /**
   * it sums all the elements of matrix A
   * (\f$ sum(sum(A))\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  RetPrecision Sum(const GenMatrix<Precision> &A) {
    return Sum<Precision, RetPrecision>(A.n_elements(), A.ptr());  
  }
  
  /**
   * it sums all the columns of matrix A
   * (\f$ sum(A)\f$)
   *
   */ 
  template<typename Precision, MemoryAlloc MemAlloc, typename RetPrecision>
  void SumCols(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_sums) {
    AllocatorTrait<MemAlloc>::Init(A.n_cols(), col_sums);    
    RetPrecision *ptr=col_sums->ptr();
    for(index_t i=0; i<A.n_cols(); i++) {
      ptr[i]=Sum<Precision, RetPrecision>(A.n_rows(),
          A.GetColumnPtr(i));
    }
  }

  /**
   * it sums all the rows of matrix A
   * (\f$ sum(A,2)\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  void SumRows(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *row_sums) {
    AllocatorTrait<MemAlloc>::Init(A.n_rows(), row_sums);    
    RetPrecision *ptr=row_sums->ptr();
    for(index_t i=0; i<A.n_rows(); i++) {
      ptr[i]=0;
      for(index_t j=0; j<A.n_cols(); j++) {
        ptr[i]+=A.get(i,j);
      }        
    }
  }
  
  /**
   * Sums the elements of a Vector
   * (\f$ sum(\vec{A})\f$)
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision>
  RetPrecision RetPrecision Sum(const GenVector<Precision> &A) {    
    return Sum<Precison, RetPrecision>(A.length(), A.ptr());
  }

  /**
   * Multiplies the elements of a memory block
   * (\f$ prod(A) \f$)
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision>
  RetPrecision Prod(const index_t length, 
      const Precision *A) {
    RetPrecision prod=1;
    for(index_t i=0; i<length; i++) {
      prod*=A[i];
    }
    return sum;
  }

  /**
   * it  multiplies all the elements of matrix A
   * (\f$ prod(prod(A))\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  RetPrecision Prod(const GenMatrix<Precision> &A) {
    return Prod<Precision, RetPrecision>(A.n_elements(), A.ptr());  
  }
  
  /**
   * it prods all the columns of matrix A
   * (\f$ prod(A)\f$)
   *
   */ 
  template<typename Precision, MemoryAlloc MemAlloc, typename RetPrecision>
  void ProdCols(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_prods) {
    AllocatorTrait<MemAlloc>::Init(A.n_cols(), col_sums);    
    RetPrecision *ptr=col_prods->ptr();
    for(index_t i=0; i<A.n_cols(); i++) {
      ptr[i]=Prod<Precision, RetPrecision>(A.n_rows(),
          A.GetColumnPtr(i));
    }
  }

  /**
   * it multiplies all the rows of matrix A
   * (\f$ prod(A,2)\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  void ProdRows(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *row_prods) {
    AllocatorTrait<MemAlloc>::Init(A.n_rows(), row_sums);    
    RetPrecision *ptr=row_prods->ptr();
    for(index_t i=0; i<A.n_prods(); i++) {
      ptr[i]=1;
      for(index_t j=0; j<A.n_cols(); j++) {
        ptr[i]*=A.get(i,j);
      }        
    }
  }
  
  /**
   * Multiplies the elements of a Vector
   * (\f$ prod(\vec{A})\f$)
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the prod. 
   */
  template<typename Precision, typename RetPrecision>
  RetPrecision RetPrecision Prod(const GenVector<Precision> &A) {    
    return Prod<Precison, RetPrecision>(A.length(), A.ptr());
  }

  /**
   * it sums all the elements of matrix A preprocessed with a function
   * (\f$ sum(sum(fun(A)))\f$)
   * Class Function is a function object so it must implement
   * the operator()
   *
   */ 
  template<typename Precision, typename RetPrecision, typename Function>
  RetPrecision FunSum(const index_t length,
      Function *fun, 
      const GenMatrix<Precision> &A) {
    RetPrecision sum=0;
    for(index_t i=0; i<length; i++) {
      sum+=fun(A[i]);
    }
    return sum;

  }
  
  /**
   * it sums all the columns of matrix A preprocessed with a function
   * (\f$ sum(fun(A))\f$)
   * Class Function is a function object so it must implement
   * the operator() 
   */ 
  template<typename Precision, typename RetPrecision, 
           MemoryAllocator MemAlloc, typename Function>
  RetPrecision FunSumCols(Function &fun, 
      const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_sums) {
    AllocatorTrait<MemAlloc>::Init(A.n_cols(), col_sums);    
    RetPrecision *ptr=col_sums->ptr();
    for(index_t i=0; i<A.n_cols(); i++) {
      ptr[i]=FunSum<Precision, RetPrecision, MemAlloc, Function>(A.n_rows(),
          fun, A.GetColumnPtr(i));
    } 
  }

  template<typename Precision, typename RetPrecision, 
           MemoryAllocator MemAlloc, typename Function>
  RetPrecision FunSumRows(Function &fun, 
                          const GenMatrix<Precision> &A,
                          GenVector<RetPrecision> *rows_sums) {
    
    AllocatorTrait<MemAlloc>::Init(A.n_rows(), row_sums);    
    RetPrecision *ptr=row_sums->ptr();
    for(index_t i=0; i<A.n_rows(); i++) {
      ptr[i]=0;
      for(index_t j=0; j<A.n_cols(); j++) {
        ptr[i]+=fun(A.get(i,j));
      }        
    }
  }
 

  /**
   * Sums the elements of a Vector preprocessed with a function
   * (\f$ sum(\vec{A}))
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision, typename Function>
  void RetPrecision FunSum(Function &fun,                      
                           const GenVector<Precision> &A) {
   return FunSum<Precison, RetPrecision, Function>(A.length(), fun, A.ptr());

  }



  /**
   *  Matlab like utility function for returning a matrix initialized
   *  with zeros
   *
   */
  template<typename Precision>
  void Zeros(const index_t rows, 
             consr index_t cols, 
             GenMatrix<Precision> *A) {
    A->Init(rows, cols, A);
    A->SetAll(Precision(0.0));
  }

  /**
   *  Matlab like utility function for returning a vector initialized
   *  with zeros
   *
   */
  template<typename Precision>
  void Zeros(const index_t length 
             GenVector<Precision> *A) {
    A->Init(length);
    A->SetAll(0.0);
  }

  /**
   *  Matlab like utility function for returning a matrix initialized
   *  with ones
   *
   */
  template<typename Precision>
  void Ones(const index_t rows, 
            const index_t cols, 
            GenMatrix<Precision> *A) {
     A->Init(rows, cols, A);
     A->SetAll(Precision(1.0));
  }

  /**
   *  Matlab like utility function for returning a vector initialized
   *  with ones
   *
   */
  template<typename Precision>
  void Ones(const index_t length, 
             GenVector<Precision> *A) {
    A->Init(length);
    A->SetAll(1.0);
  }

  /**
   *  Matlab like utility function for returning a matrix initialized
   *  with random numbers in [lo, hi] interval
   *
   */
  template<typename Precision>
  void Rand(const index_t rows, 
            const index_t cols,
            const index_t lo,
            const index_t hi,
            GenMatrix<Precision> *A) {
    A->Init(rows, cols);
    for(index_t i=0; i<rows; i++) {
      for(index_t j=0; j<cols; j++) {
        A.set(i, j, math::Rand(lo, hi));
      }
    }
  } 

  /**
   *  Matlab like utility function for returning a vector initialized
   *   with random numbers in [lo, hi] interval
   *
   */
  template<typename Precision>
  void Rand(const index_t length,
            const index_t lo,
            const index_t hi, 
            GenVector<Precision> *A) {
    A->Init(length);
    for(index_t i=0; i<length; i++) {
      A->set(i, math::Rand(lo, hi));
    }
  }
 
  /**
   *  Matlab like utility function for generating  identity matrix 
   *
   */
  template<typename Precision>
  void Eye(index_t dimension, 
           GenMatrix<Precision> *I) {
    I->InitDiagonal(dimension, Precision(1.0));
  }
};

#endif

