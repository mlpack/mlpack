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

  /**
   * Entrywise element multiplication of two memory blocks
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ C \gets A .* B\f$).
   */
  template<typename Precision>
  void DotMulInit(const index_t length,
              const Precision *A, 
              const Precision *B, 
              Precision *C);

  template<typename Precision>
  void DotMulOvewrite(const index_t length,
              const Precision *A, 
              const Precision *B, 
              Precision *C);
 
  /**
   * Entrywise element multiplication of two matrices
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ C \gets A .* B\f$).
   */
 
  template<typename Precision>
  void DotMulInit(const GenMatrix<Precision> &A, 
              const GenMatrix<Precision> &B, 
              GenMatrix<Precision> *C);

  template<typename Precision>
  void DotMulOverwrite(const GenMatrix<Precision> &A, 
              const GenMatrix<Precision> &B, 
              GenMatrix<Precision> *C);
 
  /**
   * Entrywise element multiplication of two vectors
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ \vec{C} \gets \vec{A} .* \vec{B}\f$).
   */ 
  template<typename Precision>
  void DotMulInit(const GenVector<Precision> &A, 
              const GeVector<Precision> &B, 
              GenVector<Precision> *C);

  template<typename Precision>
  void DotMulOverwrite(const GenVector<Precision> &A, 
              const GeVector<Precision> &B, 
              GenVector<Precision> *C);



  /**
   * Entrywise element multiplication of two memory blocks
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ A \gets A .* B\f$).
   */
  template<typename Precision>
  void DotMulTo(const index_t length,
              const Precision *A, 
              const Precision *B);
  /**
   * Entrywise element multiplication of two matrices
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ A \gets A .* B\f$).
   */
 
  template<typename Precision>
  void DotMulTo(const GenMatrix<Precision> *A, 
              GenMatrix<Precision> &B);
  /**
   * Entrywise element multiplication of two vectors
   * Also known as the Hadamard product
   * In Matlab this is the .* operator
   * (\f$ \vec{A} \gets \vec{A} .* \vec{B}\f$).
   */ 
  template<typename Precision>
  void DotMulTo(const GenVector<Precision> *A, 
              GeVector<Precision> &B);

  /**
   * Elementwise integer powers of memory blocks
   * (\f$ B \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowInit(const index_t length,
              index_t power,
              const Precision *A,
              Precision *B); 
 
  template<typename Precision>
  void DotIntPowOverwrite(const index_t length,
              index_t power,
              const Precision *A,
              Precision *B); 
   
  /**
   * Elementwise integer powers of matrices
   * (\f$ B \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowInit(index_t power,
              const GenMatrix<Precision> &A,
              GenMatrix<Precision> *B); 

  /**
   * Elementwise integer powers of vectors
   * (\f$ \vec{B} \gets \vec{A}.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowInit(index_t power,
              const GenVector<Precision> &A,
              GenVector<Precision> *B); 

  /**
   * Elementwise integer powers of memory blocks
   * (\f$ A \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowOverwrite(const index_t length,
              index_t power,
              Precision *A); 
  
 /**
   * Elementwise integer powers of matrices
   * (\f$ A \gets A.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowOverwrite(index_t power,
              GenMatrix<Precision> *A); 

  /**
   * Elementwise integer powers of vectors
   * (\f$ \vec{A} \gets \vec{A}.\^n \f$)
   */
  template<typename Precision>
  void DotIntPowOverwrite(index_t power,
              GenVector<Precision> *A); 


  /**
   * Elementwise rational powers of memory blocks
   * (\f$ B \gets A.\^(n/m) \f$)
   */
  template<typename Precision, int n, int m>
  void DotPowInit(const index_t length
              const Precision *A,
              Precision *B); 
 
  /**
   * Elementwise integer powers of matrices
   * (\f$ B \gets A.\^(n/m) \f$)
   */
  template<typename Precision, int n, int m>
  void DotPowInit(const GenMatrix<Precision> &A,
              GenMatrix<Precision> *B); 

 
  /**
   * Elementwise integer powers of vectors
   * (\f$ B \gets A.\^(n/m) \f$)
   */ 
  template<typename Precision, int n, int m>
  void DotPowInit(const GenVector<Precision> &A,
              GenVector<Precision> *B); 
  
  /**
   * Elementwise integer powers of vectors
   * (\f$ \vec{B} \gets \vec{A}.\^(n/m) \f$)
   */
  template<typename Precision, int n, int m>
  void DotPowInit(const GenMatrix<Precision> &A,
              Gen<Precision> *B); 

  /**
   * Elementwise integer powers of memory blocks
   * (\f$ A \gets A.\^(n/m) \f$)
   */
  template<typename Precision, int n, int m>
  void DotIntPowOverwrite(const index_t length,
              index_t power,
              Precision *A); 
 
  /**
   * Elementwise integer powers of matrices
   * (\f$ A \gets A.\^(n/m) \f$)
   */
  template<typename Precision, int n, int m>
  void DotPowOverwrite(GenMatrix<Precision> *A); 
 
  /**
   * Elementwise integer powers of vectors
   * (\f$ \vec{A} \gets vec{A}.\^(n/m) \f$)
   */ 
  template<typename Precision, int n, int m>
  void DotPowOverwrite(GenVector<Precision> *A); 
  

  /**
   * Elementwise exponential of a memory block
   * (\f$ B= \exp(A) \f$)
   */
  template<typename Precision>
  void DotExpInit(const index_t length,
              const Precision *A, 
              Precision *B);
  /**
   * Elementwise exponential of a matrix
   * (\f$ B= \exp(A) \f$)
   */
  template<typename Precision>
  void DotExpInit(const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise exponential of a vector
   * (\f$ \vec{B}= \exp(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotExpInit(const index_t length,
              const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise exponential of a memory block
   * (\f$ A= \exp(A) \f$)
   */
  template<typename Precision>
  void DotExpOverwrite(const index_t length,
              Precision *A);
  /**
   * Elementwise exponential of a matrix
   * (\f$ A= \exp(A) \f$)
   */
  template<typename Precision>
  void DotExpOverwrite(GenMatrix<Precision> *A);

  /**
   * Elementwise exponential of a vector
   * (\f$ \vec{A}= \exp(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotExpOverwrite(GenVector<Precision> *A);

  /**
   * Elementwise log of a memory block
   * (\f$ B= \log(A) \f$)
   */
  template<typename Precision>
  void DotLogInit(const index_t length,
              const Precision *A, 
              Precision *B);
  /**
   * Elementwise log of a matrix
   * (\f$ B= \log(A) \f$)
   */
  template<typename Precision>
  void DotLogInit(const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise log of a vector
   * (\f$ \vec{B}= \log(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotLogInit(const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise log of a memory block
   * (\f$ A= \log(A) \f$)
   */
  template<typename Precision>
  void DotLogOverwrite(const index_t length,
              Precision *A);
  /**
   * Elementwise log of a matrix
   * (\f$ A= \log(A) \f$)
   */
  template<typename Precision>
  void DotLogOverwrite(GenMatrix<Precision> *A);

  /**
   * Elementwise log of a vector
   * (\f$ \vec{A}= exp(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotLogOverwrite(GenVector<Precision> *A);

  /**
   * Elementwise tanh of a memory block
   * (\f$ B= \tanh(A) \f$)
   */
  template<typename Precision>
  void DotTanhInit(const index_t length,
              const Precision *A, 
              Precision *B);
  /**
   * Elementwise tanh of a matrix
   * (\f$ B= \tanh(A) \f$)
   */
  template<typename Precision>
  void DotTanhInit(const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise tanh of a vector
   * (\f$ \vec{B}= \tanh(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotTanhInit(const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise tanh of a memory block
   * (\f$ A= \tanh(A) \f$)
   */
  template<typename Precision>
  void DotTanhOverwrite(const index_t length,
              Precision *A);
  /**
   * Elementwise tanh of a matrix
   * (\f$ A= \tanh(A) \f$)
   */
  template<typename Precision>
  void DotTanhOverwrite(GenMatrix<Precision> *A);

  /**
   * Elementwise tanh of a vector
   * (\f$ \vec{A}= \tanh(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotTanhOverwrite(GenVector<Precision> *A);
  
  /**
   * Elementwise tan of a memory block
   * (\f$ B= \tan(A) \f$)
   */
  template<typename Precision>
  void DotTanInit(const index_t length,
              Precision *A, 
              Precision *B);
  /**
   * Elementwise tan of a matrix
   * (\f$ B= \tan(A) \f$)
   */
  template<typename Precision>
  void DotTanInit(const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise tan of a vector
   * (\f$ \vec{B}= \tan(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotTanInit(const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise tan of a memory block
   * (\f$ A= \tan(A) \f$)
   */
  template<typename Precision>
  void DotTanOverwrite(const index_t length,
              Precision *A);
  /**
   * Elementwise tan of a matrix
   * (\f$ A= \tan(A) \f$)
   */
  template<typename Precision>
  void DotTanOverwrite(GenMatrix<Precision> *A);

  /**
   * Elementwise tan of a vector
   * (\f$ \vec{A}= \tan(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotTanOverwrite(GenVector<Precision> *A);
  
  /**
   * Elementwise sin of a memory block
   * (\f$ B= \sin(A) \f$)
   */
  template<typename Precision>
  void DotSinInit(const index_t length,
              const Precision *A, 
              Precision *B);
  /**
   * Elementwise sin of a matrix
   * (\f$ B= \sin(A) \f$)
   */
  template<typename Precision>
  void DotSinInit(const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise sin of a vector
   * (\f$ \vec{B}= \sin(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotSinInit(const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise sin of a memory block
   * (\f$ A= \sin(A) \f$)
   */
  template<typename Precision>
  void DotSinOverwrite(const index_t length,
              Precision *A);
  /**
   * Elementwise sin of a matrix
   * (\f$ A= \sin(A) \f$)
   */
  template<typename Precision>
  void DotSinOverwrite(GenMatrix<Precision> *A);

  /**
   * Elementwise sin of a vector
   * (\f$ \vec{A}= \sin(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotSinOverwrite(GenVector<Precision> *A);
 
  /**
   * Elementwise cos of a memory block
   * (\f$ B= \cos(A) \f$)
   */
  template<typename Precision>
  void DotCosInit(const index_t length,
              const Precision *A, 
              Precision *B);
  /**
   * Elementwise cos of a matrix
   * (\f$ B= \cos(A) \f$)
   */
  template<typename Precision>
  void DotCosInit(const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise cos of a vector
   * (\f$ \vec{B}= \cos(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotCosInit(const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise cos of a memory block
   * (\f$ A= \cos(A) \f$)
   */
  template<typename Precision>
  void DotCosOverwrite(const index_t length,
              Precision *A);
  /**
   * Elementwise cos of a matrix
   * (\f$ A= \cos(A) \f$)
   */
  template<typename Precision>
  void DotCosOverwrite(GenMatrix<Precision> *A);

  /**
   * Elementwise cos of a vector
   * (\f$ \vec{A}= \cos(\vec{A}) \f$)
   */
  template<typename Precision>
  void DotCosOverwrite(GenVector<Precision> *A);
 
  /**
   * Elementwise fun of a memory block
   * (\f$ B= fun(A) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunInit(const index_t length,
              Function &fun,
              const Precision *A, 
              Precision *B);
  /**
   * Elementwise fun of a matrix
   * (\f$ B= \fun(A) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunInit(Function &fun
              const GenMatrix<Precision> &A, 
              GenMatrix<Precision> *B);

  /**
   * Elementwise fun of a vector
   * (\f$ \vec{B}= \fun(\vec{A}) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunInit(Function &fun,              
              const GenVector<Precision> &A, 
              GenVector<Precision> *B);

  /**
   * Elementwise fun of a memory block
   * (\f$ A= \fun(A) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunOverwrite(const index_t length,
              Function &fun,
              Precision *A);
  
  /**
   * Elementwise fun of a matrix
   * (\f$ A= \fun(A) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunOverwrite(Function &fun,
              GenMatrix<Precision> *A);

  /**
   * Elementwise fun of a vector
   * (\f$ \vec{A}= \fun(\vec{A}) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunOverwrite(Function &fun,
               GenVector<Precision> *A);

 /**
   * Elementwise fun of  memory blocks
   * (\f$ C= fun(A, B) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunInit(const index_t length,
              Function &fun,
              const Precision *A, 
              const Precision *B, 
              Precision *C);
  /**
   * Elementwise fun of two matrices
   * (\f$ C= \fun(A, B) \f$)
   * where fun is an arbitrary function that operates on 
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunInit(Function &fun
              const GenMatrix<Precision> &A, 
              const GenMatrix<Precision> &B,
              const GenMatrix<Precision> *C);

  /**
   * Elementwise fun of two vectors
   * (\f$ \vec{C}= \fun(\vec{A}, \vec{B}) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunInit(Function &fun,              
              const GenVector<Precision> &A, 
              const GenVector<Precision> &B, 
              GenVector<Precision> *C);

  /**
   * Elementwise fun of two memory blocks
   * (\f$ A= \fun(A, B) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunOverwrite(const index_t length,
              Function &fun,
              Precision *A, 
              const Precision *B);
  
  /**
   * Elementwise fun of two matrices
   * (\f$ A= \fun(A, B) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunOverwrite(Function &fun,
              GenMatrix<Precision> *A,
              const GenMatrix<Precision> &B);

  /**
   * Elementwise fun of two vectors
   * (\f$ \vec{A}= \fun(\vec{A}, \vec{B}) \f$)
   * a single element.
   * fun is passed as a function object. It is a class that overloads the 
   * operator()
   */
  template<typename Precision, typename Function>
  void DotFunOverwrite(Function &fun,
               GenVector<Precision> *A, 
               const GenVector<Precision> &B);

  /**
   * Sums the elements of a memory block
   * (\f$ sum(A))
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision>
  RetPrecision Sum(const index_t length, 
      const Precision *A);

  /**
   * it sums all the elements of matrix A
   * (\f$ sum(sum(A))\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  RetPrecision Sum(const index_t length, 
      const GenMatrix<Precision> &A);
  
  /**
   * it sums all the columns of matrix A
   * (\f$ sum(A)\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  void SumInit(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_sums);

  template<typename Precision, typename RetPrecision>
  void SumOverwrite(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_sums);

  /**
   * it sums all the rows of matrix A
   * (\f$ sum(A,2)\f$)
   *
   */ 
  template<typename Precision, typename RetPrecision>
  void SumInit(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *row_sums);
  
  template<typename Precision, typename RetPrecision>
  void SumOverwrite(const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *row_sums);

  /**
   * Sums the elements of a Vector
   * (\f$ sum(\vec{A}))
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision>
  void RetPrecision Sum(const GenVector<Precision> &A);

  /**
   * Sums the elements of a memory block preprocessed with a function
   * (\f$ sum(fun(A)))
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   * Class Function is a function object so it must implement
   * the operator()
   */
  template<typename Precision, typename RetPrecision, Function fun>
  RetPrecision FunSum(const index_t length,
      Function &fun, 
      const Precision *A);

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
      const GenMatrix<Precision> &A);
  
  /**
   * it sums all the columns of matrix A preprocessed with a function
   * (\f$ sum(fun(A))\f$)
   * Class Function is a function object so it must implement
   * the operator() 
   */ 
  template<typename Precision, typename RetPrecision, typename Function>
  void FunSumInit(Function &fun, 
      const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_sums);

  template<typename Precision, typename RetPrecision, typename Function>
  void FunSumOverwrite(Function &fun,
      const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *col_sums);

  /**
   * it sums all the rows of matrix A preprocessed with a function
   * (\f$ sum(A,2)\f$)
   * Class Function is a function object so it must implement
   * the operator() 
   *
   */ 
  template<typename Precision, typename RetPrecision, typename Function>
  void FunSumInit(Function &fun,
      const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *row_sums);
  
  template<typename Precision, typename RetPrecision, typename Function>
  void FunSumOverwrite(Function &fun,
      const GenMatrix<Precision> &A,
      GenVector<RetPrecision> *row_sums);

  /**
   * Sums the elements of a Vector preprocessed with a function
   * (\f$ sum(\vec{A}))
   * Notice: Because summation can lead to overflow we have a second
   * template parameter called RetPrecision for defining the Precision
   * of the sum. 
   */
  template<typename Precision, typename RetPrecision, typename Function>
  void RetPrecision FunSum(Function &fun,                      
                           const GenVector<Precision> &A);



  /**
   *  Matlab like utility function for returning a matrix initialized
   *  with zeros
   *
   */
  template<typename Precision>
  void Zeros(const index_t rows, 
             consr index_t cols, 
             GenMatrix<Precision> *A); 

  /**
   *  Matlab like utility function for returning a vector initialized
   *  with zeros
   *
   */
  template<typename Precision>
  void Zeros(const index_t length 
             GenVector<Precision> *A); 

  /**
   *  Matlab like utility function for returning a matrix initialized
   *  with ones
   *
   */
  template<typename Precision>
  void Ones(const index_t rows, 
            const index_t cols, 
            GenMatrix<Precision> *A); 

  /**
   *  Matlab like utility function for returning a vector initialized
   *  with ones
   *
   */
  template<typename Precision>
  void Ones(const index_t length, 
             GenVector<Precision> *A); 

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
            GenMatrix<Precision> *A); 

  /**
   *  Matlab like utility function for returning a vector initialized
   *   with random numbers in [lo, hi] interval
   *
   */
  template<typename Precision>
  void Rand(const index_t length,
            const index_t lo,
            const index_t hi, 
            GenVector<Precision> *A); 
 
  /**
   *  Matlab like utility function for generating  identity matrix 
   *
   */
  template<typename Precision>
  void Eye(index_t size, 
           GenMatrix<Precision> *I);
};

#endif

