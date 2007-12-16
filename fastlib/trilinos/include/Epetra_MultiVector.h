
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_MULTIVECTOR_H
#define EPETRA_MULTIVECTOR_H

class Epetra_Comm;
class Epetra_BlockMap;
class Epetra_Map;
class Epetra_Import;
class Epetra_Export;
class Epetra_Distributor;
class Epetra_Vector;

#include "Epetra_DistObject.h"
#include "Epetra_CompObject.h"
#include "Epetra_BLAS.h"
#include "Epetra_Util.h"

//! Epetra_MultiVector: A class for constructing and using dense multi-vectors, vectors and matrices in parallel.

/*! The Epetra_MultiVector class enables the construction and use of real-valued, 
  double-precision dense vectors, multi-vectors,
  and matrices in a distributed memory environment.  The dimensions and distribution of the dense
  multi-vectors is determined in part by a Epetra_Comm object, a Epetra_Map (or Epetra_LocalMap
  or Epetra_BlockMap) and the number of vectors passed to the constructors described below.

  There are several concepts that important for understanding the Epetra_MultiVector class:

  <ul>
  <li>  Multi-vectors, Vectors and Matrices.  
  <ul>
  <li> Vector - A list of real-valued, double-precision numbers.  Also a multi-vector with one vector.
  <li> Multi-Vector - A collection of one or more vectors, all having the same length and distribution.
  <li> (Dense) Matrix - A special form of multi-vector such that stride in memory between any 
  two consecutive vectors in the multi-vector is the same for all vectors.  This is identical
  to a two-dimensional array in Fortran and plays an important part in high performance
  computations.
  </ul>
  <li> Distributed Global vs. Replicated Local.
  <ul>
  <li> Distributed Global Multi-vectors - In most instances, a multi-vector will be partitioned
  across multiple memory images associated with multiple processors.  In this case, there is 
  a unique copy of each element and elements are spread across all processors specified by 
  the Epetra_Comm communicator.
  <li> Replicated Local Multi-vectors - Some algorithms use multi-vectors that are too small to
  be distributed across all processors, the Hessenberg matrix in a GMRES
  computation.  In other cases, such as with block iterative methods,  block dot product 
  functions produce small
  dense matrices that are required by all processors.  Replicated local multi-vectors handle
  these types of situation.
  </ul>
  <li> Multi-vector Functions vs. Dense Matrix Functions.
  <ul>
  <li> Multi-vector functions - These functions operate simultaneously but independently
  on each vector in the multi-vector and produce individual results for each vector.
  <li> Dense matrix functions - These functions operate on the multi-vector as a matrix, 
  providing access to selected dense BLAS and LAPACK operations.
  </ul>
  </ul>

  <b>Constructing Epetra_MultiVectors</b>

  Except for the basic constructor and copy constructor, Epetra_MultiVector constructors
  have two data access modes:
  <ol>
  <li> Copy mode - Allocates memory and makes a copy of the user-provided data. In this case, the
  user data is not needed after construction.
  <li> View mode - Creates a "view" of the user data. In this case, the
  user data is required to remain intact for the life of the multi-vector.
  </ol>

  \warning View mode is \e extremely dangerous from a data hiding perspective.
  Therefore, we strongly encourage users to develop code using Copy mode first and 
  only use the View mode in a secondary optimization phase.

  All Epetra_MultiVector constructors require a map argument that describes the layout of elements
  on the parallel machine.  Specifically, 
  \c map is a Epetra_Map, Epetra_LocalMap or Epetra_BlockMap object describing the desired
  memory layout for the multi-vector.

  There are six different Epetra_MultiVector constructors:
  <ul>
  <li> Basic - All values are zero.
  <li> Copy - Copy an existing multi-vector.
  <li> Copy from or make view of two-dimensional Fortran style array.
  <li> Copy from or make view of an array of pointers.
  <li> Copy or make view of a list of vectors from another Epetra_MultiVector object.
  <li> Copy or make view of a range of vectors from another Epetra_MultiVector object.
  </ul>

  <b>Extracting Data from Epetra_MultiVectors</b>

  Once a Epetra_MultiVector is constructed, it is possible to extract a copy of the values or create
  a view of them.

  \warning ExtractView functions are \e extremely dangerous from a data hiding perspective.
  For both ExtractView fuctions, there is a corresponding ExtractCopy function.  We
  strongly encourage users to develop code using ExtractCopy functions first and 
  only use the ExtractView functions in a secondary optimization phase.

  There are four Extract functions:
  <ul>
  <li> ExtractCopy - Copy values into a user-provided two-dimensional array.
  <li> ExtractCopy - Copy values into a user-provided array of pointers.
  <li> ExtractView - Set user-provided two-dimensional array parameters 
  to point to Epetra_MultiVector data.
  <li> ExtractView - Set user-provided array of pointer parameters 
  to point to Epetra_MultiVector data.
  </ul>

  <b>Vector, Matrix and Utility Functions</b>

  Once a Epetra_MultiVector is constructed, a variety of mathematical functions can be applied to
  the individual vectors.  Specifically:
  <ul>
  <li> Dot Products.
  <li> Vector Updates.
  <li> \e p Norms.
  <li> Weighted Norms.
  <li> Minimum, Maximum and Average Values.
  </ul>

  In addition, a matrix-matrix multiply function supports a variety of operations on any viable
  combination of global distributed and local replicated multi-vectors using calls to DGEMM, a
  high performance kernel for matrix operations.  In the near future we will add support for calls
  to other selected BLAS and LAPACK functions.

  <b> Counting Floating Point Operations </b>

  Each Epetra_MultiVector object keep track of the number
  of \e serial floating point operations performed using the specified object as the \e this argument
  to the function.  The Flops() function returns this number as a double precision number.  Using this 
  information, in conjunction with the Epetra_Time class, one can get accurate parallel performance
  numbers.  The ResetFlops() function resets the floating point counter.

  \warning A Epetra_Map, Epetra_LocalMap or Epetra_BlockMap object is required for all 
  Epetra_MultiVector constructors.

*/

//==========================================================================
class Epetra_MultiVector: public Epetra_DistObject, public Epetra_CompObject, public Epetra_BLAS {

 public:

   //! @name Constructors/destructors
  //@{ 
  //! Basic Epetra_MultiVector constuctor.
  /*! Creates a Epetra_MultiVector object and, by default, fills with zero values.  

  \param In 
  Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.

  \warning Note that, because Epetra_LocalMap
  derives from Epetra_Map and Epetra_Map derives from Epetra_BlockMap, this constructor works
  for all three types of Epetra map classes.
  \param In 
  NumVectors - Number of vectors in multi-vector.
  \param In
  zeroOut - If <tt>true</tt> then the allocated memory will be zeroed
            out initialy.  If <tt>false</tt> then this memory will not
            be touched which can be significantly faster.
  \return Pointer to a Epetra_MultiVector.

  */
  Epetra_MultiVector(const Epetra_BlockMap& Map, int NumVectors, bool zeroOut = true);

  //! Epetra_MultiVector copy constructor.
  
  Epetra_MultiVector(const Epetra_MultiVector& Source);
  
  //! Set multi-vector values from two-dimensional array.
  /*!
    \param In 
    Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In 
    Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
    \param In
    A - Pointer to an array of double precision numbers.  The first vector starts at A.
    The second vector starts at A+MyLDA, the third at A+2*MyLDA, and so on.
    \param In
    MyLDA - The "Leading Dimension", or stride between vectors in memory.
    \warning This value refers to the stride on the calling processor.  Thus it is a
    local quantity, not a global quantity.
    \param In 
    NumVectors - Number of vectors in multi-vector.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  Epetra_MultiVector(Epetra_DataAccess CV, const Epetra_BlockMap& Map, 
		     double *A, int MyLDA, int NumVectors);

  //! Set multi-vector values from array of pointers.
  /*!
    \param In 
    Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In 
    Map - A Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
    \param In
    ArrayOfPointers - An array of pointers such that ArrayOfPointers[i] points to the memory
    location containing ith vector to be copied.
    \param In 
    NumVectors - Number of vectors in multi-vector.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  Epetra_MultiVector(Epetra_DataAccess CV, const Epetra_BlockMap& Map, 
		     double **ArrayOfPointers, int NumVectors);

  //! Set multi-vector values from list of vectors in an existing Epetra_MultiVector.
  /*!
    \param In 
    Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In
    Source - An existing fully constructed Epetra_MultiVector.
    \param In
    Indices - Integer list of the vectors to copy.  
    \param In 
    NumVectors - Number of vectors in multi-vector.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  Epetra_MultiVector(Epetra_DataAccess CV,  
		     const Epetra_MultiVector& Source, int *Indices, int NumVectors);

  //! Set multi-vector values from range of vectors in an existing Epetra_MultiVector.
  /*!
    \param In 
    Epetra_DataAccess - Enumerated type set to Copy or View.
    \param In
    Source - An existing fully constructed Epetra_MultiVector.
    \param In
    StartIndex - First of the vectors to copy.  
    \param In 
    NumVectors - Number of vectors in multi-vector.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  Epetra_MultiVector(Epetra_DataAccess CV, 
		     const Epetra_MultiVector& Source, int StartIndex, 
		     int NumVectors);
  
  //! Epetra_MultiVector destructor.  
  virtual ~Epetra_MultiVector();
  //@}

  //! @name Post-construction modification routines
  //@{ 

  //! Replace current value  at the specified (GlobalRow, VectorIndex) location with ScalarValue.
  /*!
    Replaces the  existing value for a single entry in the multivector.  The
    specified global row must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    If the map associated with this multivector is an Epetra_BlockMap, only the first point entry associated
    with the global row will be modified.  To modify a different point entry, use the other version of
    this method

    \param In
    GlobalRow - Row of Multivector to modify in global index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if GlobalRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors().
  */
  int ReplaceGlobalValue(int GlobalRow, int VectorIndex, double ScalarValue);


  //! Replace current value at the specified (GlobalBlockRow, BlockRowOffset, VectorIndex) location with ScalarValue.
  /*!
    Replaces the existing value for a single entry in the multivector.  The
    specified global block row and block row offset 
    must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    \param In
    GlobalBlockRow - BlockRow of Multivector to modify in global index space.
    \param In
    BlockRowOffset - Offset into BlockRow of Multivector to modify in global index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if GlobalRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors(), set to -2 if BlockRowOffset is out-of-range.
  */
  int ReplaceGlobalValue(int GlobalBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue);


  //! Adds ScalarValue to existing value at the specified (GlobalRow, VectorIndex) location.
  /*!
    Sums the given value into the existing value for a single entry in the multivector.  The
    specified global row must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    If the map associated with this multivector is an Epetra_BlockMap, only the first point entry associated
    with the global row will be modified.  To modify a different point entry, use the other version of
    this method

    \param In
    GlobalRow - Row of Multivector to modify in global index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if GlobalRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors().
  */
  int SumIntoGlobalValue(int GlobalRow, int VectorIndex, double ScalarValue);


  //! Adds ScalarValue to existing value at the specified (GlobalBlockRow, BlockRowOffset, VectorIndex) location.
  /*!
    Sums the given value into the existing value for a single entry in the multivector.  The
    specified global block row and block row offset 
    must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    \param In
    GlobalBlockRow - BlockRow of Multivector to modify in global index space.
    \param In
    BlockRowOffset - Offset into BlockRow of Multivector to modify in global index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if GlobalRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors(), set to -2 if BlockRowOffset is out-of-range.
  */
  int SumIntoGlobalValue(int GlobalBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue);

  //! Replace current value  at the specified (MyRow, VectorIndex) location with ScalarValue.
  /*!
    Replaces the existing value for a single entry in the multivector.  The
    specified local row must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    This method is intended for use with vectors based on an Epetra_Map.  If used 
    on a vector based on a non-trivial Epetra_BlockMap, this will update only block 
    row 0, i.e. 

    Epetra_MultiVector::ReplaceMyValue  (  MyRow,  VectorIndex,  ScalarValue )  is 
    equivalent to:  
    Epetra_MultiVector::ReplaceMyValue  (  0, MyRow,  VectorIndex,  ScalarValue )


    \param In
    MyRow - Row of Multivector to modify in local index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if MyRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors().
  */
  int ReplaceMyValue(int MyRow, int VectorIndex, double ScalarValue);


  //! Replace current value at the specified (MyBlockRow, BlockRowOffset, VectorIndex) location with ScalarValue.
  /*!
    Replaces the existing value for a single entry in the multivector.  The
    specified local block row and block row offset 
    must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    \param In
    MyBlockRow - BlockRow of Multivector to modify in local index space.
    \param In
    BlockRowOffset - Offset into BlockRow of Multivector to modify in local index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if MyRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors(), set to -2 if BlockRowOffset is out-of-range.
  */
  int ReplaceMyValue(int MyBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue);


  //! Adds ScalarValue to existing value at the specified (MyRow, VectorIndex) location.
  /*!
    Sums the given value into the existing value for a single entry in the multivector.  The
    specified local row must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    If the map associated with this multivector is an Epetra_BlockMap, only the first point entry associated
    with the local row will be modified.  To modify a different point entry, use the other version of
    this method

    \param In
    MyRow - Row of Multivector to modify in local index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if MyRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors().
  */
  int SumIntoMyValue(int MyRow, int VectorIndex, double ScalarValue);


  //! Adds ScalarValue to existing value at the specified (MyBlockRow, BlockRowOffset, VectorIndex) location.
  /*!
    Sums the given value into the existing value for a single entry in the multivector.  The
    specified local block row and block row offset 
    must correspond to a GID owned by the map of the multivector on the
    calling processor.  In other words, this method does not perform cross-processor communication.

    \param In
    MyBlockRow - BlockRow of Multivector to modify in local index space.
    \param In
    BlockRowOffset - Offset into BlockRow of Multivector to modify in local index space.
    \param In
    VectorIndex - Vector within MultiVector that should to modify.
    \param In
    ScalarValue - Value to add to existing value.

    \return Integer error code, set to 0 if successful, set to 1 if MyRow not associated with calling processor
    set to -1 if VectorIndex >= NumVectors(), set to -2 if BlockRowOffset is out-of-range.
  */
  int SumIntoMyValue(int MyBlockRow, int BlockRowOffset, int VectorIndex, double ScalarValue);

  //! Initialize all values in a multi-vector with constant value.
  /*!
    \param In
    ScalarConstant - Value to use.

    \return Integer error code, set to 0 if successful.
  */
  int PutScalar (double ScalarConstant);
  
  //! Set multi-vector values to random numbers.
  /*! MultiVector uses the random number generator provided by Epetra_Util. 
		The multi-vector values will be set to random values on the interval (-1.0, 1.0).

    \return Integer error code, set to 0 if successful.

  */
  int Random();

  //@}

  //! @name Extraction methods
  //@{ 

  //! Put multi-vector values into user-provided two-dimensional array.
  /*!
    \param Out
    A - Pointer to memory space that will contain the multi-vector values.  
    The first vector will be copied to the memory pointed to by A.
    The second vector starts at A+MyLDA, the third at A+2*MyLDA, and so on.
    \param In
    MyLDA - The "Leading Dimension", or stride between vectors in memory.
    \warning This value refers to the stride on the calling processor.  Thus it is a
    local quantity, not a global quantity.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  int ExtractCopy(double *A, int MyLDA) const;

  //! Put multi-vector values into user-provided array of pointers.
  /*!
    \param Out
    ArrayOfPointers - An array of pointers to memory space that will contain the 
    multi-vector values, such that ArrayOfPointers[i] points to the memory
    location where the ith vector to be copied.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  int ExtractCopy(double **ArrayOfPointers) const;

  // ExtractView functions

  
  //! Set user-provided addresses of A and MyLDA.
  /*!
    \param 
    A (Out) - Address of a pointer to that will be set to point to the values of the multi-vector.  
    The first vector will be at the memory pointed to by A.
    The second vector starts at A+MyLDA, the third at A+2*MyLDA, and so on.
    \param 
    MyLDA (Out) - Address of the "Leading Dimension", or stride between vectors in memory.
    \warning This value refers to the stride on the calling processor.  Thus it is a
    local quantity, not a global quantity.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  int ExtractView(double **A, int *MyLDA) const;

  //! Set user-provided addresses of ArrayOfPointers.
  /*!
    \param 
    ArrayOfPointers (Out) - Address of array of pointers to memory space that will set to the
    multi-vector array of pointers, such that ArrayOfPointers[i] points to the memory
    location where the ith vector is located.

    \return Integer error code, set to 0 if successful.

    See Detailed Description section for further discussion.
  */
  int ExtractView(double ***ArrayOfPointers) const;

  //@}

  //! @name Mathematical methods
  //@{ 

  //! Computes dot product of each corresponding pair of vectors.
  /*!
    \param In
    A - Multi-vector to be used with the "\e this" multivector.
    \param Out
    Result - Result[i] will contain the ith dot product result.

    \return Integer error code, set to 0 if successful.
  */
  int Dot(const Epetra_MultiVector& A, double *Result) const;

  //! Puts element-wise absolute values of input Multi-vector in target.
  /*!
    \param In
    A - Input Multi-vector.
    \param Out
    \e this will contain the absolute values of the entries of A.

    \return Integer error code, set to 0 if successful.
    
    Note:  It is possible to use the same argument for A and \e this.
  */
  int Abs(const Epetra_MultiVector& A);

  //! Puts element-wise reciprocal values of input Multi-vector in target.
  /*!
    \param In
    A - Input Multi-vector.
    \param Out
    \e this will contain the element-wise reciprocal values of the entries of A.

    \return Integer error code, set to 0 if successful.  Returns 2 if some entry
    is too small, but not zero.  Returns 1 if some entry is zero.
    
    Note:  It is possible to use the same argument for A and \e this.  Also, 
    if a given value of A is smaller than Epetra_DoubleMin (defined in Epetra_Epetra.h),
    but nonzero, then the return code is 2.  If an entry is zero, the return code
    is 1.  However, in all cases the reciprocal value is still used, even
    if a NaN is the result.
  */
  int Reciprocal(const Epetra_MultiVector& A);

  //! Scale the current values of a multi-vector, \e this = ScalarValue*\e this.
  /*!
    \param In
    ScalarValue - Scale value.
    \param Out
    \e This - Multi-vector with scaled values.

    \return Integer error code, set to 0 if successful.
  */
  int Scale(double ScalarValue);

  //! Replace multi-vector values with scaled values of A, \e this = ScalarA*A.
  /*!
    \param In
    ScalarA - Scale value.
    \param In
    A - Multi-vector to copy.
    \param Out
    \e This - Multi-vector with values overwritten by scaled values of A.

    \return Integer error code, set to 0 if successful.
  */
  int Scale(double ScalarA, const Epetra_MultiVector& A);

  //! Update multi-vector values with scaled values of A, \e this = ScalarThis*\e this + ScalarA*A.
  /*!
    \param In
    ScalarA - Scale value for A.
    \param In
    A - Multi-vector to add.
    \param In
    ScalarThis - Scale value for \e this.
    \param Out
    \e This - Multi-vector with updatede values.

    \return Integer error code, set to 0 if successful.
  */
  int Update(double ScalarA, const Epetra_MultiVector& A, double ScalarThis);

  //! Update multi-vector with scaled values of A and B, \e this = ScalarThis*\e this + ScalarA*A + ScalarB*B.
  /*!
    \param In
    ScalarA - Scale value for A.
    \param In
    A - Multi-vector to add.
    \param In
    ScalarB - Scale value for B.
    \param In
    B - Multi-vector to add.
    \param In
    ScalarThis - Scale value for \e this.
    \param Out
    \e This - Multi-vector with updatede values.

    \return Integer error code, set to 0 if successful.
  */
  int Update(double ScalarA, const Epetra_MultiVector& A, 
	     double ScalarB, const Epetra_MultiVector& B, double ScalarThis);

  //! Compute 1-norm of each vector in multi-vector.
  /*!
    \param Out
    Result - Result[i] contains 1-norm of ith vector.

    \return Integer error code, set to 0 if successful.
  */
  int Norm1   (double * Result) const;

  //! Compute 2-norm of each vector in multi-vector.
  /*!
    \param Out
    Result - Result[i] contains 2-norm of ith vector.

    \return Integer error code, set to 0 if successful.
  */
  int Norm2   (double * Result) const;

  //! Compute Inf-norm of each vector in multi-vector.
  /*!
    \param Out
    Result - Result[i] contains Inf-norm of ith vector.

    \return Integer error code, set to 0 if successful.
  */
  int NormInf (double * Result) const;

  //! Compute Weighted 2-norm (RMS Norm) of each vector in multi-vector.
  /*!
    \param In
    Weights - Multi-vector of weights.  If Weights contains a single vector,
    that vector will be used as the weights for all vectors of \e this.  Otherwise,
    Weights should have the same number of vectors as \e this.
    \param Out
    Result - Result[i] contains the weighted 2-norm of ith vector.  Specifically
    if we denote the ith vector in the multivector by \f$x\f$, and the ith weight
    vector by \f$w\f$ and let j represent the jth entry of each vector, on return
    Result[i] will contain the following result:
    \f[\sqrt{(1/n)\sum_{j=1}^n(x_j/w_j)^2}\f],
    where \f$n\f$ is the global length of the vectors.

    \return Integer error code, set to 0 if successful.
  */
  int NormWeighted   (const Epetra_MultiVector& Weights, double * Result) const;

  //! Compute minimum value of each vector in multi-vector.
  /*! Note that the vector contents must be already initialized for this
      function to compute a well-defined result. The length of the
      vector need not be greater than zero on all processors. If length is
      greater than zero on any processor then a valid result will be computed.
    \param Out
    Result - Result[i] contains minimum value of ith vector.

    \return Integer error code, set to 0 if successful.
  */
  int MinValue  (double * Result) const;

  //! Compute maximum value of each vector in multi-vector.
  /*! Note that the vector contents must be already initialized for this
      function to compute a well-defined result. The length of the
      vector need not be greater than zero on all processors. If length is
      greater than zero on any processor then a valid result will be computed.
    \param Out
    Result - Result[i] contains maximum value of ith vector.

    \return Integer error code, set to 0 if successful.
  */
  int MaxValue  (double * Result) const;

  //! Compute mean (average) value of each vector in multi-vector.
  /*!
    \param Out
    Result - Result[i] contains mean value of ith vector.

    \return Integer error code, set to 0 if successful.
  */
  int MeanValue (double * Result) const;
  
  
  //! Matrix-Matrix multiplication, \e this = ScalarThis*\e this + ScalarAB*A*B.
  /*! This function performs a variety of matrix-matrix multiply operations, interpreting
    the Epetra_MultiVectors (\e this-aka C , A and B) as 2D matrices.  Variations are due to
    the fact that A, B and C can be local replicated or global distributed
    Epetra_MultiVectors and that we may or may not operate with the transpose of 
    A and B.  Possible cases are:
    \verbatim

    Total of 32 case (2^5).
    Num
    OPERATIONS                        case  Notes
    1) C(local) = A^X(local) * B^X(local)  4   (X=Transpose or Not, No comm needed) 
    2) C(local) = A^T(distr) * B  (distr)  1   (2D dot product, replicate C)
    3) C(distr) = A  (distr) * B^X(local)  2   (2D vector update, no comm needed)

    Note that the following operations are not meaningful for 
    1D distributions:

    1) C(local) = A^T(distr) * B^T(distr)  1
    2) C(local) = A  (distr) * B^X(distr)  2
    3) C(distr) = A^X(local) * B^X(local)  4
    4) C(distr) = A^X(local) * B^X(distr)  4
    5) C(distr) = A^T(distr) * B^X(local)  2
    6) C(local) = A^X(distr) * B^X(local)  4
    7) C(distr) = A^X(distr) * B^X(local)  4
    8) C(local) = A^X(local) * B^X(distr)  4

    \endverbatim

    \param In
    TransA - Operate with the transpose of A if = 'T', else no transpose if = 'N'.
    \param In
    TransB - Operate with the transpose of B if = 'T', else no transpose if = 'N'.

    \param In
    ScalarAB - Scalar to multiply with A*B.
    \param In
    A - Multi-vector.
    \param In
    B - Multi-vector.
    \param In
    ScalarThis - Scalar to multiply with \e this.

    \return Integer error code, set to 0 if successful.

    \warning {Each multi-vector A, B and \e this is checked if it has constant stride using the
    ConstantStride() query function.  If it does not have constant stride, a temporary
    copy is made and used for the computation.  This activity is transparent to the user,
    except that there is memory and computation overhead.  All temporary space is deleted
    prior to exit.}
	 
  */
  int Multiply(char TransA, char TransB, double ScalarAB, 
	       const Epetra_MultiVector& A, const Epetra_MultiVector& B,
	       double ScalarThis );
  


  //! Multiply a Epetra_MultiVector with another, element-by-element.
  /*! This function supports diagonal matrix multiply.  A is usually a single vector
    while B and \e this may have one or more columns.  Note that B and \e this must
    have the same shape.  A can be one vector or have the same shape as B.  The actual
    computation is \e this = ScalarThis * \e this + ScalarAB * B @ A where @ denotes element-wise
    multiplication.
  */
  int Multiply(double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B,
	       double ScalarThis );


  //! Multiply a Epetra_MultiVector by the reciprocal of another, element-by-element.
  /*! This function supports diagonal matrix scaling.  A is usually a single vector
    while B and \e this may have one or more columns.  Note that B and \e this must
    have the same shape.  A can be one vector or have the same shape as B. The actual
    computation is \e this = ScalarThis * \e this + ScalarAB * B @ A where @ denotes element-wise
    division.
  */
  int ReciprocalMultiply(double ScalarAB, const Epetra_MultiVector& A, const Epetra_MultiVector& B,
			 double ScalarThis );

  //@}

  //! @name Random number utilities
  //@{ 


  //! Set seed for Random function.
  /*!
    \param In
    Seed - Should be an integer on the interval (0, 2^31-1).

    \return Integer error code, set to 0 if successful.
  */
  int SetSeed(unsigned int Seed){return(Util_.SetSeed(Seed));};

  //! Get seed from Random function.
  /*!
    \return Current random number seed.
  */
  unsigned int Seed(){return(Util_.Seed());};

  //@}

  //! @name Overloaded operators
  //@{ 

  //! = Operator.
  /*!
    \param In
    A - Epetra_MultiVector to copy.

    \return Epetra_MultiVector.
  */
  Epetra_MultiVector& operator = (const Epetra_MultiVector& Source);
  
  // Local element access functions

  // 

  //! Vector access function.
  /*!
    \return Pointer to the array of doubles containing the local values of the ith vector in the multi-vector.
  */
  double*& operator [] (int i) { return Pointers_[i]; }
  //! Vector access function.
  /*!
    \return Pointer to the array of doubles containing the local values of the ith vector in the multi-vector.
  */
  //  const double*& operator [] (int i) const;
  double * const & operator [] (int i) const { return Pointers_[i]; }

  //! Vector access function.
  /*!
    \return An Epetra_Vector pointer to the ith vector in the multi-vector.
  */
  Epetra_Vector * & operator () (int i);
  //! Vector access function.
  /*!
    \return An Epetra_Vector pointer to the ith vector in the multi-vector.
  */
  const Epetra_Vector * & operator () (int i) const;

  //@}

  //! @name Attribute access functions
  //@{ 
  
  //! Returns the number of vectors in the multi-vector.
  int NumVectors() const {return(NumVectors_);};

  //! Returns the local vector length on the calling processor of vectors in the multi-vector.
  int MyLength() const {return(MyLength_);};

  //! Returns the global vector length of vectors in the multi-vector.
  int GlobalLength() const {return(GlobalLength_);};

  //! Returns the stride between  vectors in the multi-vector (only meaningful if ConstantStride() is true).
  int Stride() const {return(Stride_);};
  
  //! Returns true if this multi-vector has constant stride between vectors.
  bool ConstantStride() const {return(ConstantStride_);};
  //@}

  /** Replace map, only if new map has same point-structure as current map.
      return 0 if map is replaced, -1 if not.
   */
  int ReplaceMap(const Epetra_BlockMap& map);

  //! @name I/O methods
  //@{ 

  //! Print method
  virtual void Print(ostream & os) const;
  //@}

  //! @name Expert-only unsupported methods
  //@{ 

  //! Reset the view of an existing multivector to point to new user data.
  /*! Allows the (very) light-weight replacement of multivector values for an
    existing multivector that was constructed using an Epetra_DataAccess mode of View.
    No checking is performed to see if the array of values passed in contains valid 
    data.  It is assumed that the user has verified the integrity of data before calling
    this method. This method is useful for situations where a multivector is needed
    for use with an Epetra operator or matrix and the user is not passing in a multivector,
    or the multivector is being passed in with another map that is not exactly compatible
    with the operator, but has the correct number of entries.

    This method is used by AztecOO and Ifpack in the matvec, and solve methods to improve
    performance and reduce repeated calls to constructors and destructors.

    @param ArrayOfPointers Contains the array of pointers containing the multivector data.

    \return Integer error code, set to 0 if successful, -1 if the multivector was not created as a View.

    \warning This method is extremely dangerous and should only be used by experts.
  */

  int ResetView(double ** ArrayOfPointers);

  //! Get pointer to MultiVector values
  double* Values() const {return Values_;};

  //! Get pointer to individual vector pointers
  double** Pointers() const {return Pointers_;};
  //@}

  // Expert-only function
  int Reduce(); 

 protected:

  // Internal utilities
  void Assign(const Epetra_MultiVector& rhs);
  int CheckInput();

  double *Values_;    // local MultiVector coefficients

 private:


  // Internal utilities

  int AllocateForCopy(void);
  int DoCopy(void);

  inline void UpdateDoubleTemp() const
  {if (DoubleTemp_==0) DoubleTemp_=new double[NumVectors_+1]; return;}

  inline void UpdateVectors()  const {if (Vectors_==0) { Vectors_ = new Epetra_Vector *[NumVectors_]; 
    for (int i=0; i<NumVectors_; i++) Vectors_[i] = 0;}
    return;
  }

  int AllocateForView(void);
  int DoView(void);
  int ChangeGlobalValue(int GlobalBlockRow,
                        int BlockRowOffset, 
                        int VectorIndex,
                        double ScalarValue,
                        bool SumInto);
  int ChangeMyValue(int MyBlockRow,
                    int BlockRowOffset, 
                    int VectorIndex,
                    double ScalarValue,
                    bool SumInto);

  int CheckSizes(const Epetra_SrcDistObject& A);

  int CopyAndPermute(const Epetra_SrcDistObject & Source,
                     int NumSameIDs, 
                     int NumPermuteIDs,
                     int * PermuteToLIDs,
                     int * PermuteFromLIDs,
                     const Epetra_OffsetIndex * Indexor);

  int PackAndPrepare(const Epetra_SrcDistObject & Source,
                     int NumExportIDs,
                     int * ExportLIDs,
                     int & LenExports,
                     char * & Exports,
                     int & SizeOfPacket,
                     int * Sizes,
                     bool & VarSizes,
                     Epetra_Distributor & Distor);
  
  int UnpackAndCombine(const Epetra_SrcDistObject & Source,
                       int NumImportIDs,
                       int * ImportLIDs, 
                       int LenImports, 
                       char * Imports,
                       int & SizeOfPacket, 
                       Epetra_Distributor & Distor,
                       Epetra_CombineMode CombineMode,
                       const Epetra_OffsetIndex * Indexor );

  double **Pointers_;        // Pointers to each vector;
  
  int MyLength_;
  int GlobalLength_;
  int NumVectors_;
  bool UserAllocated_;
  bool ConstantStride_;
  int Stride_;
  bool Allocated_;
  mutable double * DoubleTemp_;
  mutable Epetra_Vector ** Vectors_;
  Epetra_Util Util_;

};

#endif /* EPETRA_MULTIVECTOR_H */
