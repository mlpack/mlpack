
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

#ifndef EPETRA_UTIL_H
#define EPETRA_UTIL_H

#include "Epetra_Object.h"
class Epetra_Map;
class Epetra_BlockMap;
class Epetra_CrsMatrix;
class Epetra_MultiVector;

//! Epetra_Util:  The Epetra Util Wrapper Class.
/*! The Epetra_Util class is a collection of useful functions that cut across a broad
  set of other classes.
<ul>
<li> A random number generator is provided, along with methods to set and
retrieve the random-number seed.

The random number generator is a multiplicative linear congruential generator, 
with multiplier 16807 and modulus 2^31 - 1. It is based on the algorithm described in
"Random Number Generators: Good Ones Are Hard To Find", S. K. Park and K. W. Miller, 
Communications of the ACM, vol. 31, no. 10, pp. 1192-1201.

<li> Sorting is provided by a static function on this class (i.e., it is not
necessary to construct an instance of this class to use the Sort function).

<li> A static function is provided for creating a new Epetra_Map object with
 1-to-1 ownership of entries from an existing map which may have entries that
appear on multiple processors.
</ul>

  Epetra_Util is a serial interface only.  This is appropriate since the standard 
  utilities are only specified for serial execution (or shared memory parallel).
*/
class Epetra_Util {
    
  public:
  //! Epetra_Util Constructor.
  /*! Builds an instance of a serial Util object.
   */
  Epetra_Util();


  //! Epetra_Util Copy Constructor.
  /*! Makes an exact copy of an existing Epetra_Util instance.
  */
  Epetra_Util(const Epetra_Util& Util);

  //! Epetra_Util Destructor.
  virtual ~Epetra_Util();

  //! @name Random number utilities
  //@{ 

  //! Returns a random integer on the interval (0, 2^31-1)
  unsigned int RandomInt();

  //! Returns a random double on the interval (-1.0,1.0)
  double RandomDouble();

  //! Get seed from Random function.
  /*!
    \return Current random number seed.
  */
  unsigned int Seed() const;

  //! Set seed for Random function.
  /*!
    \param In
    Seed - An integer on the interval [1, 2^31-2]

    \return Integer error code, set to 0 if successful.
  */
  int SetSeed(unsigned int Seed);

	//@}
  
  //! Epetra_Util Sort Routine (Shell sort)
  /*! 

    This function sorts a list of integer values in ascending or descending order.  Additionally it sorts any
    number of companion lists of doubles or ints.  A shell sort is used, which is fast if indices are already sorted.
    
    \param In
           SortAscending - Sort keys in ascending order if true, otherwise sort in descending order..
    \param In
           NumKeys - Number of integer values to be sorted.
    \param In/Out
           Keys - List of integers to be sorted.
    \param In
           NumDoubleCompanions - Number of lists of double precision numbers to be sorted with the key.  If set to zero,
	   DoubleCompanions is ignored and can be set to zero.
    \param In
           DoubleCompanions - DoubleCompanions[i] is a pointer to the ith list of doubles to be sorted with key.
    \param In
           NumIntCompanions - Number of lists of integers to be sorted with the key.  If set to zero, 
	   IntCompanions is ignored and can be set to zero.
    \param In
           IntCompanions - IntCompanions[i] is a pointer to the ith list of integers to be sorted with key.
	   
  */
  static void Sort(bool SortAscending, int NumKeys, int * Keys, 
		   int NumDoubleCompanions,double ** DoubleCompanions, 
		   int NumIntCompanions, int ** IntCompanions);

  //! Epetra_Util Create_Root_Map function
  /*! Function to create a new Epetra_Map object with all GIDs sent to the root processor
      which is zero by default.  All all processors will have no GIDs.  This root map can then 
      be used to create an importer or exporter that will migrate all data to the root processor.

      If root is set to -1 then the user map will be replicated completely on all processors.
  */
  static Epetra_Map Create_Root_Map(const Epetra_Map & usermap,
					int root = 0);

  //! Epetra_Util Create_OneToOne_Map function
  /*! Function to create a new Epetra_Map object with 1-to-1 ownership of
    entries from an existing map which may have entries that appear on
    multiple processors.
  */
  static Epetra_Map Create_OneToOne_Map(const Epetra_Map& usermap,
					bool high_rank_proc_owns_shared=false);

  //! Epetra_Util Create_OneToOne_Map function
  /*! Function to create a new Epetra_Map object with 1-to-1 ownership of
    entries from an existing map which may have entries that appear on
    multiple processors.
  */
  static Epetra_BlockMap Create_OneToOne_BlockMap(const Epetra_BlockMap& usermap,
						  bool high_rank_proc_owns_shared=false);

  //! Epetra_Util Chop method.  Return zero if input Value is less than ChopValue
  static double Chop(const double & Value){
    if (std::abs(Value) < chopVal_) return 0;
    return Value;
  };

  static const double chopVal_;

 private:
	unsigned int Seed_;
};


// Epetra_Util constructor
inline Epetra_Util::Epetra_Util() : Seed_(std::rand()) {}
// Epetra_Util constructor
inline Epetra_Util::Epetra_Util(const Epetra_Util& Util) : Seed_(Util.Seed_) {}
// Epetra_Util destructor
inline Epetra_Util::~Epetra_Util(){}

/** Utility function to perform a binary-search on a list of data.
    Important assumption: data is assumed to be sorted.

    @param item to be searched for
    @param list to be searched in
    @param len Length of list
    @param insertPoint Input/Output. If item is found, insertPoint is not
    referenced. If item is not found, insertPoint is set to the offset at which
    item should be inserted in list such that order (sortedness) would be
    maintained.
    @return offset Location in list at which item was found. -1 if not found.
*/
int Epetra_Util_binary_search(int item,
                              const int* list,
                              int len,
                              int& insertPoint);

/** Function to insert an item in a list, at a specified offset.
    @return error-code 0 if successful, -1 if input parameters seem
     unreasonable (offset > usedLength, offset<0, etc).

    @param item to be inserted
    @param offset location at which to insert item
    @param list array into which item is to be inserted. This array may be
           re-allocated by this function.
    @param usedLength number of items already present in list. Will be updated
          to reflect the new length.
    @param allocatedLength current allocated length of list. Will be updated
          to reflect the new allocated-length, if applicable. Re-allocation
          occurs only if usedLength==allocatedLength on entry.
    @param allocChunkSize Optional argument, defaults to 32. Increment by
          which the array should be expanded, if re-allocation is necessary.
    @return error-code 0 if successful. -1 if input parameters don't make sense.
 */
template<class T>
int Epetra_Util_insert(T item, int offset, T*& list,
                        int& usedLength,
                        int& allocatedLength,
                        int allocChunkSize=32)
{
  if (offset < 0 || offset > usedLength) {
    return(-1);
  }

  if (usedLength < allocatedLength) {
    for(int i=usedLength; i>offset; --i) {
      list[i] = list[i-1];
    }
    list[offset] = item;
    ++usedLength;
    return(0);
  }

  T* newlist = new T[allocatedLength+allocChunkSize];
  if (newlist == NULL) {
    return(-1);
  }

  allocatedLength += allocChunkSize;
  int i;
  for(i=0; i<offset; ++i) {
    newlist[i] = list[i];
  }

  newlist[offset] = item;

  for(i=offset+1; i<=usedLength; ++i) {
    newlist[i] = list[i-1];
  }

  ++usedLength;
  delete [] list;
  list = newlist;
  return(0);
}

//! Harwell-Boeing data extraction routine
/*! This routine will extract data from an existing Epetra_Crs Matrix, and
    optionally from related rhs and lhs objects in a form that is compatible with
    software that requires the Harwell-Boeing data format. The matrix must be passed
    in, but the RHS and LHS arguments may be set to zero (either or both of them).
    For each of the LHS or RHS arguments, if non-trivial and contain more than one vector, the
    vectors must have strided access.  If both LHS and RHS are non-trivial, they must have the
    same number of vectors.  If the input objects are distributed, the returned matrices will 
    contain the local part of the matrix and vectors only.

    \param A (In) Epetra_CrsMatrix.
    \param LHS (In) Left hand side multivector.  Set to zero if none not available or needed.
    \param RHS (In) Right hand side multivector.  Set to zero if none not available or needed.
    \param M (Out) Local row dimension of matrix.
    \param N (Out) Local column dimension of matrix.
    \param nz (Out) Number of nonzero entries in matrix.
    \param ptr (Out) Offsets into ind and val arrays pointing to start of each row's data.
    \param ind (Out) Column indices of the matrix, in compressed form.
    \param val (Out) Matrix values, in compressed form corresponding to the ind array.
    \param Nrhs (Out) Number of right/left hand sides found (if any) in RHS and LHS.
    \param rhs (Out) Fortran-style 2D array of RHS values.
    \param ldrhs (Out) Stride between columns of rhs.
    \param lhs (Out) Fortran-style 2D array of LHS values.
    \param ldrhs (Out) Stride between columns of lhs.
*/
int Epetra_Util_ExtractHbData(Epetra_CrsMatrix * A, Epetra_MultiVector * LHS,
			      Epetra_MultiVector * RHS,
			      int & M, int & N, int & nz, int * & ptr,
			      int * & ind, double * & val, int & Nrhs,
			      double * & rhs, int & ldrhs,
			      double * & lhs, int & ldlhs);


#endif /* EPETRA_UTIL_H */
