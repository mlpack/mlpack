
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

#ifndef EPETRA_FEVECTOR_H
#define EPETRA_FEVECTOR_H

#include <Epetra_CombineMode.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
class Epetra_IntSerialDenseVector;
class Epetra_SerialDenseVector;

/** Epetra Finite-Element Vector. This class inherits Epetra_MultiVector
  and thus provides all Epetra_MultiVector functionality, with one
  restriction: currently an Epetra_FEVector only has 1 internal vector.

  The added functionality provided by Epetra_FEVector is the ability to
  perform finite-element style vector assembly. It accepts sub-vector
  contributions, such as those that would come from element-load vectors, etc.,
  and these sub-vectors need not be wholly locally owned. In other words, the
  user can assemble overlapping data (e.g., corresponding to shared
  finite-element nodes). When the user is finished assembling their vector
  data, they then call the method Epetra_FEVector::GlobalAssemble() which
  gathers the overlapping data (all non-local data that was input on each
  processor) into the data-distribution specified by the map that the
  Epetra_FEVector is constructed with.

  Note: At the current time (Sept 6, 2002) the methods in this implementation
  assume that there is only 1 point associated with each map element. This 
  limitation will be removed in the near future.
*/

class Epetra_FEVector : public Epetra_MultiVector {
 public:
   /** Constructor that requires a map specifying a non-overlapping
      data layout. The methods SumIntoGlobalValues() and 
      ReplaceGlobalValues() will accept any global IDs, and GlobalAssemble()
      will move any non-local data onto the appropriate owning processors.
   */
   Epetra_FEVector(const Epetra_BlockMap& Map,
		   bool ignoreNonLocalEntries=false);

  /** Copy constructor. */
  Epetra_FEVector(const Epetra_FEVector& source);

   /** Destructor */
   virtual ~Epetra_FEVector();

   /** Accumulate values into the vector, adding them to any values that
       already exist for the specified indices.
   */
   int SumIntoGlobalValues(int numIDs, const int* GIDs, const double* values);

   /** Accumulate values into the vector, adding them to any values that
       already exist for the specified GIDs.

       @param GIDs List of global ids. Must be the same length as the
       accompanying list of values.

       @param values List of coefficient values. Must be the same length as
       the accompanying list of GIDs.
   */
   int SumIntoGlobalValues(const Epetra_IntSerialDenseVector& GIDs,
			   const Epetra_SerialDenseVector& values);

   /** Copy values into the vector overwriting any values that already exist
        for the specified indices.
    */
   int ReplaceGlobalValues(int numIDs, const int* GIDs, const double* values);

   /** Copy values into the vector, replacing any values that
       already exist for the specified GIDs.

       @param GIDs List of global ids. Must be the same length as the
       accompanying list of values.

       @param values List of coefficient values. Must be the same length as
       the accompanying list of GIDs.
   */
   int ReplaceGlobalValues(const Epetra_IntSerialDenseVector& GIDs,
			   const Epetra_SerialDenseVector& values);

   int SumIntoGlobalValues(int numIDs, const int* GIDs,
			   const int* numValuesPerID,
			   const double* values);

   int ReplaceGlobalValues(int numIDs, const int* GIDs,
			   const int* numValuesPerID,
			   const double* values);

   /** Gather any overlapping/shared data into the non-overlapping partitioning
      defined by the Map that was passed to this vector at construction time.
      Data imported from other processors is stored on the owning processor
      with a "sumInto" or accumulate operation.
      This is a collective method -- every processor must enter it before any
      will complete it.
   */
   int GlobalAssemble(Epetra_CombineMode mode = Add);

   /** Set whether or not non-local data values should be ignored.
    */
   void setIgnoreNonLocalEntries(bool flag) {
     ignoreNonLocalEntries_ = flag;
   }

   Epetra_FEVector& operator=(const Epetra_FEVector& source);

 private:
  int inputValues(int numIDs,
                  const int* GIDs, const double* values,
                  bool accumulate);

  int inputValues(int numIDs,
                  const int* GIDs, const int* numValuesPerID,
		  const double* values,
                  bool accumulate);

  int inputNonlocalValue(int GID, double value, bool accumulate);

  int inputNonlocalValues(int GID, int numValues, const double* values,
			  bool accumulate);

  void destroyNonlocalData();

  int myFirstID_;
  int myNumIDs_;
  double* myCoefs_;

  int* nonlocalIDs_;
  int* nonlocalElementSize_;
  int numNonlocalIDs_;
  int allocatedNonlocalLength_;
  double** nonlocalCoefs_;

  bool ignoreNonLocalEntries_;
};

#endif

