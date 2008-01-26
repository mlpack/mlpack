
/*@HEADER
// ***********************************************************************
// 
//        AztecOO: An Object-Oriented Aztec Linear Solver Package 
//                 Copyright (2002) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
//@HEADER
*/

#ifndef AZTECOO_STATUSTESTMAXITERS_H
#define AZTECOO_STATUSTESTMAXITERS_H

#include "AztecOO_StatusTest.h"
class Epetra_MultiVector;

//! AztecOO_StatusTestMaxIters: An AztecOO_StatusTest class specifying a maximum number of iterations.

/* This implementation of the AztecOO_StatusTest base class tests the number of iterations performed
   against a maximum number allowed.

  \warning Presently it is not valid to associate one status test instance with two different AztecOO objects.
*/

class AztecOO_StatusTestMaxIters: public AztecOO_StatusTest {

 public:

  //@{ \name Constructors/destructors.
  //! Constructor
  AztecOO_StatusTestMaxIters(int MaxIters);

  //! Destructor
  virtual ~AztecOO_StatusTestMaxIters() {};
  //@}

  //@{ \name Methods that implement the AztecOO_StatusTest interface.

  //! Indicates if residual vector is required by this convergence test: returns false for this class.
  bool ResidualVectorRequired() const {return(false);} ;

  //! Check convergence status: Unconverged, Converged, Failed.
  /*! This method checks to see if the convergence criteria are met..

    \param CurrentIter (In) Current iteration of iterative method.  Compared against MaxIters value
    passed in at construction.  If CurrentIter < MaxIters, we return with StatusType = Unconverged.
    Otherwise, StatusType will be set to Failed.

    \param CurrentResVector (In) Ignored by this class.

    \param CurrentResNormEst (In) Ignored by this class.

    \param SolutionUpdated (In) Ignored by this class.


    \return StatusType Unconverged if CurrentIter<MaxIters, Failed if CurrentIters>=MaxIters.
  */
  AztecOO_StatusType CheckStatus(int CurrentIter, Epetra_MultiVector * CurrentResVector, 
					 double CurrentResNormEst,
				 bool SolutionUpdated);
  AztecOO_StatusType GetStatus() const {return(status_);};

  ostream& Print(ostream& stream, int indent = 0) const;
  //@}
  
  //@{ \name Methods to access data members.

  //! Returns the maximum number of iterations set in the constructor.
  int GetMaxIters() const {return(MaxIters_);};

  //! Returns the current number of iterations from the most recent StatusTest call.
  int GetNumIters() const {return(Niters_);};

  //@}

private:

  //@{ \name Private data members.
  //! Maximum number of iterations allowed
  int MaxIters_;

  //! Current number of iterations
  int Niters_;

  //! Status
  AztecOO_StatusType status_;
  //@}

};

#endif /* AZTECOO_STATUSTESTMAXITERS_H */
