//@HEADER
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

#ifndef AZTECOO_CONDITIONNUMBER_H
#define AZTECOO_CONDITIONNUMBER_H

class Epetra_Map;
class Epetra_Vector;
class Epetra_Operator;
class AztecOO;

/*!

 \brief Condition number estimator using AztecOO.

 This object will estimate the condition number of an Epetra_Operator.
 */
class AztecOOConditionNumber {

 public:
  
  //! Solver type to use.
  enum SolverType {
    //! CG for symmetric matrices
    CG_, 
    //! GMRES for nonsymmetric
    GMRES_
  };
  
  //! Constructor.  
  AztecOOConditionNumber();
  
  //! Destructor
  ~AztecOOConditionNumber();
  
  //! Initialization
  void initialize(const Epetra_Operator& op,
		  SolverType solverType=GMRES_,
		  int krylovSubspaceSize=100, 
		  bool printSolve=false);
  
  //! Estimates the condition number. 
  int computeConditionNumber(int maxIters, double tol);

  //! Return condition number computed by last call to computeConditionNumber.
  double getConditionNumber();

 protected:

  //! Frees all memory allocated with new by this object.
  void freeMemory();

 protected:

  //! Condition number calculated in computeConditionNumber.
  double conditionNumber_;

  //! Map to create left hand side vector.  
  Epetra_Map* domainMap_;

  //! Map to create right hand side vector.  
  Epetra_Map* rangeMap_;

  //! Operator supplied by user in initialization.
  Epetra_Operator* operator_;

  //! RHS vector.  This is initializaed to a random vector.
  Epetra_Vector* rhs_;

  //! Dummy vector.  Initializaed to zero.
  Epetra_Vector* dummy_;

  //! solver object.
  AztecOO* solver_;

  //! Conditional for printing solve to output.
  bool printSolve_;

};


#endif
