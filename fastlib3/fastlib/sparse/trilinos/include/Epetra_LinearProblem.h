
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

#ifndef EPETRA_LINEARPROBLEM_H
#define EPETRA_LINEARPROBLEM_H

#include "Epetra_RowMatrix.h"
#include "Epetra_Operator.h"
#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum ProblemDifficultyLevel {easy, moderate, hard, unsure};
#endif

//! Epetra_LinearProblem:  The Epetra Linear Problem Class.
/*! The Epetra_LinearProblem class is a wrapper that encapsulates the 
  general information needed for solving a linear system of equations.  
  Currently it accepts a Epetra matrix, initial guess and RHS and
  returns the solution.
  the elapsed time for each calling processor.
*/


class Epetra_LinearProblem {
    
  public:
    //! @name Constructors/Destructor
  //@{ 
  //!  Epetra_LinearProblem Default Constructor.
  /*! Creates an empty Epetra_LinearProblem instance. The operator A, left-hand-side X
      and right-hand-side B must be set use the SetOperator(), SetLHS() and SetRHS()
      methods respectively.
  */
  Epetra_LinearProblem(void);

  //!  Epetra_LinearProblem Constructor to pass in an operator as a matrix.
  /*! Creates a Epetra_LinearProblem instance where the operator is passed in as a matrix. 
  */
  Epetra_LinearProblem(Epetra_RowMatrix * A, Epetra_MultiVector * X,
			 Epetra_MultiVector * B);

  //!  Epetra_LinearProblem Constructor to pass in a basic Epetra_Operator.
  /*! Creates a Epetra_LinearProblem instance for the case where an operator is not necessarily a matrix. 
  */
  Epetra_LinearProblem(Epetra_Operator * A, Epetra_MultiVector * X,
                       Epetra_MultiVector * B);
  //! Epetra_LinearProblem Copy Constructor.
  /*! Makes copy of an existing Epetra_LinearProblem instance.
  */
  Epetra_LinearProblem(const Epetra_LinearProblem& Problem);

  //! Epetra_LinearProblem Destructor.
  /*! Completely deletes a Epetra_LinearProblem object.  
  */
  virtual ~Epetra_LinearProblem(void);
  //@}
  
  //! @name Integrity check method
  //@{ 

  //! Check input parameters for existence and size consistency.
  /*! Returns 0 if all input parameters are valid.  Returns +1 if operator is not a matrix. 
      This is not necessarily an error, but no scaling can be done if the user passes in an
      Epetra_Operator that is not an Epetra_Matrix 
  */
  int CheckInput() const;
  //@}
  
  //! @name Set methods
  //@{ 

  void AssertSymmetric(){OperatorSymmetric_ = true;};
#ifdef DOXYGEN_SHOULD_SKIP_THIS
  enum ProblemDifficultyLevel {easy, moderate, hard, unsure};
#endif
  //! Set problem difficulty level.
  /*! Sets Aztec options and parameters based on a definition of easy moderate or hard problem.
      Relieves the user from explicitly setting a large number of individual parameter values.
      This function can be used in conjunction with the SetOptions() and SetParams() functions.
  */
  void SetPDL(ProblemDifficultyLevel PDL) {PDL_ = PDL;};

  //! Set Operator A of linear problem AX = B using an Epetra_RowMatrix.
  /*! Sets a pointer to a Epetra_RowMatrix.  No copy of the operator is made.
  */
  void SetOperator(Epetra_RowMatrix * A)
    { A_ = A; Operator_ = A; }

  //! Set Operator A of linear problem AX = B using an Epetra_Operator.
  /*! Sets a pointer to a Epetra_Operator.  No copy of the operator is made.
  */
  void SetOperator(Epetra_Operator * A)
    { A_ = dynamic_cast<Epetra_RowMatrix *>(A); Operator_ = A; }

  //! Set left-hand-side X of linear problem AX = B.
  /*! Sets a pointer to a Epetra_MultiVector.  No copy of the object is made.
  */
  void SetLHS(Epetra_MultiVector * X) {X_ = X;}

  //! Set right-hand-side B of linear problem AX = B.
  /*! Sets a pointer to a Epetra_MultiVector.  No copy of the object is made.
  */
  void SetRHS(Epetra_MultiVector * B) {B_ = B;}
  //@}
  
  //! @name Computational methods
  //@{ 
  //! Perform left scaling of a linear problem.
  /*! Applies the scaling vector D to the left side of the matrix A() and
    to the right hand side B().  Note that the operator must be an Epetra_RowMatrix,
      not just an Epetra_Operator (the base class of Epetra_RowMatrix).
    \param In
           D - Vector containing scaling values.  D[i] will be applied 
               to the ith row of A() and B().
    \return Integer error code, set to 0 if successful. Return -1 if operator is not a matrix.
  */
  int LeftScale(const Epetra_Vector & D);

  //! Perform right scaling of a linear problem. 
  /*! Applies the scaling vector D to the right side of the matrix A().
      Apply the inverse of D to the initial guess.  Note that the operator must be an Epetra_RowMatrix,
      not just an Epetra_Operator (the base class of Epetra_RowMatrix).
    \param In
           D - Vector containing scaling values.  D[i] will be applied 
               to the ith row of A().  1/D[i] will be applied to the
               ith row of B().
    \return Integer error code, set to 0 if successful. Return -1 if operator is not a matrix.
  */
  int RightScale(const Epetra_Vector & D);
  //@}

  //! @name Accessor methods
  //@{ 
  //! Get a pointer to the operator A.
  Epetra_Operator * GetOperator() const {return(Operator_);};
  //! Get a pointer to the matrix A.
  Epetra_RowMatrix * GetMatrix() const {return(A_);};
  //! Get a pointer to the left-hand-side X.
  Epetra_MultiVector * GetLHS() const {return(X_);};
  //! Get a pointer to the right-hand-side B.
  Epetra_MultiVector * GetRHS() const {return(B_);};
  //! Get problem difficulty level.
  ProblemDifficultyLevel GetPDL() const {return(PDL_);};
  //! Get operator symmetry bool.
  bool IsOperatorSymmetric() const {return(OperatorSymmetric_);};
  //@}

 private:

  Epetra_Operator * Operator_;
  Epetra_RowMatrix * A_;
  Epetra_MultiVector * X_;
  Epetra_MultiVector * B_;

  bool OperatorSymmetric_;
  ProblemDifficultyLevel PDL_;
  bool LeftScaled_;
  bool RightScaled_;
  Epetra_Vector * LeftScaleVector_;
  Epetra_Vector * RightScaleVector_;
  Epetra_LinearProblem & operator=(const Epetra_LinearProblem& Problem);
};

#endif /* EPETRA_LINEARPROBLEM_H */
