
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

#ifndef EPETRA_SERIALSPDDENSESOLVER_H
#define EPETRA_SERIALSPDDENSESOLVER_H
#include "Epetra_SerialDenseSolver.h"
class Epetra_SerialSymDenseMatrix;

//! Epetra_SerialSpdDenseSolver: A class for constructing and using symmetric positive definite dense matrices.

/*! The Epetra_SerialSpdDenseSolver class enables the construction and use of real-valued, symmetric positive definite, 
    double-precision dense matrices.  It is built on the Epetra_DenseMatrix class which in turn is built on the 
    BLAS and LAPACK via the Epetra_BLAS and 
    Epetra_LAPACK classes. 

The Epetra_SerialSpdDenseSolver class is intended to provide full-featured support for solving linear and eigen system
problems for symmetric positive definite matrices.  It is written on top of BLAS and LAPACK and thus has excellent
performance and numerical capabilities.  Using this class, one can either perform simple factorizations and solves or
apply all the tricks available in LAPACK to get the best possible solution for very ill-conditioned problems.

<b>Epetra_SerialSpdDenseSolver vs. Epetra_LAPACK</b>

The Epetra_LAPACK class provides access to most of the same functionality as Epetra_SerialSpdDenseSolver.
The primary difference is that Epetra_LAPACK is a "thin" layer on top of LAPACK and Epetra_SerialSpdDenseSolver
attempts to provide easy access to the more sophisticated aspects of solving dense linear and eigensystems.
<ul>
<li> When you should use Epetra_LAPACK:  If you are simply looking for a convenient wrapper around the Fortran LAPACK
     routines and you have a well-conditioned problem, you should probably use Epetra_LAPACK directly.
<li> When you should use Epetra_SerialSpdDenseSolver: If you want to (or potentially want to) solve ill-conditioned 
     problems or want to work with a more object-oriented interface, you should probably use Epetra_SerialSpdDenseSolver.
     
</ul>

<b>Constructing Epetra_SerialSpdDenseSolver Objects</b>

There are three Epetra_DenseMatrix constructors.  The first constructs a zero-sized object which should be made
to appropriate length using the Shape() or Reshape() functions and then filled with the [] or () operators. 
The second is a constructor that accepts user
data as a 2D array, the third is a copy constructor. The second constructor has
two data access modes (specified by the Epetra_DataAccess argument):
<ol>
  <li> Copy mode - Allocates memory and makes a copy of the user-provided data. In this case, the
       user data is not needed after construction.
  <li> View mode - Creates a "view" of the user data. In this case, the
       user data is required to remain intact for the life of the object.
</ol>

\warning View mode is \e extremely dangerous from a data hiding perspective.
Therefore, we strongly encourage users to develop code using Copy mode first and 
only use the View mode in a secondary optimization phase.

<b>Setting vectors used for linear solves</b>

Setting the X and B vectors (which are Epetra_DenseMatrix objects) used for solving linear systems 
is done separately from the constructor.  This allows
a single matrix factor to be used for multiple solves.  Similar to the constructor, the vectors X and B can
be copied or viewed using the Epetra_DataAccess argument.

<b>Extracting Data from Epetra_SerialSpdDenseSolver Objects</b>

Once a Epetra_SerialSpdDenseSolver is constructed, it is possible to view the data via access functions.

\warning Use of these access functions cam be \e extremely dangerous from a data hiding perspective.


<b>Vector and Utility Functions</b>

Once a Epetra_SerialSpdDenseSolver is constructed, several mathematical functions can be applied to
the object.  Specifically:
<ul>
  <li> Factorizations.
  <li> Solves.
  <li> Condition estimates.
  <li> Equilibration.
  <li> Norms.
</ul>

The final useful function is Flops().  Each Epetra_SerialSpdDenseSolver object keep track of the number
of \e serial floating point operations performed using the specified object as the \e this argument
to the function.  The Flops() function returns this number as a double precision number.  Using this 
information, in conjunction with the Epetra_Time class, one can get accurate parallel performance
numbers.

<b>Strategies for Solving Linear Systems</b>
In many cases, linear systems can be accurately solved by simply computing the Cholesky factorization
of the matrix and then performing a forward back solve with a given set of right hand side vectors.  However,
in some instances, the factorization may be very poorly conditioned and the simple approach may not work.  In
these situations, equilibration and iterative refinement may improve the accuracy, or prevent a breakdown in
the factorization. 

Epetra_SerialSpdDenseSolver will use equilibration with the factorization if, once the object
is constructed and \e before it is factored, you call the function FactorWithEquilibration(true) to force 
equilibration to be used.  If you are uncertain if equilibration should be used, you may call the function
ShouldEquilibrate() which will return true if equilibration could possibly help.  ShouldEquilibrate() uses
guidelines specified in the LAPACK User Guide, namely if SCOND < 0.1 and AMAX < Underflow or AMAX > Overflow, to 
determine if equilibration \e might be useful. 
 
Epetra_SerialSpdDenseSolver will use iterative refinement after a forward/back solve if you call
SolveToRefinedSolution(true).  It will also compute forward and backward error estimates if you call
EstimateSolutionErrors(true).  Access to the forward (back) error estimates is available via FERR() (BERR()).

Examples using Epetra_SerialSpdDenseSolver can be found in the Epetra test directories.

*/

//=========================================================================
class Epetra_SerialSpdDenseSolver : public Epetra_SerialDenseSolver {

 public:
   //! @name Constructor/Destructor Methods
  //@{ 
  //! Default constructor; matrix should be set using SetMatrix(), LHS and RHS set with SetVectors().
  Epetra_SerialSpdDenseSolver();
  

  //! Epetra_SerialDenseSolver destructor.  
  virtual ~Epetra_SerialSpdDenseSolver();
  //@}

  //! @name Set Methods
  //@{ 

  //Let the compiler know we intend to overload the SetMatrix function,
  //rather than hide it.
  using Epetra_SerialDenseSolver::SetMatrix;

  //! Sets the pointers for coefficient matrix; special version for symmetric matrices
  int SetMatrix(Epetra_SerialSymDenseMatrix & A);
  //@}
  
  //! @name Factor/Solve/Invert Methods
  //@{ 

  //! Computes the in-place Cholesky factorization of the matrix using the LAPACK routine \e DPOTRF.
  /*!
    \return Integer error code, set to 0 if successful.
  */
  int Factor(void);

  //! Computes the solution X to AX = B for the \e this matrix and the B provided to SetVectors()..
  /*!
    \return Integer error code, set to 0 if successful.
  */
  int Solve(void);

  //! Inverts the \e this matrix.
  /*! Note: This function works a little differently that DPOTRI in that it fills the entire
      matrix with the inverse, independent of the UPLO specification.

    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int Invert(void);

  //! Computes the scaling vector S(i) = 1/sqrt(A(i,i) of the \e this matrix.
  /*! 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int ComputeEquilibrateScaling(void);

  //! Equilibrates the \e this matrix.
  /*! 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int EquilibrateMatrix(void);

  //! Equilibrates the current RHS.
  /*! 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int EquilibrateRHS(void);


  //! Apply Iterative Refinement.
  /*! 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int ApplyRefinement(void);

  //! Unscales the solution vectors if equilibration was used to solve the system.
  /*! 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int UnequilibrateLHS(void);

  //! Returns the reciprocal of the 1-norm condition number of the \e this matrix.
  /*! 
    \param Value Out
           On return contains the reciprocal of the 1-norm condition number of the \e this matrix.
    
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  int ReciprocalConditionEstimate(double & Value);
  //@}

  //! @name Query methods
  //@{ 


  //! Returns true if the LAPACK general rules for equilibration suggest you should equilibrate the system.
  bool ShouldEquilibrate() {ComputeEquilibrateScaling(); return(ShouldEquilibrate_);};
  //@}

  //! @name Data Accessor methods
  //@{ 
    
  //! Returns pointer to current matrix.
  Epetra_SerialSymDenseMatrix * SymMatrix()  const {return(SymMatrix_);};
       
  //! Returns pointer to factored matrix (assuming factorization has been performed).
  Epetra_SerialSymDenseMatrix * SymFactoredMatrix()  const {return(SymFactor_);};

  //! Ratio of smallest to largest equilibration scale factors for the \e this matrix (returns -1 if not yet computed).
  /*! If SCOND() is >= 0.1 and AMAX() is not close to overflow or underflow, then equilibration is not needed.
   */
  double SCOND() {return(SCOND_);};

  //Let the compiler know we intend to overload the AMAX function,
  //rather than hide it.
  using Epetra_SerialDenseSolver::AMAX;

  //! Returns the absolute value of the largest entry of the \e this matrix (returns -1 if not yet computed).
  double AMAX() {return(AMAX_);};  
  //@}

 private:

  double SCOND_;
  Epetra_SerialSymDenseMatrix * SymMatrix_; // Need pointer to symmetric matrix for Spd-specific methods
  Epetra_SerialSymDenseMatrix * SymFactor_; // Need pointer to symmetric matrix for Spd-specific methods

  // Epetra_SerialSpdDenseSolver copy constructor (put here because we don't want user access)
  
  Epetra_SerialSpdDenseSolver(const Epetra_SerialSpdDenseSolver& Source);
  Epetra_SerialSpdDenseSolver & operator=(const Epetra_SerialSpdDenseSolver& Source);
};

#endif /* EPETRA_SERIALSPDDENSESOLVER_H */
