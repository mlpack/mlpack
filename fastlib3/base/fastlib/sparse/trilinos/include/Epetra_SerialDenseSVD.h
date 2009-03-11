
/* Copyright (2001) Sandia Corportation. Under the terms of Contract 
 * DE-AC04-94AL85000, there is a non-exclusive license for use of this 
 * work by or on behalf of the U.S. Government.  Export of this program
 * may require a license from the United States Government. */


/* NOTICE:  The United States Government is granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
 * license in ths data to reproduce, prepare derivative works, and
 * perform publicly and display publicly.  Beginning five (5) years from
 * July 25, 2001, the United States Government is granted for itself and
 * others acting on its behalf a paid-up, nonexclusive, irrevocable
 * worldwide license in this data to reproduce, prepare derivative works,
 * distribute copies to the public, perform publicly and display
 * publicly, and to permit others to do so.
 * 
 * NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT
 * OF ENERGY, NOR SANDIA CORPORATION, NOR ANY OF THEIR EMPLOYEES, MAKES
 * ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
 * RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
 * INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
 * THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS. */

#ifndef _EPETRA_SERIALDENSESVD_H_
#define _EPETRA_SERIALDENSESVD_H_

#include "Epetra_SerialDenseOperator.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_Object.h" 
#include "Epetra_CompObject.h"
#include "Epetra_BLAS.h"
#include "Epetra_LAPACK.h"


//! Epetra_SerialDenseSVD: A class for SVDing dense linear problems.

/*! The Epetra_SerialDenseSVD class enables the definition, in terms of Epetra_SerialDenseMatrix 
    and Epetra_SerialDenseVector objects, of a dense linear problem, followed by the solution of that problem via the
    most sophisticated techniques available in LAPACK.

The Epetra_SerialDenseSVD class is intended to provide full-featured support for solving linear 
problems for general dense rectangular (or square) matrices.  It is written on top of BLAS and LAPACK and thus has excellent
performance and numerical capabilities.  Using this class, one can either perform simple factorizations and solves or
apply all the tricks available in LAPACK to get the best possible solution for very ill-conditioned problems.

<b>Epetra_SerialDenseSVD vs. Epetra_LAPACK</b>

The Epetra_LAPACK class provides access to most of the same functionality as Epetra_SerialDenseSolver.
The primary difference is that Epetra_LAPACK is a "thin" layer on top of LAPACK and Epetra_SerialDenseSolver
attempts to provide easy access to the more sophisticated aspects of solving dense linear and eigensystems.
<ul>
<li> When you should use Epetra_LAPACK:  If you are simply looking for a convenient wrapper around the Fortran LAPACK
     routines and you have a well-conditioned problem, you should probably use Epetra_LAPACK directly.
<li> When you should use Epetra_SerialDenseSolver: If you want to (or potentially want to) solve ill-conditioned 
     problems or want to work with a more object-oriented interface, you should probably use Epetra_SerialDenseSolver.
     
</ul>

<b>Constructing Epetra_SerialDenseSVD Objects</b>

There is a single Epetra_SerialDenseSVD constructor.   However, the matrix, right hand side and solution 
vectors must be set prior to executing most methods in this class.

<b>Setting vectors used for linear solves</b>

The matrix A, the left hand side X and the right hand side B (when solving AX = B, for X), can be set by appropriate set
methods.  Each of these three objects must be an Epetra_SerialDenseMatrix or and Epetra_SerialDenseVector object.  The
set methods are as follows:
<ul>
<li> SetMatrix() - Sets the matrix.
<li> SetVectors() - Sets the left and right hand side vector(s).
</ul>

<b>Vector and Utility Functions</b>

Once a Epetra_SerialDenseSVD is constructed, several mathematical functions can be applied to
the object.  Specifically:
<ul>
  <li> Factorizations.
  <li> Solves.
  <li> Condition estimates.
  <li> Norms.
</ul>

<b>Counting floating point operations </b>
The Epetra_SerialDenseSVD class has Epetra_CompObject as a base class.  Thus, floating point operations 
are counted and accumulated in the Epetra_Flop object (if any) that was set using the SetFlopCounter()
method in the Epetra_CompObject base class.

Examples using Epetra_SerialDenseSVD can be found in the Epetra test directories.

*/

//=========================================================================
class Epetra_SerialDenseSVD : public virtual Epetra_SerialDenseOperator, public Epetra_CompObject, public virtual Epetra_Object, public Epetra_BLAS, public Epetra_LAPACK{
  public:
  
    //! @name Constructor/Destructor Methods
  //@{ 
  //! Default constructor; matrix should be set using SetMatrix(), LHS and RHS set with SetVectors().
  Epetra_SerialDenseSVD();
  
  //! Epetra_SerialDenseSVD destructor.  
  virtual ~Epetra_SerialDenseSVD();
  //@}

  //! @name Set Methods
  //@{ 

  //! Sets the pointers for coefficient matrix
  int SetMatrix(Epetra_SerialDenseMatrix & A);

  //! Sets the pointers for left and right hand side vector(s).
  /*! Row dimension of X must match column dimension of matrix A, row dimension of B 
      must match row dimension of A.  X and B must have the same dimensions.
  */
  int SetVectors(Epetra_SerialDenseMatrix & X, Epetra_SerialDenseMatrix & B);
  //@}

  //! @name Strategy modifying Methods
  //@{ 

  //! Causes equilibration to be called just before the matrix factorization as part of the call to Factor.
  /*! This function must be called before the factorization is performed. 
   */
//  void FactorWithEquilibration(bool Flag) {Equilibrate_ = Flag; return;};

  //! If Flag is true, causes all subsequent function calls to work with the transpose of \e this matrix, otherwise not.
  void SolveWithTranspose(bool Flag) {Transpose_ = Flag; if (Flag) TRANS_ = 'T'; else TRANS_ = 'N'; return;};

  //! Causes all solves to compute solution to best ability using iterative refinement.
//  void SolveToRefinedSolution(bool Flag) {RefineSolution_ = Flag; return;};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Causes all solves to estimate the forward and backward solution error. 
  /* Error estimates will be in the arrays FERR and BERR, resp, after the solve step is complete.
      These arrays are accessible via the FERR() and BERR() access functions.
  */
//  void EstimateSolutionErrors(bool Flag) {EstimateSolutionErrors_ = Flag; return;};
  //@}

  //! @name Factor/Solve/Invert Methods
  //@{ 

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Computes the SVD factorization of the matrix using the LAPACK routine \e DGESVD.
  /* 
    \return Integer error code, set to 0 if successful.
  */
//  virtual int Factor(void);
  virtual int Factor(void);

  //! Computes the solution X to AX = B for the \e this matrix and the B provided to SetVectors()..
  /*! Inverse of Matrix must be formed
    \return Integer error code, set to 0 if successful.
  */
  virtual int Solve(void);

  //! Inverts the \e this matrix.
  /*!
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
  virtual int Invert( double rthresh = 0.0, double athresh = 0.0 );

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Computes the scaling vector S(i) = 1/sqrt(A(i,i) of the \e this matrix.
  /* 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
//  virtual int ComputeEquilibrateScaling(void);

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Equilibrates the \e this matrix.
  /* 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
//  virtual int EquilibrateMatrix(void);

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Equilibrates the current RHS.
  /* 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
//  int EquilibrateRHS(void);


  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Apply Iterative Refinement.
  /* 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
//  virtual int ApplyRefinement(void);

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Unscales the solution vectors if equilibration was used to solve the system.
  /* 
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
//  int UnequilibrateLHS(void);

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns the reciprocal of the 1-norm condition number of the \e this matrix.
  /* 
    \param Value Out
           On return contains the reciprocal of the 1-norm condition number of the \e this matrix.
    
    \return Integer error code, set to 0 if successful. Otherwise returns the LAPACK error code INFO.
  */
//  virtual int ReciprocalConditionEstimate(double & Value);
  //@}

  //! @name Query methods
  //@{ 

  //! Returns true if transpose of \e this matrix has and will be used.
  bool Transpose() {return(Transpose_);};

  //! Returns true if matrix is factored (factor available via AF() and LDAF()).
  bool Factored() {return(Factored_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns true if factor is equilibrated (factor available via AF() and LDAF()).
//  bool A_Equilibrated() {return(A_Equilibrated_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns true if RHS is equilibrated (RHS available via B() and LDB()).
//  bool B_Equilibrated() {return(B_Equilibrated_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns true if the LAPACK general rules for equilibration suggest you should equilibrate the system.
//  virtual bool ShouldEquilibrate() {ComputeEquilibrateScaling(); return(ShouldEquilibrate_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns true if forward and backward error estimated have been computed (available via FERR() and BERR()).
//  bool SolutionErrorsEstimated() {return(SolutionErrorsEstimated_);};

  //! Returns true if matrix inverse has been computed (inverse available via AF() and LDAF()).
  bool Inverted() {return(Inverted_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns true if the condition number of the \e this matrix has been computed (value available via ReciprocalConditionEstimate()).
//  bool ReciprocalConditionEstimated() {return(ReciprocalConditionEstimated_);};

  //! Returns true if the current set of vectors has been solved.
  bool Solved() {return(Solved_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns true if the current set of vectors has been refined.
//  bool SolutionRefined() {return(SolutionRefined_);};
  //@}

  //! @name Data Accessor methods
  //@{ 
    
  //! Returns pointer to current matrix.
   Epetra_SerialDenseMatrix * Matrix()  const {return(Matrix_);};
       
  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns pointer to factored matrix (assuming factorization has been performed).
//   Epetra_SerialDenseMatrix * FactoredMatrix()  const {return(Factor_);};

  //! Returns pointer to inverted matrix (assuming inverse has been performed).
   Epetra_SerialDenseMatrix * InvertedMatrix()  const {return(Inverse_);};

  //! Returns pointer to current LHS.
  Epetra_SerialDenseMatrix * LHS()  const {return(LHS_);};
    
  //! Returns pointer to current RHS.
  Epetra_SerialDenseMatrix * RHS()  const {return(RHS_);};
    
  //! Returns row dimension of system.
  int M()  const {return(M_);};

  //! Returns column dimension of system.
  int N()  const {return(N_);};

  //! Returns pointer to the \e this matrix.
  double * A()  const {return(A_);};

  //! Returns the leading dimension of the \e this matrix.
  int LDA()  const {return(LDA_);};

  //! Returns pointer to current RHS.
  double * B()  const {return(B_);};

  //! Returns the leading dimension of the RHS.
  int LDB()  const {return(LDB_);};

  //! Returns the number of current right hand sides and solution vectors.
  int NRHS()  const {return(NRHS_);};

  //! Returns pointer to current solution.
  double * X()  const {return(X_);};

  //! Returns the leading dimension of the solution.
  int LDX()  const {return(LDX_);};

  double * S() const {return(S_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns pointer to the factored matrix (may be the same as A() if factorization done in place).
//  double * AF()  const {return(AF_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns the leading dimension of the factored matrix.
//  int LDAF()  const {return(LDAF_);};

  //! Returns pointer to the inverted matrix (may be the same as A() if factorization done in place).
  double * AI()  const {return(AI_);};

  //! Returns the leading dimension of the inverted matrix.
  int LDAI()  const {return(LDAI_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns pointer to pivot vector (if factorization has been computed), zero otherwise.
//  int * IPIV()  const {return(IPIV_);};

  //! Returns the 1-Norm of the \e this matrix (returns -1 if not yet computed).
  double ANORM()  const {return(ANORM_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns the reciprocal of the condition number of the \e this matrix (returns -1 if not yet computed).
//  double RCOND()  const {return(RCOND_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Ratio of smallest to largest row scale factors for the \e this matrix (returns -1 if not yet computed).
  /* If ROWCND() is >= 0.1 and AMAX() is not close to overflow or underflow, then equilibration is not needed.
   */
//  double ROWCND()  const {return(ROWCND_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Ratio of smallest to largest column scale factors for the \e this matrix (returns -1 if not yet computed).
  /* If COLCND() is >= 0.1 then equilibration is not needed.
   */
//  double COLCND()  const {return(COLCND_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns the absolute value of the largest entry of the \e this matrix (returns -1 if not yet computed).
//  double AMAX()  const {return(AMAX_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns a pointer to the forward error estimates computed by LAPACK.
//  double * FERR()  const {return(FERR_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns a pointer to the backward error estimates computed by LAPACK.
//  double * BERR()  const {return(BERR_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns a pointer to the row scaling vector used for equilibration.
//  double * R()  const {return(R_);};

  // NOTE: doxygen-style documentation needs to be re-enabled if this function is re-enabled
  // Returns a pointer to the column scale vector used for equilibration.
//  double * C()  const {return(C_);};
  //@}

  //! @name I/O methods
  //@{ 
  //! Print service methods; defines behavior of ostream << operator.
  virtual void Print(ostream& os) const;
  //@}

  //! @name Additional methods for support of Epetra_SerialDenseOperator interface
  //@{ 

    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
	does not support transpose use, this method should return a value of -1.
      
    \param In
	   UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.

    \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does not support transpose.
  */
    virtual int SetUseTranspose(bool UseTranspose) { UseTranspose_ = UseTranspose; return (0); }

    //! Returns the result of a Epetra_SerialDenseOperator applied to a Epetra_SerialDenseMatrix X in Y.
    /*! 
    \param In
	   X - A Epetra_SerialDenseMatrix to multiply with operator.
    \param Out
	   Y -A Epetra_SerialDenseMatrix containing result.

    \return Integer error code, set to 0 if successful.
  */
    virtual int Apply(const Epetra_SerialDenseMatrix& X, Epetra_SerialDenseMatrix& Y)
    { return Y.Multiply( UseTranspose_, false, 1.0, *Matrix(), X, 0.0 ); }

    //! Returns the result of a Epetra_SerialDenseOperator inverse applied to an Epetra_SerialDenseMatrix X in Y.
    /*! 
    \param In
	   X - A Epetra_SerialDenseMatrix to solve for.
    \param Out
	   Y -A Epetra_SerialDenseMatrix containing result.

    \return Integer error code, set to 0 if successful.

  */
    virtual int ApplyInverse(const Epetra_SerialDenseMatrix & X, Epetra_SerialDenseMatrix & Y)
    { SetVectors(const_cast<Epetra_SerialDenseMatrix&>(X),Y);
      SolveWithTranspose(UseTranspose_);
      return Solve(); }

    //! Returns the infinity norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_\infty\f$ such that
       \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].

       \warning This method must not be called unless HasNormInf() returns true.
    */ 
    virtual double NormInf() const { return Matrix()->NormInf(); }
  
    //! Returns a character string describing the operator
    virtual const char * Label() const { return Epetra_Object::Label(); }

    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const { return UseTranspose_; }

    //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
    virtual bool HasNormInf() const { return true; }

    //! Returns the row dimension of operator
    virtual int RowDim() const { return M(); }

    //! Returns the column dimension of operator
    virtual int ColDim() const { return N(); }

  //@}
  
  void AllocateWORK() {if (WORK_==0) {LWORK_ = 4*N_; WORK_ = new double[LWORK_];} return;};
  void AllocateIWORK() {if (IWORK_==0) IWORK_ = new int[N_]; return;};
  void InitPointers();
  void DeleteArrays();
  void ResetMatrix();
  void ResetVectors();


//  bool Equilibrate_;
//  bool ShouldEquilibrate_;
//  bool A_Equilibrated_;
//  bool B_Equilibrated_;
  bool Transpose_;
  bool Factored_;
//  bool EstimateSolutionErrors_;
//  bool SolutionErrorsEstimated_;
  bool Solved_;
  bool Inverted_;
//  bool ReciprocalConditionEstimated_;
//  bool RefineSolution_;
//  bool SolutionRefined_;

  char TRANS_;

  int M_;
  int N_;
  int Min_MN_;
  int NRHS_;
  int LDA_;
//  int LDAF_;
  int LDAI_;
  int LDB_;
  int LDX_;
  int INFO_;
  int LWORK_;

//  int * IPIV_;
  int * IWORK_;

  double ANORM_;
//  double RCOND_;
//  double ROWCND_;
//  double COLCND_;
//  double AMAX_;

  Epetra_SerialDenseMatrix * Matrix_;
  Epetra_SerialDenseMatrix * LHS_;
  Epetra_SerialDenseMatrix * RHS_;
//  Epetra_SerialDenseMatrix * Factor_;
  Epetra_SerialDenseMatrix * Inverse_;
  
  double * A_;
//  double * FERR_;
//  double * BERR_;
//  double * AF_;
  double * AI_;
  double * WORK_;
//  double * R_;
//  double * C_;
  double * U_;
  double * S_;
  double * Vt_;

  double * B_;
  double * X_;

  bool UseTranspose_;

 private:
  // Epetra_SerialDenseSolver copy constructor (put here because we don't want user access)
  
  Epetra_SerialDenseSVD(const Epetra_SerialDenseSVD& Source);
  Epetra_SerialDenseSVD & operator=(const Epetra_SerialDenseSVD& Source);
};

#endif /* _EPETRA_SERIALDENSESVD_H_ */
