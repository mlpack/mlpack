
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

#ifndef _AZTECOO_H_
#define _AZTECOO_H_

#include "AztecOO_ConfigDefs.h"

class Epetra_Comm;
class Epetra_BlockMap;
class Epetra_MultiVector;
class Epetra_RowMatrix;
#include "Epetra_LinearProblem.h"
#include "Epetra_Object.h"
#include "Epetra_Vector.h"
#include "AztecOO_StatusTest.h"
#include "az_aztec.h"

#ifdef HAVE_AZTECOO_TEUCHOS
//forward declaration for Teuchos::ParameterList
namespace Teuchos {
  class ParameterList;
}
#endif


//! AztecOO:  An object-oriented wrapper for Aztec.
/*! Currently it accepts a Petra matrix, initial guess and RHS as
  separate arguments, or alternatively, accepts a Epetra_LinearProblem.
  If constructed using a Epetra_LinearProblem, AztecOO will infer some
  solver/preconditioner, etc., options and parameters. Users may override
  these choices and manually choose from among the full set of Aztec options
  using the SetAztecOption() and SetAztecParam() functions.

  AztecOO will solve a linear systems of equations: \f$ AX=B \f$, using Epetra
  objects and the Aztec solver library, where \f$A\f$ is an Epetra_Operator or Epetra_RowMatrix (note
  that the Epetra_Operator class is a base class for Epetra_RowMatrix so that Epetra_RowMatrix \e isa
  Epetra_Operator.) \f$X\f$ and \f$B\f$ are Epetra_MultiVector objects.

  \warning AztecOO does not presently support solution of more than one simultaneous right-hand-side.
*/

class AztecOO {

 public:

  /** \name Constructors/destructors. */ //@{

  //!  AztecOO Constructor.
  /*! Creates a AztecOO instance, passing in already-defined objects for the linear operator
    (as an Epetra_Operator),
    left-hand-side and right-hand-side.

    Note: Use of this constructor may prohibit use of native AztecOO preconditioners, since
    an Epetra_Operator is not necessarily an Epetra_RowMatrix and all AztecOO incomplete
    factorization preconditioners are based on having explicit access to matrix coefficients.
    Polynomial preconditioners are available if the Epetra_Operator passed in here has a
    non-trivial definition of the NormInf() method and HasNormInf() returns true.
  */
  AztecOO(Epetra_Operator * A, Epetra_MultiVector * X, Epetra_MultiVector * B);

  //!  AztecOO Constructor.
  /*! Creates a AztecOO instance, passing in already-defined objects for the linear operator
    (as an Epetra_RowMatrix),
    left-hand-side and right-hand-side.

    Note: Use of this constructor allows full access to native AztecOO preconditioners, using
    the Epetra_RowMatrix A passed in here as the basis for computing the preconditioner.
    All AztecOO incomplete
    factorization preconditioners are based on having explicit access to matrix coefficients.
    Polynomial preconditioners are also available.  It is possible to change the matrix used for
    computing incomplete factorization by calling the SetPrecMatrix() method.  It is
    also possible to provide a user-supplied preconditioner via SetPrecOperator().
  */
  AztecOO(Epetra_RowMatrix * A, Epetra_MultiVector * X, Epetra_MultiVector * B);

  //! AztecOO Constructor.
  /*! Creates a AztecOO instance, using a Epetra_LinearProblem,
    passing in an already-defined Epetra_LinearProblem object. The Epetra_LinearProblem class
    is the preferred method for passing in the linear problem to AztecOO because this class
    provides scaling capabilities and self-consistency checks that are not available when
    using other constructors.

    Note: If the Epetra_LinearProblem passed in here has a non-trivial pointer to an Epetra_Matrix
    then use of this constructor allows full access to native AztecOO preconditioners, using
    the Epetra_RowMatrix A passed in here as the basis for computing the preconditioner.
    All AztecOO incomplete
    factorization preconditioners are based on having explicit access to matrix coefficients.
    Polynomial preconditioners are also available.  It is possible to change the matrix used for
    computing incomplete factorization by calling the SetPrecMatrix() method.  It is
    also possible to provide a user-supplied preconditioner by call SetPrecOperator().

    If the Epetra_LinearProblems passed in here has only an Epetra_Operator, then use
    of this constructor may prohibit use of native AztecOO preconditioners, since
    an Epetra_Operator is not necessarily an Epetra_RowMatrix and all AztecOO incomplete
    factorization preconditioners are based on having explicit access to matrix coefficients.
    Polynomial preconditioners are available if the Epetra_Operator passed in here has a
    non-trivial definition of the NormInf() method and HasNormInf() returns true.
  */
  AztecOO(const Epetra_LinearProblem& LinearProblem);

  //! AztecOO Default constructor.
  AztecOO();

  //! AztecOO Copy Constructor.
  /*! Makes copy of an existing AztecOO instance.
   */
  AztecOO(const AztecOO& Solver);

  //! AztecOO Destructor.
  /*! Completely deletes a AztecOO object.
   */
  virtual ~AztecOO(void);
  //@}

  /** \name Post-construction setup methods. */ //@{

  //! AztecOO Epetra_LinearProblem Set
  /*! Associates an already defined Epetra_LinearProblem as the problem that
     will be solved during iterations.  This method allows the user to change
     which problem is being solved by an existing AztecOO object.

     Internally calls SetUserMatrix() if the Epetra_LinearProblem's operator
     can be cast to Epetra_RowMatrix, otherwise calls SetUserOperator().

    *** IMPORTANT WARNING ***
    This method calls SetUserMatrix(), which also sets the *preconditioner*
    matrix to the matrix passed in, by internally calling SetPrecMatrix(),
    but *ONLY* if SetPrecMatrix() hasn't previously been called.
    If the user wants to make sure that any pre-existing preconditioner is
    replaced, they must set the optional bool argument 'call_SetPrecMatrix'
    to true, which will force this function to call SetPrecMatrix().

    \warning If a preconditioner has been pre-built and associated with this
     AztecOO object, the Epetra_LinearProblem being passed in to this method
     \e must have compatible domain and range maps.
  */
  int SetProblem(const Epetra_LinearProblem& prob,
                 bool call_SetPrecMatrix=false);

  //! AztecOO User Operator Set
  /*! Associates an already defined Epetra_Operator as the linear operator for the linear system
    system that will be solved during
    iterations.
    This set method allows the user to pass any type of linear operator to AztecOO, as long
    as the operator implements the Epetra_Operator pure virtual class, and has proper
    domain and range map dimensions. Epetra_CrsMatrix and Epetra_VbrMatrix objects can be passed in through
    this method.
  */
  int SetUserOperator(Epetra_Operator * UserOperator);

  //! AztecOO User Matrix Set
  /*! Associates an already defined Epetra_Matrix as the matrix that will be
    used by AztecOO as the linear operator when solving the linear system.
    Epetra_CrsMatrix and Epetra_VbrMatrix objects can be passed in through
    this method.

    *** IMPORTANT WARNING ***
    This method sets the preconditioner matrix to the matrix passed in
    here, by internally calling SetPrecMatrix(), but *ONLY* if SetPrecMatrix()
    hasn't previously been called. If the user wants to make sure that any
    pre-existing preconditioner is replaced, they must set the optional bool
    argument 'call_SetPrecMatrix' to true, which will
    force this function to call SetPrecMatrix().
  */
  int SetUserMatrix(Epetra_RowMatrix * UserMatrix,
                    bool call_SetPrecMatrix=false);

  //! AztecOO LHS Set
  /*! Associates an already defined Epetra_MultiVector (or Epetra_Vector) as the initial guess
    and location where the solution will be return.
  */
  int SetLHS(Epetra_MultiVector * X);


  //! AztecOO RHS Set
  /*! Associates an already defined Epetra_MultiVector (or Epetra_Vector) as the right-hand-side of
    the linear system.
  */
  int SetRHS(Epetra_MultiVector * B);

  //! AztecOO Preconditioner Matrix Set
  /*! Associates an already defined Epetra_Matrix as the matrix that will be used by
    AztecOO when constructing native AztecOO preconditioners.  By default,
    if AztecOO native preconditioners are used, the original operator matrix will be used as
    the source for deriving the preconditioner.  However, there are instances where a user would like
    to have the preconditioner be defined using a different matrix than the original operator matrix.
    Another common situation is where the user may not have the operator in matrix form but has a matrix
    that approximates the operator and can be used as the basis for an incomplete factorization.
    This set method allows the user to pass any Epetra_RowMatrix to AztecOO for use in constructing an AztecOO
    native preconditioner, as long
    as the matrix implements the Epetra_RowMatrix pure virtual class, and has proper
    domain and range map dimensions.  Epetra_CrsMatrix and Epetra_VbrMatrix objects can be passed in through
    this method.
  */
  int SetPrecMatrix(Epetra_RowMatrix * PrecMatrix);

  //! AztecOO External Preconditioner Set
  /*! Associates an already defined Epetra_Operator as the preconditioner that will be called during
    iterations.
    This set method allows the user to pass any type of preconditioner to AztecOO, as long
    as the preconditioner implements the Epetra_Operator pure virtual class, and has proper
    domain and range map dimensions.  Ifpack preconditioners can be passed in through this method.
  */
  int SetPrecOperator(Epetra_Operator * PrecOperator);

  //! AztecOO External Convergence/Status Test Set
  /*! Assigns an already defined AztecOO_StatusTest object as the class that will determine when
    iterations should stop, either because convergence was reached or the iteration failed.
    This method allows a large variety of convergence tests to be used with AztecOO. The AztecOO_StatusTest
    class is a pure virtual class, so any class that implements its interface can be passed in
    to this set method.  A number of pre-defined AztecOO_StatusTest derived classes are already
    available, including AztecOO_StatusTestCombo, a class that allows logical combinations of 
    other status test objects for sophisticated convergence testing.
  */
  int SetStatusTest(AztecOO_StatusTest * StatusTest);

  //! Set std::ostream for Aztec's screen output.
  /*! This sets the destination for output that Aztec would normally
      send to stdout.
  */
  void SetOutputStream(std::ostream& ostrm);

  //! Set std::ostream for Aztec's error output.
  /*! This sets the destination for output that Aztec would normally
      send to stderr.
  */
  void SetErrorStream(std::ostream& errstrm);
  //@}

  //** \name Post-construction setup methods (classic approach: ONLY EXPERTS SHOULD USE THESE METHODS). */ //@{

  //! AztecOO External Preconditioner Set (object)
  /*! Associates an already defined Aztec preconditioner with this solve.
   */
  int SetPreconditioner(AZ_PRECOND * Prec) {Prec_ = Prec; return(0);};


  //! AztecOO External Preconditioner Set (function and data)
  /*! Associates an external function and data pointer with preconditioner
   */
  int SetPreconditioner(AZ_PREC_FUN  prec_function,
                        void *prec_data);

  //! AztecOO External Scaling Set
  /*! Associates an already defined Aztec scaling object with this solve.
   */
  int SetScaling(struct AZ_SCALING * Scaling) {Scaling_ = Scaling; return(0);};


  //! AztecOO Label Matrix for Aztec
  /*! This is used to label individual matrices within Aztec. This might
    be useful if several Aztec invocations are involved corresponding
    to different matrices.
  */
  int  SetMatrixName(int label);
  //@}

  /** \name Explicit preconditioner construction/assessment/destruction methods. */ //@{
  //! Forces explicit construction and retention of an AztecOO native preconditioner.
  /*! AztecOO typically constructs the preconditioner on the first call to the solve function.
    However, there are situations where we would like to compute the preconditioner ahead
    of time.  One particular case is when we want to confirm that the preconditioner
    well-conditioned.  This method allows us to precompute the preconditioner.  It also
    provides a estimate of the condition number of the preconditioner.  If \it condest is
    large, e.g., > 1.0e+14, it is likely the preconditioner will fail.  In this case, using
    threshold values (available in the incomplete factorizations) can be used to reduce
    the condition number.

    Note: This method does not work for user-defined preconditioners (defined via calls to
    SetPrecOperator().  It will return with an error code  of -1 for this case.
  */
  int ConstructPreconditioner(double & condest);

  //! Destroys a preconditioner computed using ConstructPreconditioner().
  /*! The ConstructPreconditioner() method creates a persistent preconditioner.
    In other words the preconditioner will be used by all calls to the Iterate()
    method.  DestroyPreconditioner() deletes the current preconditioner and restores
    AztecOO to a state where the preconditioner will computed on first use of the
    preconditioner solve.
  */
  int DestroyPreconditioner();

  //! Returns the condition number estimate for the current preconditioner, if one exists, returns -1.0 if no estimate.
  double Condest() const {return(condest_);};
  //@}

  /** \name Check/Attribute Access Methods. */ //@{

  //! Prints a summary of solver parameters, performs simple sanity checks.
  int CheckInput() const {
    return(AZ_check_input(Amat_->data_org, options_, params_, proc_config_));};

  //! Get a pointer to the Linear Problem used to construct this solver; returns zero if not available.
  Epetra_LinearProblem * GetProblem() const {return(Problem_);};
   //! Get a pointer to the user operator A.
  Epetra_Operator * GetUserOperator() const {
    return(UserOperatorData_ != 0 ? UserOperatorData_->A : 0);
  };
  //! Get a pointer to the user matrix A.
  Epetra_RowMatrix * GetUserMatrix() const {
    return(UserMatrixData_ != 0 ? UserMatrixData_->A : 0);
  };
  //! Get a pointer to the preconditioner operator.
  Epetra_Operator * GetPrecOperator() const {
    return(PrecOperatorData_ != 0 ? PrecOperatorData_->A : 0);
  };
  //! Get a pointer to the matrix used to construct the preconditioner.
  Epetra_RowMatrix * GetPrecMatrix() const {
    return(PrecMatrixData_ != 0 ? PrecMatrixData_->A : 0);
  };
  //! Get a pointer to the left-hand-side X.
  Epetra_MultiVector * GetLHS() const {return(X_);};
  //! Get a pointer to the right-hand-side B.
  Epetra_MultiVector * GetRHS() const {return(B_);};

  //! Print linear-system to files.
  /**
     \param name Print the matrix to the file A_'name', and print the
     solution and rhs vectors to files X_'name' and B_'name', respectively.
     Will only produce a matrix file if the run-time-type of the matrix is
     either Epetra_CrsMatrix or Epetra_VbrMatrix.
  */
  void PrintLinearSystem(const char* name);
  //@}

  /** \name Standard AztecOO option and parameter setting methods. */ //@{

#ifdef HAVE_AZTECOO_TEUCHOS
  //! Method to set options/parameters using a ParameterList object.
  /*! This method extracts any mixture of options and parameters from a
    ParameterList object and uses them to set values in AztecOO's internal
    options and params arrays.
    This method may be called repeatedly. This method does not reset default
    values or previously-set values unless those values are contained in the
    current ParameterList argument. Note that if the method SetAztecDefaults()
    is called after this method has been called, any parameters set by this
    method will be lost.

    A ParameterList is a collection of named ParameterEntry objects. AztecOO
    recognizes names which mirror the macros defined in az_aztec_defs.h. In
    addition, it recognizes case insensitive versions of those names, with or
    without the prepended 'AZ_'. So the following are equivalent and valid:
    "AZ_solver", "SOLVER", "Solver". To set an entry in the Aztec options
    array, the type of the ParameterEntry value may be either a string or an int
    in some cases. E.g., if selecting the solver, the following are equivalent
    and valid: AZ_gmres (which is an int), "AZ_gmres" (which is a string) or
    "GMRES" (case-insensitive, 'AZ_' is optional).

    To set an entry in the Aztec params array, the type of the
    ParameterEntry value must be double.

    By default, this method will silently ignore parameters which have
    unrecognized names or invalid types. Users may set the optional argument
    specifying that warnings be printed for unused parameters. Alternatively,
    users may iterate the ParameterList afterwards and check the isUsed
    attribute on the ParameterEntry objects.

    @param parameterlist Object containing parameters to be parsed by AztecOO.
    @param cerr_warning_if_unused Optional argument, default value is false.
        If true, a warning is printed to cerr stating which parameters are
        not used due to having unrecognized names or values of the wrong type.
        Default behavior is to silently ignore unused parameters.
  */
  int SetParameters(Teuchos::ParameterList& parameterlist,
                    bool cerr_warning_if_unused=false);
#endif

  //! AztecOO function to restore default options/parameter settings.
  /*! This function is called automatically within AztecOO's constructor,
    but if constructed using a Epetra_LinearProblem object, some options are
    reset based on the ProblemDifficultyLevel associated with the
    Epetra_LinearProblem.

    See the Aztec 2.1 User Guide for a complete list of these options.

    \warning In AztecOO, the default value of options[AZ_poly_ord] is set to 1.
    This is different than Aztec 2.1, but the preferred value since Jacobi preconditioning
    is used much more often than polynomial preconditioning and one step of Jacobi is
    far more effective than 3 steps.

  */
  int SetAztecDefaults();

  //! AztecOO option setting function.
  /*! Set a specific Aztec option value.
    Example: problem.SetAztecOption(AZ_precond, AZ_Jacobi)

    See the Aztec 2.1 User Guide for a complete list of these options.
  */
  int SetAztecOption(int option, int value)
    {options_[option] = value; return(0);};

  //! AztecOO option getting function.
  /*! Get a specific Aztec optioin value.
    Example: problem.GetAztecOption(AZ_precond)

    See the Aztec 2.1 User Guide for a complete list of these options.
  */
  int GetAztecOption(int option)
    {return(options_[option]);}

  //! AztecOO param setting function.
  /*! Set a specific Aztec parameter value.
    Example: problem.SetAztecParam(AZ_drop, 1.0E-6)

    See the Aztec 2.1 User Guide for a complete list of these parameters.
  */
  int SetAztecParam(int param, double value)
    {params_[param] = value; return(0);};

  //! AztecOO option setting function.
  /*! Return a pointer to an array (size AZ_OPTIONS_SIZE) of all of the currently set aztec options. 
   */
  const int* GetAllAztecOptions() const
    { return options_; };

  //! AztecOO param setting function.
  /*! Return a pointer to an array (size AZ_PARAMS_SIZE) of all of the currently set aztec parameters. 
   */
  const double* GetAllAztecParams() const
    { return params_; };

  //! AztecOO option setting function.
  /*! Set all Aztec option values using an existing Aztec options array.
   */
  int SetAllAztecOptions(const int * options)
    {for (int i=0; i<AZ_OPTIONS_SIZE; i++) options_[i] = options[i]; return(0);};

  //! AztecOO param setting function.
  /*! Set all Aztec parameter values using an existing Aztec params array.
   */
  int SetAllAztecParams(const double * params)
    {for (int i=0; i<AZ_PARAMS_SIZE; i++) params_[i] = params[i]; return(0);};
  //@}

  /** \name Standard AztecOO solve methods. */ //@{
  //! AztecOO iteration function.
  /*! Iterates on the current problem until MaxIters or Tolerance is reached.
   */
  int Iterate(int MaxIters, double Tolerance);

  //! AztecOO iteration function.
  /*! Iterates on the specified matrix and vectors until MaxIters or Tolerance
    is reached..
  */
  int Iterate(Epetra_RowMatrix * A,
              Epetra_MultiVector * X,
              Epetra_MultiVector * B,
              int MaxIters, double Tolerance);


  //@}

  /** \name Specialist AztecOO solve method. */ //@{
  //! AztecOO iteration functions.
  /*! Iterates on the current problem until MaxIters or Tolerance is reached..
    This one should be suitable for recursive invocations of Aztec.
  */
  int recursiveIterate(int MaxIters, double Tolerance);

  //! Return the Aztec status after iterating.
  /*! Returns pointer to the underlying Aztec Status array
	  (of length AZ_STATUS_SIZE).  See the Aztec documenation.
   */
  const double *GetAztecStatus() const { return status_; };

  //@}

  /** \name Adaptive Solve methods. */ //@{

  //! Force the AdaptiveIterate() method to use default adaptive strategy.
  int SetUseAdaptiveDefaultsTrue(){useAdaptiveDefaults_ = true;return(0);};

  //! Set the parameter that control the AdaptiveIterate() method.
  /*! The AdaptiveIterate() method attempts to solve a given problem using multiple preconditioner
    and iterative method tuning parameters.  There are defaults that are coded into AdaptiveIterate()
    method, but the defaults can be over-ridden by the use of the SetAdaptiveParams() method. Details of
    condition number management follow:
    \verbinclude Managing_conditioning_howto.txt

    \param NumTrials In
    The number of Athresh and Rthresh pairs that should be tried when attempting to stabilize
    the preconditioner.
    \param athresholds In
    The list of absolute threshold values that should be tried when attempting to stabilize
    the preconditioner.
    \param rthresholds In
    The list of relative threshold values that should be tried when attempting to stabilize
    the preconditioner.
    \param condestThreshold In
    If the condition number estimate of the preconditioner is above this number, no attempt will be
    made to try iterations.  Instead a new preconditioner will be computed using the next threshold
    pair.
    \param maxFill In
    In addition to managing the condest, the AdaptiveIterate() method will also try to increase
    the preconditioner fill if it is determined that this might help.  maxFill specifies the
    maximum fill allowed.
    \param maxKspace In
    In addition to managing the condest, the AdaptiveIterate() method will also try to increase
    the Krylov subspace size if GMRES is being used and it is determined that this might help.
    maxKspace specifies the maximum Krylov subspace allowed.

  */
  int SetAdaptiveParams(int NumTrials, double * athresholds, double * rthresholds,
                        double condestThreshold, double maxFill, int maxKspace);

  //! Attempts to solve the given linear problem using an adaptive strategy.
  int AdaptiveIterate(int MaxIters, int MaxSolveAttempts, double Tolerance);
  //@}

  /** \name Post-solve access functions */ //@{

  //! Returns the total number of iterations performed on this problem.
  int NumIters() const {return((int) status_[AZ_its]);};

  //! Returns the true unscaled residual for this problem.
  double TrueResidual() const {return(status_[AZ_r]);};

  //! Returns the true scaled residual for this problem.
  double ScaledResidual() const {return(status_[AZ_scaled_r]);};

  //! Returns the recursive residual for this problem.
  double RecursiveResidual() const {return(status_[AZ_rec_r]);};

  //! Returns the solve time.
  double SolveTime() const {return(status_[AZ_solve_time]);}

  //! AztecOO status extraction function.
  /*! Extract Aztec status array into user-provided array.  The array must be of
    length AZ_STATUS_SIZE as defined in the az_aztec.h header file.
  */
  int GetAllAztecStatus(double * status)
    {for (int i=0; i<AZ_STATUS_SIZE; i++) status[i] = status_[i]; return(0);};
  //@}


  struct MatrixData {
    Epetra_RowMatrix * A;
    Epetra_Vector * X;
    Epetra_Vector * Y;
    Epetra_Vector * SourceVec;
    Epetra_Vector * TargetVec;

    MatrixData(Epetra_RowMatrix * inA = 0, Epetra_Vector * inX = 0,
               Epetra_Vector * inY = 0, Epetra_Vector * inSourceVec = 0,
               Epetra_Vector * inTargetVec = 0)
      : A(inA), X(inX), Y(inY), SourceVec(inSourceVec), TargetVec(inTargetVec){}

      ~MatrixData();
  };

  struct OperatorData {
    Epetra_Operator * A;
    Epetra_Vector * X;
    Epetra_Vector * Y;
    OperatorData(Epetra_Operator * inA = 0, Epetra_Vector * inX = 0,
                 Epetra_Vector * inY = 0)
      : A(inA), X(inX), Y(inY) {}
      ~OperatorData();
  };

 protected:

  int AllocAzArrays();
  void DeleteAzArrays();
  int SetAztecVariables();
  int SetProblemOptions(ProblemDifficultyLevel PDL,
                        bool ProblemSymmetric);
  int SetProcConfig(const Epetra_Comm & Comm);

  void DeleteMemory();


  Epetra_LinearProblem * Problem_;
  Epetra_MultiVector * X_;
  Epetra_MultiVector * B_;
  Epetra_Vector * ResidualVector_;

  int N_local_;
  int x_LDA_;
  double *x_;
  int b_LDA_;
  double *b_;
  int * proc_config_;
  int * options_;
  double * params_;
  double * status_;
  AZ_MATRIX *Amat_;
  AZ_MATRIX *Pmat_;
  AZ_PRECOND *Prec_;
  struct AZ_SCALING * Scaling_;
  bool Scaling_created_;
  AztecOO_StatusTest * StatusTest_;
  struct AZ_CONVERGE_STRUCT *conv_info_;

  double condest_;
  bool useAdaptiveDefaults_;
  int NumTrials_;
  double maxFill_;
  int maxKspace_;
  double * athresholds_;
  double * rthresholds_;
  double condestThreshold_;
  bool inConstructor_; // Shuts down zero pointer error reporting while in a constructor
  bool procConfigSet_;

  MatrixData * UserMatrixData_;
  MatrixData * PrecMatrixData_;
  OperatorData * UserOperatorData_;
  OperatorData * PrecOperatorData_;

  std::ostream* out_stream_;
  std::ostream* err_stream_;
};

// External prototypes
extern "C" void Epetra_Aztec_matvec(double x[], double y[], AZ_MATRIX *Amat, int proc_config[]);
extern "C" double Epetra_Aztec_matnorminf(AZ_MATRIX *Amat);
extern "C" void Epetra_Aztec_operatorvec(double x[], double y[], AZ_MATRIX *Amat, int proc_config[]);
extern "C" double Epetra_Aztec_operatornorminf(AZ_MATRIX *Amat);
extern "C" void Epetra_Aztec_precond(double x[], int input_options[],
                                     int proc_config[], double input_params[], AZ_MATRIX *Amat,
                                     AZ_PRECOND *prec);
extern "C" int Epetra_Aztec_getrow(int columns[], double values[], int row_lengths[],
                                   AZ_MATRIX *Amat, int N_requested_rows,
                                   int requested_rows[], int allocated_space);
extern "C" int Epetra_Aztec_comm_wrapper(double vec[], AZ_MATRIX *Amat);

void AztecOO_StatusTest_wrapper(void * conv_test_obj,void * res_vector_obj,
			   int iteration, double * res_vector, int print_info, 
			   int sol_updated, int * converged, int * isnan, 
			   double * rnorm, int * r_avail); 
#endif /* _AZTECOO_H_ */

