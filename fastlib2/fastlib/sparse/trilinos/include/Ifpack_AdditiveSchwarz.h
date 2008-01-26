#ifndef IFPACK_ADDITIVESCHWARZ_H
#define IFPACK_ADDITIVESCHWARZ_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_Reordering.h"
#include "Ifpack_RCMReordering.h"
#include "Ifpack_METISReordering.h"
#include "Ifpack_LocalFilter.h"
#include "Ifpack_SingletonFilter.h"
#include "Ifpack_ReorderFilter.h"
#include "Ifpack_Utils.h"
#include "Ifpack_OverlappingRowMatrix.h"
#include "Epetra_CombineMode.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_Time.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RefCountPtr.hpp"

//! Ifpack_AdditiveSchwarz: a class to define Additive Schwarz preconditioners of Epetra_RowMatrix's.

/*!
  Class  Ifpack_AdditiveSchwarz enables the construction of Additive
  Schwarz (one-level overlapping domain decomposition) preconditioners, 
  for a given Epetra_RowMatrix.
  Ifpack_AdditiveSchwarz is derived from Ifpack_Preconditioner,
  itself derived from Epetra_Operator. An application of
  the Additive Schwarz preconditioner can be obtained 
  by calling method ApplyInverse().

  One-level overlapping domain decomposition preconditioners use 
  local solvers, of Dirichlet type. This means that the inverse of
  the local matrix (with minimal or wider overlap) is applied to
  the residual to be preconditioned.

  The preconditioner can be written as:
  \f[
  P_{AS}^{-1} = \sum_{i=1}^M P_i A_i^{-1} R_i ,
  \f]
  where \f$M\f$ is the number of subdomains (that is, the number of 
  processors in
  the computation), \f$R_i\f$ is an operator that restricts the global
  vector to the vector lying on subdomain \f$i\f$, \f$P_i\f$ is the 
  prolongator operator, and
  \f[
  A_i = R_i A P_i.
  \f]

  The construction of Schwarz preconditioners is mainly composed by
  two steps:
  - definition of the restriction and prolongation operator
  \f$R_i\f$ and \f$R_i^T\f$. If minimal overlap is chosen, their
  implementation is trivial, \f$R_i\f$ will return all the local
  components. For wider overlaps, instead, Epetra_Import and
  Epetra_Export will be used to import/export data. The user
  must provide both the matrix to be preconditioned (which is suppose
  to have minimal-overlap) and the matrix with wider overlap.
  - definition of a technique to apply the inverse of \f$A_i\f$.
  To solve on each subdomain, the user can adopt any class, derived
  from Ifpack_Preconditioner. This can be easily accomplished, as
  Ifpack_AdditiveSchwarz is templated with the solver for each subdomain.

  The local matrix \f$A_i\f$ can be filtered, to eliminate singletons, and
  reordered. At the present time, RCM and METIS can be used to reorder the
  local matrix.
  
  The complete list of supported parameters is reported in page \ref ifp_params.

  \author Marzio Sala, SNL 9214.

  \date Last modified on 22-Jan-05.
*/

template<typename T>
class Ifpack_AdditiveSchwarz : public virtual Ifpack_Preconditioner {
      
public:

  //@{ \name Constructors/Destructors
  //! Ifpack_AdditiveSchwarz constructor with given Epetra_RowMatrix.
  /*! Creates an Ifpack_AdditiveSchwarz preconditioner with overlap.
   * To use minimal-overlap, OverlappingMatrix is omitted
   * (as defaulted to 0).
   *
   * \param
   * Matrix - (In) Pointer to matrix to be preconditioned
   *
   * \param
   * OverlappingMatrix - (In) Pointer to the matrix extended with the
   *                     desired level of overlap.
   */
  Ifpack_AdditiveSchwarz(Epetra_RowMatrix* Matrix,
			 int OverlapLevel = 0);
  
  //! Destructor
  virtual ~Ifpack_AdditiveSchwarz() {};
  //@}

  //@{ \name Atribute set methods.

    //! If set true, transpose of this operator will be applied (not implemented).
    /*! This flag allows the transpose of the given operator to be used 
     * implicitly.  
      
    \param 
	   UseTranspose - (In) If true, multiply by the transpose of operator, 
	   otherwise just use operator.

    \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does not support transpose.
  */
    virtual int SetUseTranspose(bool UseTranspose);
  //@}
  
  //@{ \name Mathematical functions.

  //! Applies the matrix to X, returns the result in Y.
  /*! 
    \param
    X - (In) A Epetra_MultiVector of dimension NumVectors 
       to multiply with matrix.
    \param
    Y -(Out) A Epetra_MultiVector of dimension NumVectors 
       containing the result.

    \return Integer error code, set to 0 if successful.
    */
    virtual int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Applies the preconditioner to X, returns the result in Y.
  /*! 
    \param
    X - (In) A Epetra_MultiVector of dimension NumVectors to be preconditioned.
    \param
    Y -(Out) A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.

    \warning In order to work with AztecOO, any implementation of this method 
    must support the case where X and Y are the same object.
    */
    virtual int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns the infinity norm of the global matrix (not implemented)
    virtual double NormInf() const;
  //@}
  
  //@{ \name Atribute access functions

    //! Returns a character string describing the operator
    virtual const char * Label() const;

    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const;

    //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
    virtual bool HasNormInf() const;

    //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
    virtual const Epetra_Comm & Comm() const;

    //! Returns the Epetra_Map object associated with the domain of this operator.
    virtual const Epetra_Map & OperatorDomainMap() const;

    //! Returns the Epetra_Map object associated with the range of this operator.
    virtual const Epetra_Map & OperatorRangeMap() const;
  //@}

  //! Returns \c true if the preconditioner has been successfully initialized.
  virtual bool IsInitialized() const
  {
    return(IsInitialized_);
  }

  //! Returns \c true if the preconditioner has been successfully computed.
  virtual bool IsComputed() const
  {
    return(IsComputed_);
  }

  //! Sets the parameters.
  /*! Sets the parameter for the additive Schwarz preconditioner,
   * as well as for all the preconditioners that may need to
   * be defined on each subblock.
   * Parameters accepted by List are:
   * - \c "schwarz: combine mode" : It must be an Epetra_CombineMode.
   *     Default: \c Zero.
   *     It Can be assume of the following values:
   *   - Add: Components on the receiving processor will be added together;
   *   - Zero: Off-processor components will be ignored;
   *   - Insert: Off-processor components will be inserted into locations on 
   *     receiving processor replacing existing values.
   *   - Average: Off-processor components will be averaged with existing;
   *   - AbsMax: Magnitudes of Off-processor components will be 
   *     maxed with magnitudes of existing components on the receiving 
   *     processor.
   * - \c "schwarz: compute condest" : if \c true, \c Compute() will
   *    estimate the condition number of the preconditioner. 
   *    Default: \c true.
   */
  virtual int SetParameters(Teuchos::ParameterList& List);

  // @}

  // @{ Query methods
  
  //! Initialized the preconditioner.
  virtual int Initialize();

  //! Computes the preconditioner.
  virtual int Compute();

  //! Computes the estimated condition number and returns its value.
  virtual double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                         const int MaxIters = 1550,
                         const double Tol = 1e-9,
			 Epetra_RowMatrix* Matrix = 0);

  //! Returns the estimated condition number, or -1.0 if not computed.
  virtual double Condest() const
  {
    return(Condest_);
  }

  //! Returns a refernence to the internally stored matrix.
  virtual const Epetra_RowMatrix& Matrix() const
  {
    return(*Matrix_);
  }

  //! Returns \c true is an overlapping matrix is present.
  virtual bool IsOverlapping() const
  {
    return(IsOverlapping_);
  }

  //! Prints major information about this preconditioner.
  virtual std::ostream& Print(std::ostream&) const;
  
  virtual const T* Inverse() const
  {
    return(&*Inverse_);
  }

  //! Returns the number of calls to Initialize().
  virtual int NumInitialize() const
  {
    return(NumInitialize_);
  }

  //! Returns the number of calls to Compute().
  virtual int NumCompute() const
  {
    return(NumCompute_);
  }

  //! Returns the number of calls to ApplyInverse().
  virtual int NumApplyInverse() const
  {
    return(NumApplyInverse_);
  }

  //! Returns the time spent in Initialize().
  virtual double InitializeTime() const
  {
    return(InitializeTime_);
  }

  //! Returns the time spent in Compute().
  virtual double ComputeTime() const
  {
    return(ComputeTime_);
  }

  //! Returns the time spent in ApplyInverse().
  virtual double ApplyInverseTime() const
  {
    return(ApplyInverseTime_);
  }

  //! Returns the number of flops in the initialization phase.
  virtual double InitializeFlops() const
  {
    return(InitializeFlops_);
  }

  virtual double ComputeFlops() const
  {
    return(ComputeFlops_);
  }

  virtual double ApplyInverseFlops() const
  {
    return(ApplyInverseFlops_);
  }

  //! Returns the level of overlap.
  virtual int OverlapLevel() const
  {
    return(OverlapLevel_);
  }

  //! Returns a reference to the internally stored list.
  virtual const Teuchos::ParameterList& List() const
  {
    return(List_);
  }

protected:

  // @}

  // @{ Internal merhods.
  
  //! Copy constructor (should never be used)
  Ifpack_AdditiveSchwarz(const Ifpack_AdditiveSchwarz& RHS)
  { }

  //! Sets up the localized matrix and the singleton filter.
  int Setup();
  
  // @}

  // @{ Internal data.
  
  //! Pointers to the matrix to be preconditioned.
  Teuchos::RefCountPtr<const Epetra_RowMatrix> Matrix_;
  //! Pointers to the overlapping matrix.
  Teuchos::RefCountPtr<Ifpack_OverlappingRowMatrix> OverlappingMatrix_;
  //! Localized version of Matrix_ or OverlappingMatrix_.
  Teuchos::RefCountPtr<Ifpack_LocalFilter> LocalizedMatrix_;
  //! Contains the label of \c this object.
  string Label_;
  //! If true, the preconditioner has been successfully initialized.
  bool IsInitialized_;
  //! If true, the preconditioner has been successfully computed.
  bool IsComputed_;
  //! Pointer to the local solver.
  Teuchos::RefCountPtr<T> Inverse_;
  //! If \c true, solve with the transpose (not supported by all solvers).
  bool UseTranspose_;
  //! If true, overlapping is used
  bool IsOverlapping_;
  //! Level of overlap among the processors.
  int OverlapLevel_;
  //! Stores a copy of the list given in SetParameters()
  Teuchos::ParameterList List_;
  //! Combine mode for off-process elements (only if overlap is used)
  Epetra_CombineMode CombineMode_;
  //! Contains the estimated condition number.
  double Condest_;
  //! If \c true, compute the condition number estimate each time Compute() is called.
  bool ComputeCondest_;
  //! If \c true, reorder the local matrix.
  bool UseReordering_;
  //! Type of reordering of the local matrix.
  string ReorderingType_;
  //! Pointer to a reordering object.
  Teuchos::RefCountPtr<Ifpack_Reordering> Reordering_;
  //! Pointer to the reorderd matrix.
  Teuchos::RefCountPtr<Ifpack_ReorderFilter> ReorderedLocalizedMatrix_;
  //! Filter for singletons.
  bool FilterSingletons_;
  //! filtering object.
  Teuchos::RefCountPtr<Ifpack_SingletonFilter> SingletonFilter_;
  //! Contains the number of successful calls to Initialize().
  int NumInitialize_;
  //! Contains the number of successful call to Compute().
  int NumCompute_;
  //! Contains the number of successful call to ApplyInverse().
  mutable int NumApplyInverse_;
  //! Contains the time for all successful calls to Initialize().
  double InitializeTime_;
  //! Contains the time for all successful calls to Compute().
  double ComputeTime_;
  //! Contains the time for all successful calls to ApplyInverse().
  mutable double ApplyInverseTime_;
  //! Contains the number of flops for Initialize().
  double InitializeFlops_;
  //! Contains the number of flops for Compute().
  double ComputeFlops_;
  //! Contain sthe number of flops for ApplyInverse().
  mutable double ApplyInverseFlops_;
  //! Object used for timing purposes.
  Teuchos::RefCountPtr<Epetra_Time> Time_;

}; // class Ifpack_AdditiveSchwarz<T>

//==============================================================================
template<typename T>
Ifpack_AdditiveSchwarz<T>::
Ifpack_AdditiveSchwarz(Epetra_RowMatrix* Matrix,
		       int OverlapLevel) :
  IsInitialized_(false),
  IsComputed_(false),
  UseTranspose_(false),
  IsOverlapping_(false),
  OverlapLevel_(OverlapLevel),
  CombineMode_(Zero),
  Condest_(-1.0),
  ComputeCondest_(true),
  UseReordering_(false),
  ReorderingType_("none"),
  FilterSingletons_(false),
  NumInitialize_(0),
  NumCompute_(0),
  NumApplyInverse_(0),
  InitializeTime_(0.0),
  ComputeTime_(0.0),
  ApplyInverseTime_(0.0),
  InitializeFlops_(0.0),
  ComputeFlops_(0.0),
  ApplyInverseFlops_(0.0)
{
  // Construct a reference-counted pointer with the input matrix, don't manage the memory.
  Matrix_ = Teuchos::rcp( Matrix, false );

  if (Matrix_->Comm().NumProc() == 1)
    OverlapLevel_ = 0;

  if ((OverlapLevel_ != 0) && (Matrix_->Comm().NumProc() > 1))
    IsOverlapping_ = true;
  // Sets parameters to default values
  Teuchos::ParameterList List;
  SetParameters(List);
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::Setup()
{

  Epetra_RowMatrix* MatrixPtr;

  if (OverlappingMatrix_ != Teuchos::null)
    LocalizedMatrix_ = Teuchos::rcp( new Ifpack_LocalFilter(OverlappingMatrix_) );
  else
    LocalizedMatrix_ = Teuchos::rcp( new Ifpack_LocalFilter(Matrix_) );

  if (LocalizedMatrix_ == Teuchos::null)
    IFPACK_CHK_ERR(-5);

  // users may want to skip singleton check
  if (FilterSingletons_) {
    SingletonFilter_ = Teuchos::rcp( new Ifpack_SingletonFilter(LocalizedMatrix_) );
    MatrixPtr = &*SingletonFilter_;
  }
  else
    MatrixPtr = &*LocalizedMatrix_;

  if (UseReordering_) {

    // create reordering and compute it
    if (ReorderingType_ == "rcm")
      Reordering_ = Teuchos::rcp( new Ifpack_RCMReordering() );
    else if (ReorderingType_ == "metis")
      Reordering_ = Teuchos::rcp( new Ifpack_METISReordering() );
    else {
      cerr << "reordering type not correct (" << ReorderingType_ << ")" << endl;
      exit(EXIT_FAILURE);
    }
    if (Reordering_ == Teuchos::null) IFPACK_CHK_ERR(-5);

    IFPACK_CHK_ERR(Reordering_->SetParameters(List_));
    IFPACK_CHK_ERR(Reordering_->Compute(*MatrixPtr));

    // now create reordered localized matrix
    ReorderedLocalizedMatrix_ = 
      Teuchos::rcp( new Ifpack_ReorderFilter(Teuchos::rcp( MatrixPtr, false ), Reordering_) );

    if (ReorderedLocalizedMatrix_ == Teuchos::null) IFPACK_CHK_ERR(-5);

    MatrixPtr = &*ReorderedLocalizedMatrix_;
  }

  Inverse_ = Teuchos::rcp( new T(MatrixPtr) );

  if (Inverse_ == Teuchos::null)
    IFPACK_CHK_ERR(-5);

  return(0);
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::SetParameters(Teuchos::ParameterList& List)
{
 
  // compute the condition number each time Compute() is invoked.
  ComputeCondest_ = List.get("schwarz: compute condest", ComputeCondest_);
  // combine mode
  if( Teuchos::ParameterEntry *combineModeEntry = List.getEntryPtr("schwarz: combine mode") )
  {
    if( typeid(std::string) == combineModeEntry->getAny().type() )
    {
      std::string mode = List.get("schwarz: combine mode", "Add");
      if (mode == "Add")
        CombineMode_ = Add;
      else if (mode == "Zero")
        CombineMode_ = Zero;
      else if (mode == "Insert")
        CombineMode_ = Insert;
      else if (mode == "InsertAdd")
        CombineMode_ = InsertAdd;
      else if (mode == "Average")
        CombineMode_ = Average;
      else if (mode == "AbsMax")
        CombineMode_ = AbsMax;
      else
      {
        TEST_FOR_EXCEPTION(
          true,std::logic_error
          ,"Error, The (Epetra) combine mode of \""<<mode<<"\" is not valid!  Only the values"
          " \"Add\", \"Zero\", \"Insert\", \"InsertAdd\", \"Average\", and \"AbsMax\" are accepted!"
          );
      }
    }
    else if ( typeid(Epetra_CombineMode) == combineModeEntry->getAny().type() )
    {
      CombineMode_ = Teuchos::any_cast<Epetra_CombineMode>(combineModeEntry->getAny());
    }
    else
    {
      // Throw exception with good error message!
      Teuchos::getParameter<std::string>(List,"schwarz: combine mode");
    }
  }
  else
  {
    // Make the default be a string to be consistent with the valid parameters!
    List.get("schwarz: combine mode","Zero");
  }
  // type of reordering
  ReorderingType_ = List.get("schwarz: reordering type", ReorderingType_);
  if (ReorderingType_ == "none")
    UseReordering_ = false;
  else 
    UseReordering_ = true;
  // if true, filter singletons. NOTE: the filtered matrix can still have
  // singletons! A simple example: upper triangular matrix, if I remove
  // the lower node, I still get a matrix with a singleton! However, filter
  // singletons should help for PDE problems with Dirichlet BCs.
  FilterSingletons_ = List.get("schwarz: filter singletons", FilterSingletons_);

  // This copy may be needed by Amesos or other preconditioners.
  List_ = List;

  return(0);
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::Initialize()
{
  IsInitialized_ = false;
  IsComputed_ = false; // values required
  Condest_ = -1.0; // zero-out condest

  if (Time_ == Teuchos::null)
    Time_ = Teuchos::rcp( new Epetra_Time(Comm()) );

  Time_->ResetStartTime();

  // compute the overlapping matrix if necessary
  if (IsOverlapping_) {
    OverlappingMatrix_ = 
      Teuchos::rcp( new Ifpack_OverlappingRowMatrix(Matrix_, OverlapLevel_) );
    if (OverlappingMatrix_ == Teuchos::null)
      IFPACK_CHK_ERR(-5);
  }

  IFPACK_CHK_ERR(Setup());

  if (Inverse_ == Teuchos::null)
    IFPACK_CHK_ERR(-5);

  if (LocalizedMatrix_ == Teuchos::null)
    IFPACK_CHK_ERR(-5);

  IFPACK_CHK_ERR(Inverse_->SetUseTranspose(UseTranspose()));
  IFPACK_CHK_ERR(Inverse_->SetParameters(List_));
  IFPACK_CHK_ERR(Inverse_->Initialize());

  // Label is for Aztec-like solvers
  Label_ = "Ifpack_AdditiveSchwarz, ";
  if (UseTranspose())
    Label_ += ", transp";
  Label_ += ", ov = " + Ifpack_toString(OverlapLevel_)
    + ", local solver = \n\t\t***** `" + string(Inverse_->Label()) + "'";

  IsInitialized_ = true;
  ++NumInitialize_;
  InitializeTime_ += Time_->ElapsedTime();

  // count flops by summing up all the InitializeFlops() in each
  // Inverse. Each Inverse() can only give its flops -- it acts on one
  // process only
  double partial = Inverse_->InitializeFlops();
  double total;
  Comm().SumAll(&partial, &total, 1);
  InitializeFlops_ += total;

  return(0);
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::Compute()
{

  if (IsInitialized() == false)
    IFPACK_CHK_ERR(Initialize());

  Time_->ResetStartTime();
  IsComputed_ = false;
  Condest_ = -1.0;
  
  IFPACK_CHK_ERR(Inverse_->Compute());

  IsComputed_ = true; // need this here for Condest(Ifpack_Cheap)
  ++NumCompute_;
  ComputeTime_ += Time_->ElapsedTime();

  // sum up flops
  double partial = Inverse_->ComputeFlops();
   double total;
  Comm().SumAll(&partial, &total, 1);
  ComputeFlops_ += total;

  // reset the Label
  string R = "";
  if (UseReordering_)
    R = ReorderingType_ + " reord, ";

  if (ComputeCondest_)
    Condest(Ifpack_Cheap);
  
  // add Condest() to label
  Label_ = "Ifpack_AdditiveSchwarz, ov = " + Ifpack_toString(OverlapLevel_)
    + ", local solver = \n\t\t***** `" + string(Inverse_->Label()) + "'"
    + "\n\t\t***** " + R + "Condition number estimate = "
    + Ifpack_toString(Condest());

  return(0);
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::SetUseTranspose(bool UseTranspose)
{
  // store the flag -- it will be set in Initialize() if Inverse_ does not
  // exist.
  UseTranspose_ = UseTranspose;

  // If Inverse_ exists, pass it right now.
  if (Inverse_!=Teuchos::null)
    IFPACK_CHK_ERR(Inverse_->SetUseTranspose(UseTranspose));
  return(0);
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  IFPACK_CHK_ERR(Matrix_->Apply(X,Y));
  return(0);
}

//==============================================================================
template<typename T>
double Ifpack_AdditiveSchwarz<T>::NormInf() const
{
  return(-1.0);
}

//==============================================================================
template<typename T>
const char * Ifpack_AdditiveSchwarz<T>::Label() const
{
  return(Label_.c_str());
}

//==============================================================================
template<typename T>
bool Ifpack_AdditiveSchwarz<T>::UseTranspose() const
{
  return(UseTranspose_);
}

//==============================================================================
template<typename T>
bool Ifpack_AdditiveSchwarz<T>::HasNormInf() const
{
  return(false);
}

//==============================================================================
template<typename T>
const Epetra_Comm & Ifpack_AdditiveSchwarz<T>::Comm() const
{
  return(Matrix_->Comm());
}

//==============================================================================
template<typename T>
const Epetra_Map & Ifpack_AdditiveSchwarz<T>::OperatorDomainMap() const
{
  return(Matrix_->OperatorDomainMap());
}

//==============================================================================
template<typename T>
const Epetra_Map & Ifpack_AdditiveSchwarz<T>::OperatorRangeMap() const
{
  return(Matrix_->OperatorRangeMap());
}

//==============================================================================
template<typename T>
int Ifpack_AdditiveSchwarz<T>::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // compute the preconditioner is not done by the user
  if (!IsComputed())
    IFPACK_CHK_ERR(-3);
  
  int NumVectors = X.NumVectors();

  if (NumVectors != Y.NumVectors())
    IFPACK_CHK_ERR(-2); // wrong input

  Time_->ResetStartTime();

  Teuchos::RefCountPtr<Epetra_MultiVector> OverlappingX;
  Teuchos::RefCountPtr<Epetra_MultiVector> OverlappingY;
  Teuchos::RefCountPtr<Epetra_MultiVector> Xtmp;

  // for flop count, see bottom of this function
  double pre_partial = Inverse_->ApplyInverseFlops();
  double pre_total;
  Comm().SumAll(&pre_partial, &pre_total, 1);

  // process overlap, may need to create vectors and import data
  if (IsOverlapping()) {
    OverlappingX = Teuchos::rcp( new Epetra_MultiVector(OverlappingMatrix_->RowMatrixRowMap(),
							X.NumVectors()) );
    OverlappingY = Teuchos::rcp( new Epetra_MultiVector(OverlappingMatrix_->RowMatrixRowMap(),
							Y.NumVectors()) );
    if (OverlappingY == Teuchos::null) IFPACK_CHK_ERR(-5);

    OverlappingY->PutScalar(0.0);
    OverlappingX->PutScalar(0.0);
    IFPACK_CHK_ERR(OverlappingMatrix_->ImportMultiVector(X,*OverlappingX,Insert));
    // FIXME: this will not work with overlapping and non-zero starting
    // solutions. The same for other cases below.
    // IFPACK_CHK_ERR(OverlappingMatrix_->ImportMultiVector(Y,*OverlappingY,Insert));
  }
  else {
    Xtmp = Teuchos::rcp( new Epetra_MultiVector(X) );
    OverlappingX = Xtmp;
    OverlappingY = Teuchos::rcp( &Y, false );
  }

  if (FilterSingletons_) {
    // process singleton filter
    Epetra_MultiVector ReducedX(SingletonFilter_->Map(),NumVectors);
    Epetra_MultiVector ReducedY(SingletonFilter_->Map(),NumVectors);
    IFPACK_CHK_ERR(SingletonFilter_->SolveSingletons(*OverlappingX,*OverlappingY));
    IFPACK_CHK_ERR(SingletonFilter_->CreateReducedRHS(*OverlappingY,*OverlappingX,ReducedX));

    // process reordering
    if (!UseReordering_) {
      IFPACK_CHK_ERR(Inverse_->ApplyInverse(ReducedX,ReducedY));
    }
    else {
      Epetra_MultiVector ReorderedX(ReducedX);
      Epetra_MultiVector ReorderedY(ReducedY);
      IFPACK_CHK_ERR(Reordering_->P(ReducedX,ReorderedX));
      IFPACK_CHK_ERR(Inverse_->ApplyInverse(ReorderedX,ReorderedY));
      IFPACK_CHK_ERR(Reordering_->Pinv(ReorderedY,ReducedY));
    }

    // finish up with singletons
    IFPACK_CHK_ERR(SingletonFilter_->UpdateLHS(ReducedY,*OverlappingY));
  }
  else {
    // process reordering
    if (!UseReordering_) {
      IFPACK_CHK_ERR(Inverse_->ApplyInverse(*OverlappingX,*OverlappingY));
    }
    else {
      Epetra_MultiVector ReorderedX(*OverlappingX);
      Epetra_MultiVector ReorderedY(*OverlappingY);
      IFPACK_CHK_ERR(Reordering_->P(*OverlappingX,ReorderedX));
      IFPACK_CHK_ERR(Inverse_->ApplyInverse(ReorderedX,ReorderedY));
      IFPACK_CHK_ERR(Reordering_->Pinv(ReorderedY,*OverlappingY));
    }
  }

  if (IsOverlapping()) {
    IFPACK_CHK_ERR(OverlappingMatrix_->ExportMultiVector(*OverlappingY,Y,
							 CombineMode_));
  }

  // add flops. Note the we only have to add the newly counted
  // flops -- and each Inverse returns the cumulative sum
  double partial = Inverse_->ApplyInverseFlops();
  double total;
  Comm().SumAll(&partial, &total, 1);
  ApplyInverseFlops_ += total - pre_total;

  // FIXME: right now I am skipping the overlap and singletons
  ++NumApplyInverse_;
  ApplyInverseTime_ += Time_->ElapsedTime();

  return(0);
 
}

//==============================================================================
template<typename T>
std::ostream& Ifpack_AdditiveSchwarz<T>::
Print(std::ostream& os) const
{
  double IF = InitializeFlops();
  double CF = ComputeFlops();
  double AF = ApplyInverseFlops();

  double IFT = 0.0, CFT = 0.0, AFT = 0.0;
  if (InitializeTime() != 0.0)
    IFT = IF / InitializeTime();
  if (ComputeTime() != 0.0)
    CFT = CF / ComputeTime();
  if (ApplyInverseTime() != 0.0)
    AFT = AF / ApplyInverseTime();

  if (Matrix().Comm().MyPID())
    return(os);

  os << endl;
  os << "================================================================================" << endl;
  os << "Ifpack_AdditiveSchwarz, overlap level = " << OverlapLevel_ << endl;
  if (CombineMode_ == Insert)
    os << "Combine mode                          = Insert" << endl;
  else if (CombineMode_ == Add)
    os << "Combine mode                          = Add" << endl;
  else if (CombineMode_ == Zero)
    os << "Combine mode                          = Zero" << endl;
  else if (CombineMode_ == Average)
    os << "Combine mode                          = Average" << endl;
  else if (CombineMode_ == AbsMax)
    os << "Combine mode                          = AbsMax" << endl;

  os << "Condition number estimate             = " << Condest_ << endl;
  os << "Global number of rows                 = " << Matrix_->NumGlobalRows() << endl;

  os << endl;
  os << "Phase           # calls   Total Time (s)       Total MFlops     MFlops/s" << endl;
  os << "-----           -------   --------------       ------------     --------" << endl;
  os << "Initialize()    "   << std::setw(5) << NumInitialize()
     << "  " << std::setw(15) << InitializeTime() 
     << "  " << std::setw(15) << 1.0e-6 * IF 
     << "  " << std::setw(15) << 1.0e-6 * IFT << endl;
  os << "Compute()       "   << std::setw(5) << NumCompute() 
     << "  " << std::setw(15) << ComputeTime()
     << "  " << std::setw(15) << 1.0e-6 * CF
     << "  " << std::setw(15) << 1.0e-6 * CFT << endl;
  os << "ApplyInverse()  "   << std::setw(5) << NumApplyInverse() 
     << "  " << std::setw(15) << ApplyInverseTime()
     << "  " << std::setw(15) << 1.0e-6 * AF
     << "  " << std::setw(15) << 1.0e-6 * AFT << endl;
  os << "================================================================================" << endl;
  os << endl;

  return(os);
}

#include "Ifpack_Condest.h"
//==============================================================================
template<typename T>
double Ifpack_AdditiveSchwarz<T>::
Condest(const Ifpack_CondestType CT, const int MaxIters, 
        const double Tol, Epetra_RowMatrix* Matrix)
{
  if (!IsComputed()) // cannot compute right now
    return(-1.0);

  Condest_ = Ifpack_Condest(*this, CT, MaxIters, Tol, Matrix);

  return(Condest_);
}

#endif // IFPACK_ADDITIVESCHWARZ_H
