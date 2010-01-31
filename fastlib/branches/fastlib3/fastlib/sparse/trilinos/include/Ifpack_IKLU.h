/*@HEADER
// ***********************************************************************
// 
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
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

#ifndef IFPACK_IKLU_H
#define IFPACK_IKLU_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_CondestType.h"
#include "Ifpack_ScalingType.h"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_IKLU_Utils.h"	
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Time.h"
#include "Teuchos_RefCountPtr.hpp"

class Epetra_RowMatrix;
class Epetra_SerialComm;
class Epetra_Comm;
class Epetra_Map;
class Epetra_MultiVector;

namespace Teuchos {
  class ParameterList;
}

//! Ifpack_IKLU: A class for constructing and using an incomplete Cholesky factorization of a given Epetra_RowMatrix.

/*! The Ifpack_IKLU class computes a "Relaxed" IKLU factorization with level k fill 
    of a given Epetra_RowMatrix. 

    <P> Please refer to \ref ifp_ilu for a general description of the ILU algorithm.

    <P>The complete list of supported parameters is reported in page \ref ifp_params. 

    \author Heidi Thornquist, Org. 1437

    \date Last modified on 28-Nov-06.
*/    
class Ifpack_IKLU: public Ifpack_Preconditioner {
      
public:
  // @{ Constructors and Destructors
  //! Ifpack_IKLU constuctor with variable number of indices per row.
  Ifpack_IKLU(const Epetra_RowMatrix* A);
  
  //! Ifpack_IKLU Destructor 
  virtual ~Ifpack_IKLU();

  // @}
  // @{ Construction methods
  //! Set parameters using a Teuchos::ParameterList object.
  /* This method is only available if the Teuchos package is enabled.
     This method recognizes five parameter names: level_fill, drop_tolerance,
     absolute_threshold, relative_threshold and overlap_mode. These names are
     case insensitive. For level_fill the ParameterEntry must have type int, the 
     threshold entries must have type double and overlap_mode must have type
     Epetra_CombineMode.
  */
  int SetParameters(Teuchos::ParameterList& parameterlis);

  //! Initialize L and U with values from user matrix A.
  /*! Copies values from the user's matrix into the nonzero pattern of L and U.
    \param In 
           A - User matrix to be factored.
    \warning The graph of A must be identical to the graph passed in to Ifpack_IlukGraph constructor.
             
   */
  int Initialize();

  //! Returns \c true if the preconditioner has been successfully initialized.
  bool IsInitialized() const
  {
    return(IsInitialized_);
  }

  //! Compute IC factor U using the specified graph, diagonal perturbation thresholds and relaxation parameters.
  /*! This function computes the RILU(k) factors L and U using the current:
    <ol>
    <li> Ifpack_IlukGraph specifying the structure of L and U.
    <li> Value for the RILU(k) relaxation parameter.
    <li> Value for the \e a \e priori diagonal threshold values.
    </ol>
    InitValues() must be called before the factorization can proceed.
   */
  int Compute();

  //! If factor is completed, this query returns true, otherwise it returns false.
  bool IsComputed() const {return(IsComputed_);};

  // Mathematical functions.
  
  //! Returns the result of a Ifpack_IKLU forward/back solve on a Epetra_MultiVector X in Y.
  /*! 
    \param 
    X - (In) A Epetra_MultiVector of dimension NumVectors to solve for.
    \param 
    Y - (Out) A Epetra_MultiVector of dimension NumVectorscontaining result.
    
    \return Integer error code, set to 0 if successful.
  */
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Computed the estimated condition number and returns the value.
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap, 
                 const int MaxIters = 1550,
                 const double Tol = 1e-9,
		 Epetra_RowMatrix* Matrix = 0);

  //! Returns the computed estimated condition number, or -1.0 if no computed.
  double Condest() const
  {
    return(Condest_);
  }

  //! If set true, transpose of this operator will be applied.
  /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
      affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
      does not support transpose use, this method should return a value of -1.
      
     \param
     UseTranspose - (In) If true, multiply by the transpose of operator, otherwise just use operator.

     \return Always returns 0.
  */
  int SetUseTranspose(bool UseTranspose) {UseTranspose_ = UseTranspose; return(0);};

  //! Returns 0.0 because this class cannot compute Inf-norm.
  double NormInf() const {return(0.0);};

  //! Returns false because this class cannot compute an Inf-norm.
  bool HasNormInf() const {return(false);};

  //! Returns the current UseTranspose setting.
  bool UseTranspose() const {return(UseTranspose_);};

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const {return(A_.OperatorDomainMap());};

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const{return(A_.OperatorRangeMap());};

  //! Returns the Epetra_BlockMap object associated with the range of this matrix operator.
  const Epetra_Comm & Comm() const{return(Comm_);};

  //! Returns a reference to the matrix to be preconditioned.
  const Epetra_RowMatrix& Matrix() const
  {
    return(A_);
  }

  //! Returns a reference to the L factor.
  const Epetra_CrsMatrix & L() const {return(*L_);};
  
  //! Returns a reference to the U factor.
  const Epetra_CrsMatrix & U() const {return(*U_);};
    
  //! Returns the label of \c this object.
  const char* Label() const
  {
    return(Label_.c_str());
  }

  //! Sets the label for \c this object
  int SetLabel(const char* Label)
  {
    Label_ = Label;
    return(0);
  }
 
  //! Prints basic information on iostream. This function is used by operator<<.
  virtual ostream& Print(std::ostream& os) const;

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
    return(0.0);
  }

  virtual double ComputeFlops() const
  {
    return(ComputeFlops_);
  }

  virtual double ApplyInverseFlops() const
  {
    return(ApplyInverseFlops_);
  }

  inline double LevelOfFill() const {
    return(LevelOfFill_);
  }

  //! Set relative threshold value
  inline double RelaxValue() const {
    return(Relax_);
  }

  //! Get absolute threshold value
  inline double AbsoluteThreshold() const
  {
    return(Athresh_);
  }

  //! Get relative threshold value
  inline double RelativeThreshold() const
  {
    return(Rthresh_);
  }
    
  //! Gets the dropping tolerance
  inline double DropTolerance() const
  {
    return(DropTolerance_);
  }
    
  //! Returns the number of nonzero entries in the global graph.
  int NumGlobalNonzeros() const {
    // FIXME: diagonal of L_ should not be stored
    return(L().NumGlobalNonzeros() + U().NumGlobalNonzeros() - L().NumGlobalRows());
  }
 
  //! Returns the number of nonzero entries in the local graph.
  int NumMyNonzeros() const {
    return(L().NumMyNonzeros() + U().NumMyNonzeros());
  }

private:
  
  // @}
  // @{ Internal methods

  //! Copy constructor (should never be used)
  Ifpack_IKLU(const Ifpack_IKLU& RHS) :
    A_(RHS.Matrix()),
    Comm_(RHS.Comm()),
    Time_(Comm())
  {};

  //! operator= (should never be used)
  Ifpack_IKLU& operator=(const Ifpack_IKLU& RHS)
  {
    return(*this);
  }

  //! Releases all allocated memory.
  void Destroy();

  // @}
  // @{ Internal data

  //! reference to the matrix to be preconditioned.
  const Epetra_RowMatrix& A_;
  //! Reference to the communicator object.
  const Epetra_Comm& Comm_;
  //! L factor
  Teuchos::RefCountPtr<Epetra_CrsMatrix> L_;
  //! U factor
  Teuchos::RefCountPtr<Epetra_CrsMatrix> U_;
  //! Condition number estimate.
  double Condest_;
  //! relaxation value
  double Relax_;
  //! Absolute threshold
  double Athresh_;
  //! Relative threshold
  double Rthresh_;
  //! Level-of-fill
  double LevelOfFill_;
  //! Discards all elements below this tolerance
  double DropTolerance_;
  //! Label for \c this object
  string Label_;
  //! \c true if \c this object has been initialized
  bool IsInitialized_;
  //! \c true if \c this object has been computed
  bool IsComputed_;
  //! \c true if transpose has to be used.
  bool UseTranspose_;
  //! Number of local rows.
  int NumMyRows_;
  //! Number of local nonzeros
  int NumMyNonzeros_;
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
  //! Contains the number of flops for Compute().
  double ComputeFlops_;
  //! Contain sthe number of flops for ApplyInverse().
  mutable double ApplyInverseFlops_;
  //! Used for timing purposed
  mutable Epetra_Time Time_;
  //! Global number of nonzeros in L and U factors
  int GlobalNonzeros_;
  Teuchos::RefCountPtr<Epetra_SerialComm> SerialComm_;
  Teuchos::RefCountPtr<Epetra_Map> SerialMap_;

  //! Containers for the matrix storage and permutation
  csr* csrA_;
  css* cssS_;
  //! Container for the L and U factor
  csrn* csrnN_;

}; // Ifpack_IKLU

#endif /* IFPACK_IKLU_H */
