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

#ifndef IFPACK_SPARSKIT_H
#define IFPACK_SPARSKIT_H

#include "Ifpack_ConfigDefs.h"
#ifdef HAVE_IFPACK_SPARSKIT
#include "Ifpack_CondestType.h"
#include "Ifpack_ScalingType.h"
#include "Ifpack_Preconditioner.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
class Epetra_Comm;
class Epetra_Map;
class Epetra_MultiVector;
namespace Teuchos {
  class ParameterList;
}

//! Ifpack_SPARSKIT: A class for constructing and using an SPARSKIT's incomplete factorizations.

class Ifpack_SPARSKIT: public Ifpack_Preconditioner {
      
 public:
  //! Ifpack_SPARSKIT constuctor with variable number of indices per row.
  Ifpack_SPARSKIT(Epetra_RowMatrix* A);
  
  //! Ifpack_SPARSKIT Destructor
  virtual ~Ifpack_SPARSKIT();

  //! Set parameters using a Teuchos::ParameterList object.
  /* This method is only available if the Teuchos package is enabled.
     This method recognizes five parameter names: level_fill, drop_tolerance,
     absolute_threshold, relative_threshold and overlap_mode. These names are
     case insensitive. For level_fill the ParameterEntry must have type int, the 
     threshold entries must have type double and overlap_mode must have type
     Epetra_CombineMode.
  */
  int SetParameters(Teuchos::ParameterList& parameterlis);

  int SetParameter(const string Name, const int Value)
  {
    IFPACK_CHK_ERR(-98);
  }
  int SetParameter(const string Name, const double Value)
  {
    IFPACK_CHK_ERR(-98);
  }

  const Epetra_RowMatrix& Matrix() const
  {
    return(A_);
  }

  Epetra_RowMatrix& Matrix()
  {
    return(A_);
  }

  bool IsInitialized() const
  {
    return(IsInitialized_);
  }

  //! Initializes the preconditioner (do-nothing).
  int Initialize();

  //! Computes the preconditioner by converting the matrix into SPARSKIT's format, then computed the L and U factors.
  int Compute();

  //! If factor is completed, this query returns true, otherwise it returns false.
  bool IsComputed() const 
  {
    return(IsComputed_);
  }

  // Mathematical functions.
  
  //! Returns the result of a forward/back solve on a Epetra_MultiVector X in Y.
  /*! 
    \param In
    Trans -If true, solve transpose problem.
    \param In
    X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
    Y -A Epetra_MultiVector of dimension NumVectorscontaining result.
    
    \return Integer error code, set to 0 if successful.
  */
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
  {
    IFPACK_CHK_ERR(-1);
  }

  //! Returns the maximum over all the condition number estimate for each local ILU set of factors.
  /*! This functions computes a local condition number estimate on each processor and return the
      maximum over all processor of the estimate.
   \param In
    Trans -If true, solve transpose problem.
    \param Out
    ConditionNumberEstimate - The maximum across all processors of 
    the infinity-norm estimate of the condition number of the inverse of LDU.
  */
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap, 
                 const int MaxIters = 1550, 
                 const double Tol = 1e-9,
		 Epetra_RowMatrix* Matrix = 0);

  double Condest() const
  {
    return(Condest_);
  }

  // Atribute access functions
  
  //@{ \name Additional methods required to support the Epetra_Operator interface.

    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
	does not support transpose use, this method should return a value of -1.
      
    \param In
	   UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.

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
  //@}

    const char* Label() const
    {
      return(Label_.c_str());
    }

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
    return(0.0);
  }

  virtual double ApplyInverseFlops() const
  {
    return(0.0);
  }

private:
  Epetra_RowMatrix& A_;
  const Epetra_Comm& Comm_;
  bool UseTranspose_;
  int lfil_;
  double droptol_;
  double tol_;
  double permtol_;
  double alph_;
  int mbloc_;
  string Type_;

  // Factorization in MSR format.
  std::vector<double> alu_;
  std::vector<int> jlu_;
  std::vector<int> ju_;

  string Label_;
  // Permutation vector if required by ILUTP and ILUDP.
  std::vector<int> iperm_;

  double Condest_;

  bool IsInitialized_;
  bool IsComputed_;
 
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

};

#endif // HAVE_IFPACK_SPARSKIT
#endif /* IFPACK_SPARSKIT_H */
