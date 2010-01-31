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

#ifndef IFPACK_ILU_H
#define IFPACK_ILU_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_Condest.h"
#include "Ifpack_ScalingType.h"
#include "Ifpack_IlukGraph.h"
#include "Epetra_CompObject.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_BlockMap.h"
#include "Epetra_Map.h"
#include "Epetra_Object.h"
#include "Epetra_Comm.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Time.h"
#include "Teuchos_RefCountPtr.hpp"

namespace Teuchos {
  class ParameterList;
}

//! Ifpack_ILU: A class for constructing and using an incomplete lower/upper (ILU) factorization of a given Epetra_RowMatrix.

/*! The Ifpack_ILU class computes a "Relaxed" ILU factorization with level k fill 
    of a given Epetra_RowMatrix. 

    <P> Please refer to \ref ifp_ilu for a general description of the ILU algorithm.

    <P>The complete list of supported parameters is reported in page \ref ifp_params. 

    \author Mike Heroux, Marzio Sala, SNL 9214.

    \date Last modified on 22-Jan-05.
*/    

class Ifpack_ILU: public Ifpack_Preconditioner {
      
public:
  // @{ Constructors and destructors.
  //! Constructor
  Ifpack_ILU(Epetra_RowMatrix* A);
  
  //! Destructor
  ~Ifpack_ILU()
  {
    Destroy();
  }

  // @}
  // @{ Construction methods
  
  //! Initialize the preconditioner, does not touch matrix values.
  int Initialize();
  
  //! Returns \c true if the preconditioner has been successfully initialized.
  bool IsInitialized() const
  {
    return(IsInitialized_);
  }

  //! Compute ILU factors L and U using the specified graph, diagonal perturbation thresholds and relaxation parameters.
  /*! This function computes the ILU(k) factors L and U using the current:
    <ol>
    <li> Ifpack_IlukGraph specifying the structure of L and U.
    <li> Value for the ILU(k) relaxation parameter.
    <li> Value for the \e a \e priori diagonal threshold values.
    </ol>
    InitValues() must be called before the factorization can proceed.
   */
  int Compute();

  //! If factor is completed, this query returns true, otherwise it returns false.
  bool IsComputed() const 
  {
    return(IsComputed_);
  }

  //! Set parameters using a Teuchos::ParameterList object.
  /* This method is only available if the Teuchos package is enabled.
     This method recognizes four parameter names: relax_value,
     absolute_threshold, relative_threshold and overlap_mode. These names are
     case insensitive, and in each case except overlap_mode, the ParameterEntry
     must have type double. For overlap_mode, the ParameterEntry must have
     type Epetra_CombineMode.
  */
  int SetParameters(Teuchos::ParameterList& parameterlist);

  //! If set true, transpose of this operator will be applied.
  /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
      affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
      does not support transpose use, this method should return a value of -1.
      
      \param
       UseTranspose - (In) If true, multiply by the transpose of operator, otherwise just use operator.

      \return Always returns 0.
  */
  int SetUseTranspose(bool UseTranspose) {UseTranspose_ = UseTranspose; return(0);};
  // @}

  // @{ Mathematical functions.
  // Applies the matrix to X, returns the result in Y.
  int Apply(const Epetra_MultiVector& X, 
	       Epetra_MultiVector& Y) const
  {
    return(Multiply(false,X,Y));
  }

  int Multiply(bool Trans, const Epetra_MultiVector& X, 
	       Epetra_MultiVector& Y) const;

  //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
  /*! In this implementation, we use several existing attributes to determine how virtual
      method ApplyInverse() should call the concrete method Solve().  We pass in the UpperTriangular(), 
      the Epetra_CrsMatrix::UseTranspose(), and NoDiagonal() methods. The most notable warning is that
      if a matrix has no diagonal values we assume that there is an implicit unit diagonal that should
      be accounted for when doing a triangular solve.

    \param 
	   X - (In) A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
	   Y - (Out) A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Computes the estimated condition number and returns the value.
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap, 
                 const int MaxIters = 1550,
                 const double Tol = 1e-9,
		 Epetra_RowMatrix* Matrix = 0);

  //! Returns the computed estimated condition number, or -1.0 if not computed.
  double Condest() const
  {
    return(Condest_);
  }

  // @}
  // @{ Query methods
  
  //! Returns the address of the L factor associated with this factored matrix.
  const Epetra_CrsMatrix & L() const {return(*L_);};
    
  //! Returns the address of the D factor associated with this factored matrix.
  const Epetra_Vector & D() const {return(*D_);};
    
  //! Returns the address of the L factor associated with this factored matrix.
  const Epetra_CrsMatrix & U() const {return(*U_);};

  //! Returns a character string describing the operator
  const char* Label() const {return(Label_);}

  //! Sets label for \c this object.
  int SetLabel(const char* Label)
  {
    strcpy(Label_,Label);
    return(0);
  }
  
  //! Returns 0.0 because this class cannot compute Inf-norm.
  double NormInf() const {return(0.0);};

  //! Returns false because this class cannot compute an Inf-norm.
  bool HasNormInf() const {return(false);};

  //! Returns the current UseTranspose setting.
  bool UseTranspose() const {return(UseTranspose_);};

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const {return(U_->OperatorDomainMap());};

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const{return(L_->OperatorRangeMap());};

  //! Returns the Epetra_BlockMap object associated with the range of this matrix operator.
  const Epetra_Comm & Comm() const{return(Comm_);};

  //! Returns a reference to the matrix to be preconditioned.
  const Epetra_RowMatrix& Matrix() const
  { 
    return(*A_);
  }

  //! Prints on stream basic information about \c this object.
  virtual ostream& Print(ostream& os) const;

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

private:

  // @}
  // @{ Private methods

  //! Copy constructor (should never be used)
  Ifpack_ILU(const Ifpack_ILU& RHS) :
    Comm_(RHS.Comm()),
    Time_(RHS.Comm())
  {}

  //! operator= (should never be used)
  Ifpack_ILU& operator=(const Ifpack_ILU& RHS)
  {
    return(*this);
  }

  //! Destroys all internal data
  void Destroy();

  //! Returns the result of a Ifpack_ILU forward/back solve on a Epetra_MultiVector X in Y.
  /*! 
    \param In
    Trans -If true, solve transpose problem.
    \param 
    X - (In) A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
    Y - (Out) A Epetra_MultiVector of dimension NumVectorscontaining result.
    
    \return Integer error code, set to 0 if successful.
  */
  int Solve(bool Trans, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  int ComputeSetup();
  int InitAllValues(const Epetra_RowMatrix & A, int MaxNumEntries);

  //! Returns the level of fill.
  int LevelOfFill() const {return LevelOfFill_;}

  //! Get ILU(k) relaxation parameter
  double RelaxValue() const {return RelaxValue_;}

  //! Get absolute threshold value
  double AbsoluteThreshold() const {return Athresh_;}

  //! Get relative threshold value
  double RelativeThreshold() const {return Rthresh_;}

  //! Returns the number of global matrix rows.
  int NumGlobalRows() const {return(Graph().NumGlobalRows());};
  
  //! Returns the number of global matrix columns.
  int NumGlobalCols() const {return(Graph().NumGlobalCols());};
  
  //! Returns the number of nonzero entries in the global graph.
  int NumGlobalNonzeros() const {return(L().NumGlobalNonzeros()+U().NumGlobalNonzeros());};
  
  //! Returns the number of diagonal entries found in the global input graph.
  virtual int NumGlobalBlockDiagonals() const {return(Graph().NumGlobalBlockDiagonals());};
  
  //! Returns the number of local matrix rows.
  int NumMyRows() const {return(Graph().NumMyRows());};
  
  //! Returns the number of local matrix columns.
  int NumMyCols() const {return(Graph().NumMyCols());};
  
  //! Returns the number of nonzero entries in the local graph.
  int NumMyNonzeros() const {return(L().NumMyNonzeros()+U().NumMyNonzeros());};
  
  //! Returns the number of diagonal entries found in the local input graph.
  virtual int NumMyBlockDiagonals() const {return(Graph().NumMyBlockDiagonals());};
  
  //! Returns the number of nonzero diagonal values found in matrix.
  virtual int NumMyDiagonals() const {return(NumMyDiagonals_);};
  
  //! Returns the index base for row and column indices for this graph.
  int IndexBase() const {return(Graph().IndexBase());};
  
  //! Returns the address of the Ifpack_IlukGraph associated with this factored matrix.
  const Ifpack_IlukGraph & Graph() const {return(*Graph_);};
  
  //! Returns a reference to the matrix.
  Epetra_RowMatrix& Matrix()
  {
    return(*A_);
  }

  // @}
  // @{ Internal data
  
  //! Pointer to the Epetra_RowMatrix to factorize
  Teuchos::RefCountPtr<Epetra_RowMatrix> A_;
  Teuchos::RefCountPtr<Ifpack_IlukGraph> Graph_;
  Teuchos::RefCountPtr<Epetra_CrsGraph> CrsGraph_;
  Teuchos::RefCountPtr<Epetra_Map> IlukRowMap_;
  Teuchos::RefCountPtr<Epetra_Map> IlukDomainMap_;
  Teuchos::RefCountPtr<Epetra_Map> IlukRangeMap_;
  const Epetra_Map * U_DomainMap_;
  const Epetra_Map * L_RangeMap_;
  const Epetra_Comm & Comm_;
  //! Contains the L factors
  Teuchos::RefCountPtr<Epetra_CrsMatrix> L_;
  //! Contains the U factors.
  Teuchos::RefCountPtr<Epetra_CrsMatrix> U_;
  Teuchos::RefCountPtr<Epetra_CrsGraph> L_Graph_;
  Teuchos::RefCountPtr<Epetra_CrsGraph> U_Graph_;
  //! Diagonal of factors
  Teuchos::RefCountPtr<Epetra_Vector> D_;
  bool UseTranspose_;

  int NumMyDiagonals_;
  bool Allocated_;
  bool ValuesInitialized_;
  bool Factored_;
  //! Relaxation value
  double RelaxValue_;
  //! absolute threshold
  double Athresh_;
  //! relative threshold
  double Rthresh_;
  //! condition number estimate
  double Condest_;
  //! Level of fill
  int LevelOfFill_;
  //! If \c true, the preconditioner has been successfully initialized.
  bool IsInitialized_;
  //! If \c true, the preconditioner has been successfully computed.
  bool IsComputed_;
  //! Label of \c this object.
  char Label_[160];
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
  //! Used for timing issues
  mutable Epetra_Time Time_;

};

#endif /* IFPACK_ILU_H */
