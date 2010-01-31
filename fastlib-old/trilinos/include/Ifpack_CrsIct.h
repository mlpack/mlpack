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

#ifndef _IFPACK_CRSICT_H_
#define _IFPACK_CRSICT_H_

#include "Ifpack_ScalingType.h"
#include "Ifpack_IlukGraph.h"
#include "Epetra_CombineMode.h"
#include "Epetra_CompObject.h"
#include "Epetra_Operator.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Object.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

#include "Teuchos_RefCountPtr.hpp"

class Epetra_Comm;
class Epetra_Map;

namespace Teuchos {
  class ParameterList;
}

//! Ifpack_CrsIct: A class for constructing and using an incomplete Cholesky factorization of a given Epetra_CrsMatrix.

/*! The Ifpack_CrsIct class computes a threshold based incomplete LDL^T factorization 
    of a given Epetra_CrsMatrix.  The factorization 
    that is produced is a function of several parameters:
<ol>
  <li> Maximum number of entries per row/column in factor - The factorization will contain at most this number of nonzero
       terms in each row/column of the factorization.

  <li> Diagonal perturbation - Prior to computing the factorization, it is possible to modify the diagonal entries of the matrix
       for which the factorization will be computing.  If the absolute and relative perturbation values are zero and one,
       respectively, the
       factorization will be compute for the original user matrix A.  Otherwise, the factorization
       will computed for a matrix that differs from the original user matrix in the diagonal values only.  Below we discuss
       the details of diagonal perturbations.
       The absolute and relative threshold values are set by calling SetAbsoluteThreshold() and SetRelativeThreshold(), 
       respectively.
</ol>

<b> Estimating Preconditioner Condition Numbers </b>

For ill-conditioned matrices, we often have difficulty computing usable incomplete
factorizations.  The most common source of problems is that the factorization may encounter a small or zero pivot,
in which case the factorization can fail, or even if the factorization
succeeds, the factors may be so poorly conditioned that use of them in
the iterative phase produces meaningless results.  Before we can fix
this problem, we must be able to detect it.  To this end, we use a
simple but effective condition number estimate for \f$(LU)^{-1}\f$.

The condition of a matrix \f$B\f$, called \f$cond_p(B)\f$, is defined as
\f$cond_p(B) = \|B\|_p\|B^{-1}\|_p\f$ in some appropriate norm \f$p\f$.  \f$cond_p(B)\f$
gives some indication of how many accurate floating point
digits can be expected from operations involving the matrix and its
inverse.  A condition number approaching the accuracy of a given
floating point number system, about 15 decimal digits in IEEE double
precision, means that any results involving \f$B\f$ or \f$B^{-1}\f$ may be
meaningless.

The \f$\infty\f$-norm of a vector \f$y\f$ is defined as the maximum of the
absolute values of the vector entries, and the \f$\infty\f$-norm of a
matrix C is defined as
\f$\|C\|_\infty = \max_{\|y\|_\infty = 1} \|Cy\|_\infty\f$.
A crude lower bound for the \f$cond_\infty(C)\f$ is
\f$\|C^{-1}e\|_\infty\f$ where \f$e = (1, 1, \ldots, 1)^T\f$.  It is a
lower bound because \f$cond_\infty(C) = \|C\|_\infty\|C^{-1}\|_\infty
\ge \|C^{-1}\|_\infty \ge |C^{-1}e\|_\infty\f$.

For our purposes, we want to estimate \f$cond_\infty(LU)\f$, where \f$L\f$ and
\f$U\f$ are our incomplete factors.  Edmond in his Ph.D. thesis demonstrates that
\f$\|(LU)^{-1}e\|_\infty\f$ provides an effective estimate for
\f$cond_\infty(LU)\f$.  Furthermore, since finding \f$z\f$ such that \f$LUz = y\f$
is a basic kernel for applying the preconditioner, computing this
estimate of \f$cond_\infty(LU)\f$ is performed by setting \f$y = e\f$, calling
the solve kernel to compute \f$z\f$ and then
computing \f$\|z\|_\infty\f$.


<b>\e A \e priori Diagonal Perturbations</b>

Given the above method to estimate the conditioning of the incomplete factors,
if we detect that our factorization is too ill-conditioned
we can improve the conditioning by perturbing the matrix diagonal and
restarting the factorization using
this more diagonally dominant matrix.  In order to apply perturbation,
prior to starting
the factorization, we compute a diagonal perturbation of our matrix
\f$A\f$ and perform the factorization on this perturbed
matrix.  The overhead cost of perturbing the diagonal is minimal since
the first step in computing the incomplete factors is to copy the
matrix \f$A\f$ into the memory space for the incomplete factors.  We
simply compute the perturbed diagonal at this point. 

The actual perturbation values we use are the diagonal values \f$(d_1, d_2, \ldots, d_n)\f$
with \f$d_i = sgn(d_i)\alpha + d_i\rho\f$, \f$i=1, 2, \ldots, n\f$, where
\f$n\f$ is the matrix dimension and \f$sgn(d_i)\f$ returns
the sign of the diagonal entry.  This has the effect of
forcing the diagonal values to have minimal magnitude of \f$\alpha\f$ and
to increase each by an amount proportional to \f$\rho\f$, and still keep
the sign of the original diagonal entry.

<b>Constructing Ifpack_CrsIct objects</b>

Constructing Ifpack_CrsIct objects is a multi-step process.  The basic steps are as follows:
<ol>
  <li> Create Ifpack_CrsIct instance, including storage,  via constructor.
  <li> Enter values via one or more Put or SumInto functions.
  <li> Complete construction via FillComplete call.
</ol>

Note that, even after a matrix is constructed, it is possible to update existing matrix entries.  It is \e not possible to
create new entries.

<b> Counting Floating Point Operations </b>

Each Ifpack_CrsIct object keep track of the number
of \e serial floating point operations performed using the specified object as the \e this argument
to the function.  The Flops() function returns this number as a double precision number.  Using this 
information, in conjunction with the Epetra_Time class, one can get accurate parallel performance
numbers.  The ResetFlops() function resets the floating point counter.

\warning A Epetra_Map is required for the Ifpack_CrsIct constructor.

*/    


class Ifpack_CrsIct: public Epetra_Object, public Epetra_CompObject, public virtual Epetra_Operator {
      
  // Give ostream << function some access to private and protected data/functions.

  friend ostream& operator << (ostream& os, const Ifpack_CrsIct& A);

 public:
  //! Ifpack_CrsIct constuctor with variable number of indices per row.
  /*! Creates a Ifpack_CrsIct object and allocates storage.  
    
    \param In 
           A - User matrix to be factored.
    \param In
           Graph - Graph generated by Ifpack_IlukGraph.
  */
  Ifpack_CrsIct(const Epetra_CrsMatrix &A, double Droptol = 1.0E-4, int Lfil = 20);
  
  //! Copy constructor.
  Ifpack_CrsIct(const Ifpack_CrsIct & IctOperator);

  //! Ifpack_CrsIct Destructor
  virtual ~Ifpack_CrsIct();

  //! Set absolute threshold value
  void SetAbsoluteThreshold( double Athresh) {Athresh_ = Athresh; return;}

  //! Set relative threshold value
  void SetRelativeThreshold( double Rthresh) {Rthresh_ = Rthresh; return;}

  //! Set overlap mode type
  void SetOverlapMode( Epetra_CombineMode OverlapMode) {OverlapMode_ = OverlapMode; return;}

  //! Set parameters using a Teuchos::ParameterList object.
  /* This method is only available if the Teuchos package is enabled.
     This method recognizes five parameter names: level_fill, drop_tolerance,
     absolute_threshold, relative_threshold and overlap_mode. These names are
     case insensitive. For level_fill the ParameterEntry must have type int, the 
     threshold entries must have type double and overlap_mode must have type
     Epetra_CombineMode.
  */
  int SetParameters(const Teuchos::ParameterList& parameterlist,
                    bool cerr_warning_if_unused=false);

  //! Initialize L and U with values from user matrix A.
  /*! Copies values from the user's matrix into the nonzero pattern of L and U.
    \param In 
           A - User matrix to be factored.
    \warning The graph of A must be identical to the graph passed in to Ifpack_IlukGraph constructor.
             
   */
  int InitValues(const Epetra_CrsMatrix &A);

  //! If values have been initialized, this query returns true, otherwise it returns false.
  bool ValuesInitialized() const {return(ValuesInitialized_);};

  //! Compute IC factor U using the specified graph, diagonal perturbation thresholds and relaxation parameters.
  /*! This function computes the RILU(k) factors L and U using the current:
    <ol>
    <li> Ifpack_IlukGraph specifying the structure of L and U.
    <li> Value for the RILU(k) relaxation parameter.
    <li> Value for the \e a \e priori diagonal threshold values.
    </ol>
    InitValues() must be called before the factorization can proceed.
   */
  int Factor();

  //! If factor is completed, this query returns true, otherwise it returns false.
  bool Factored() const {return(Factored_);};
  

  // Mathematical functions.
  
  
  //! Returns the result of a Ifpack_CrsIct forward/back solve on a Epetra_MultiVector X in Y.
  /*! 
    \param In
    Trans -If true, solve transpose problem.
    \param In
    X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
    Y -A Epetra_MultiVector of dimension NumVectorscontaining result.
    
    \return Integer error code, set to 0 if successful.
  */
  int Solve(bool Trans, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Returns the result of multiplying U, D and U^T in that order on an Epetra_MultiVector X in Y.
  /*! 
    \param In
    Trans -If true, multiply by L^T, D and U^T in that order.
    \param In
    X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
    Y -A Epetra_MultiVector of dimension NumVectorscontaining result.
    
    \return Integer error code, set to 0 if successful.
  */
  int Multiply(bool Trans, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Returns the maximum over all the condition number estimate for each local ILU set of factors.
  /*! This functions computes a local condition number estimate on each processor and return the
      maximum over all processor of the estimate.
   \param In
    Trans -If true, solve transpose problem.
    \param Out
    ConditionNumberEstimate - The maximum across all processors of 
    the infinity-norm estimate of the condition number of the inverse of LDU.
  */
  int Condest(bool Trans, double & ConditionNumberEstimate) const;
  // Atribute access functions
  
  //! Get absolute threshold value
  double GetAbsoluteThreshold() {return Athresh_;}

  //! Get relative threshold value
  double GetRelativeThreshold() {return Rthresh_;}
    
  //! Get overlap mode type
  Epetra_CombineMode GetOverlapMode() {return OverlapMode_;}

  //! Returns the number of nonzero entries in the global graph.
  int NumGlobalNonzeros() const {return(U().NumGlobalNonzeros()+D().GlobalLength());};
 
  //! Returns the number of nonzero entries in the local graph.
  int NumMyNonzeros() const {return(U().NumMyNonzeros()+D().MyLength());};
  //! Returns the address of the D factor associated with this factored matrix.
  const Epetra_Vector & D() const {return(*D_);};
    
  //! Returns the address of the U factor associated with this factored matrix.
  const Epetra_CrsMatrix & U() const {return(*U_);};

  //@{ \name Additional methods required to support the Epetra_Operator interface.

    //! Returns a character string describing the operator
    const char * Label() const {return(Epetra_Object::Label());};
    
    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
	does not support transpose use, this method should return a value of -1.
      
    \param In
	   UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.

    \return Always returns 0.
  */
  int SetUseTranspose(bool UseTranspose) {UseTranspose_ = UseTranspose; return(0);};

    //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
    /*! Note that this implementation of Apply does NOT perform a forward back solve with
        the LDU factorization.  Instead it applies these operators via multiplication with 
	U, D and L respectively.  The ApplyInverse() method performs a solve.

    \param In
	   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
	   Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {
    return(Multiply(Ifpack_CrsIct::UseTranspose(), X, Y));};

    //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
    /*! In this implementation, we use several existing attributes to determine how virtual
        method ApplyInverse() should call the concrete method Solve().  We pass in the UpperTriangular(), 
	the Epetra_CrsMatrix::UseTranspose(), and NoDiagonal() methods. The most notable warning is that
	if a matrix has no diagonal values we assume that there is an implicit unit diagonal that should
	be accounted for when doing a triangular solve.

    \param In
	   X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
	   Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {
    return(Solve(Ifpack_CrsIct::UseTranspose(), X, Y));};

    //! Returns 0.0 because this class cannot compute Inf-norm.
    double NormInf() const {return(0.0);};

    //! Returns false because this class cannot compute an Inf-norm.
    bool HasNormInf() const {return(false);};

    //! Returns the current UseTranspose setting.
    bool UseTranspose() const {return(UseTranspose_);};

    //! Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map & OperatorDomainMap() const {return(A_.DomainMap());};

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map & OperatorRangeMap() const{return(A_.RangeMap());};

    //! Returns the Epetra_BlockMap object associated with the range of this matrix operator.
    const Epetra_Comm & Comm() const{return(Comm_);};
  //@}

 protected:
  void SetFactored(bool Flag) {Factored_ = Flag;};
  void SetValuesInitialized(bool Flag) {ValuesInitialized_ = Flag;};
  bool Allocated() const {return(Allocated_);};
  int SetAllocated(bool Flag) {Allocated_ = Flag; return(0);};
  
 private:
  
  
  int Allocate();
    
  const Epetra_CrsMatrix &A_;
  const Epetra_Comm & Comm_;
  Teuchos::RefCountPtr<Epetra_CrsMatrix> U_;
  Teuchos::RefCountPtr<Epetra_Vector> D_;
  bool UseTranspose_;

  
  bool Allocated_;
  bool ValuesInitialized_;
  bool Factored_;
  mutable double Condest_;
  double Athresh_;
  double Rthresh_;
  double Droptol_;
  int Lfil_;

  mutable Teuchos::RefCountPtr<Epetra_MultiVector> OverlapX_;
  mutable Teuchos::RefCountPtr<Epetra_MultiVector> OverlapY_;
  int LevelOverlap_;
  Epetra_CombineMode OverlapMode_;

  void * Aict_;
  void * Lict_;
  double * Ldiag_;

};

//! << operator will work for Ifpack_CrsIct.
ostream& operator << (ostream& os, const Ifpack_CrsIct& A);

#endif /* _IFPACK_CRSICT_H_ */
