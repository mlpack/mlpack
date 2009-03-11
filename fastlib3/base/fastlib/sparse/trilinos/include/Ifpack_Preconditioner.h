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

#ifndef IFPACK_PRECONDITIONER_H
#define IFPACK_PRECONDITIONER_H

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_CondestType.h"
#include "Epetra_Object.h"
#include "Epetra_Operator.h"
#include "Teuchos_ParameterList.hpp"
class Epetra_RowMatrix;

//! Ifpack_Preconditioner: basic class for preconditioning in Ifpack

/*!
  Class Ifpack_Preconditioner is a pure virtual class, and it defines
  the structure of all Ifpack preconditioners.

  This class is a simple extension to Epetra_Operator. It provides 
  the following additional methods:
  - Initialize() performs all operations based on the graph
    of the matrix (without considering the numerical values);
  - IsInitialized() returns true if the preconditioner
    has been successfully initialized;
  - Compute() computes all is required to apply the
    preconditioner, using matrix values  (and assuming that the
    sparsity of the matrix has not been changed);
  - IsComputed() should return true if the preconditioner
    has been successfully computed, false otherwise.
  - Condest() returns an estimation of the condition number, or -1.0
    if not available
  - Matrix() returns a reference to the matrix to be preconditioned.

It is required that Compute() call Initialize() if IsInitialized()
returns false. The preconditioner is applied by ApplyInverse()
(which returns if IsComputed() is false). Every time that Initialize()
is called, the object destroys all the previously allocated 
information, and re-initialize the preconditioner. Every time
Compute() is called, the object re-computed the actual values of
the preconditioner.

<b>Estimating Preconditioner Condition Numbers</b>

The condition of a matrix \f$B\f$, called \f$cond_p(B)\f$, is defined as
\f$cond_p(B) = \|B\|_p\|B^{-1}\|_p\f$ in some appropriate norm \f$p\f$.  \f$cond_p(B)\f$
gives some indication of how many accurate floating point
digits can be expected from operations involving the matrix and its
inverse.  A condition number approaching the accuracy of a given
floating point number system, about 15 decimal digits in IEEE double
precision, means that any results involving \f$B\f$ or \f$B^{-1}\f$ may be
meaningless.

Method Compute() can be use to estimate of the condition number.
Compute() requires one parameter, of type Ifpack_CondestType
(default value is Ifpack_Cheap; other valid choices are Ifpack_CG and
Ifpack_GMRES).

While Ifpack_CG and Ifpack_GMRES construct and AztecOO solver, and
use methods AZ_cg_condnum and AZ_gmres_condnum to evaluate an
accurate (but very expensive) estimate of the condition number, 
Ifpack_Cheap computes \f$\|(P)^{-1}e\|_\infty\f$, which is
only a very crude estimation of the actual condition number. Note that
this estimated number can be less than 1.0. 
However, this approach has the following advantages:
- since finding \f$z\f$ such that \f$P z = y\f$
is a basic kernel for applying the preconditioner, computing this
estimate of \f$cond_\infty(P^{-1})\f$ is performed by setting \f$y = e\f$, calling
the solve kernel to compute \f$z\f$ and then
computing \f$\|z\|_\infty\f$;
- the only cost is one application of the preconditioner.

If this estimate is very large, the application of the computed 
preconditioner may generate large numerical errors. Hence, the user
may check this number, and decide to recompute the preconditioner is
the computed estimate is larger than a given threshold. This is particularly useful in ICT and RILUK factorizations, as for 
ill-conditioned matrices, we often have difficulty computing usable incomplete
factorizations.  The most common source of problems is that the factorization may encounter a small or zero pivot,
in which case the factorization can fail, or even if the factorization
succeeds, the factors may be so poorly conditioned that use of them in
the iterative phase produces meaningless results.  Before we can fix
this problem, we must be able to detect it.  

  
\note 
  If IFPACK is configured with Teuchos support, method SetParameters()
  should be adopted. Otherwise, users can set parameters (one at-a-time),
  using methods SetParameter(), for integers and doubles.
  \date Sep-04

  Ifpack_Preconditioner objects overload the << operator. Derived
  classes should specify a Print() method, that will be used in
  operator <<.

*/

class Ifpack_Preconditioner : public Epetra_Operator {

public:

  //! Sets all parameters for the preconditioner.
  virtual int SetParameters(Teuchos::ParameterList& List) = 0;

  //! Computes all it is necessary to initialize the preconditioner.
  virtual int Initialize() = 0;

  //! Returns true if the  preconditioner has been successfully initialized, false otherwise.
  virtual bool IsInitialized() const = 0;

  //! Computes all it is necessary to apply the preconditioner.
  virtual int Compute() = 0;

  //! Returns true if the  preconditioner has been successfully computed, false otherwise.
  virtual bool IsComputed() const = 0;

  //! Computes the condition number estimate, returns its value.
  virtual double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                         const int MaxIters = 1550,
                         const double Tol = 1e-9,
			 Epetra_RowMatrix* Matrix = 0) = 0;

  //! Returns the computed condition number estimate, or -1.0 if not computed.
  virtual double Condest() const = 0;

  //! Applies the preconditioner to vector X, returns the result in Y.
  virtual int ApplyInverse(const Epetra_MultiVector& X,
			   Epetra_MultiVector& Y) const = 0;

  //! Returns a pointer to the matrix to be preconditioned.
  virtual const Epetra_RowMatrix& Matrix() const = 0;

  //! Returns the number of calls to Initialize().
  virtual int NumInitialize() const = 0;

  //! Returns the number of calls to Compute().
  virtual int NumCompute() const = 0;

  //! Returns the number of calls to ApplyInverse().
  virtual int NumApplyInverse() const = 0;

  //! Returns the time spent in Initialize().
  virtual double InitializeTime() const = 0;

  //! Returns the time spent in Compute().
  virtual double ComputeTime() const = 0;

  //! Returns the time spent in ApplyInverse().
  virtual double ApplyInverseTime() const = 0;

  //! Returns the number of flops in the initialization phase.
  virtual double InitializeFlops() const = 0;

  //! Returns the number of flops in the computation phase.
  virtual double ComputeFlops() const = 0;

  //! Returns the number of flops in the application of the preconditioner.
  virtual double ApplyInverseFlops() const = 0;

  //! Prints basic information on iostream. This function is used by operator<<.
  virtual ostream& Print(std::ostream& os) const = 0;

};

inline ostream& operator<<(ostream& os, const Ifpack_Preconditioner& obj)
{
  return(obj.Print(os));
}

#endif // IFPACK_PRECONDITIONER_H
