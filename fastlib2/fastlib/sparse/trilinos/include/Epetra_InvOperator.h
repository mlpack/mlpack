
/*@HEADER
// ***********************************************************************
// 
//        Epetra: Linear Algebra Services Package 
//          Copyright (2001) Sandia Corporation
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

#ifndef EPETRA_INVOPERATOR_H
#define EPETRA_INVOPERATOR_H

class Epetra_MultiVector;
class Epetra_BlockMap;
class Epetra_Comm;
#include <string>
#include "Epetra_Operator.h"

//! Epetra_InvOperator: An implementation of the Epetra_Operator class that reverses the role of Apply() and ApplyInverse() methods.
/*! The Epetra_InvOperator class implements Epetra_Operator using another pre-constructed Epetra_Operator object.
    Once constructed, an Epetra_InvOperator can be used as the inverse of the input operator
    object as long as the appropriate Apply and ApplyInverse methods are implemented in the original Epetra_Operator object.
*/    

class Epetra_InvOperator: public virtual Epetra_Operator {
      
 public:

   //! @name Constructor
  //@{ 
    //! Uses an Epetra_Operator instance to implement the Epetra_Operator interface.
  /*! Facilitates the use of an Epetra_Operator instance as an inverse operator.  
    \param In - A fully-constructed Epetra_Operator object.
  */
  Epetra_InvOperator(Epetra_Operator * operatorIn) {
    operator_ = operatorIn; 
    Label_ = "Inverse of " + string(operatorIn->Label());
    return;
  }
    //! Destructor
  virtual ~Epetra_InvOperator(){}
  //@}
  
  //! @name Atribute set methods
  //@{ 

    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
	does not support transpose use, this method should return a value of -1.
      
    \param In
	   UseTranspose - If true, multiply by the transpose of operator, otherwise just use operator.

    \warning - This method has no effect and returns -1 as error code.
  */
  int SetUseTranspose(bool UseTranspose){EPETRA_CHK_ERR(operator_->SetUseTranspose(UseTranspose)); return(0);}
  //@}
  
  //! @name Mathematical functions
  //@{ 

    //! Returns the result of a Epetra_InvOperator applied to a Epetra_MultiVector X in Y.
    /*! 
    \param In
	   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
	   Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \warning - This method has no effect and returns -1 as error code.
  */
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {EPETRA_CHK_ERR(operator_->ApplyInverse(X,Y)); return(0);}

  //! Returns the result of a Epetra_InvOperator inverse applied to an Epetra_MultiVector X in Y.
  /*! 
    \param In
    X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
    Y -A Epetra_MultiVector of dimension NumVectors containing result.
    
    \return Integer error code, set to 0 if successful.
  */
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const{EPETRA_CHK_ERR(operator_->Apply(X,Y)); return(0);}
  
  //! Returns the infinity norm of the global matrix.
  /* Returns the quantity \f$ \| A \|_\infty\f$ such that
     \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].
     
     \warning This method must not be called unless HasNormInf() returns true.
  */ 
  double NormInf() const {return(operator_->NormInf());}
  
  //! @name Atribute access functions
  //@{ 

  //! Returns a character string describing the operator
  const char * Label() const {return(Label_.c_str());}

  //! Returns a pointer to the Epetra_Operator operator object that was used to create this Epetra_InvOperator object.
  Epetra_Operator * Operator() const {return(operator_);}

  
  //! Returns the current UseTranspose setting.
  bool UseTranspose() const {return(operator_->UseTranspose());}
  
  //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
  bool HasNormInf() const {return(operator_->HasNormInf());};
  
  //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
  const Epetra_Comm & Comm() const {return(operator_->Comm());}
  
  //! Returns the Epetra_BlockMap object associated with the domain of this matrix operator.
  const Epetra_Map & OperatorDomainMap() const
  {
    if (!UseTranspose()) return(operator_->OperatorRangeMap());
    else return(operator_->OperatorDomainMap());
  }
  
  //! Returns the Epetra_BlockMap object associated with the range of this matrix operator.
  const Epetra_Map & OperatorRangeMap() const
  {
    if (!UseTranspose()) return(operator_->OperatorDomainMap());
    else return(operator_->OperatorRangeMap());
  }
  //@}
  
 protected:

  Epetra_Operator * operator_;
  string Label_;
};

#endif /* EPETRA_INVOPERATOR_H */

