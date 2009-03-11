
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

#ifndef _EPETRA_SERIALDENSEOPERATOR_H_
#define _EPETRA_SERIALDENSEOPERATOR_H_

class Epetra_SerialDenseMatrix;

//! Epetra_SerialDenseOperator: A pure virtual class for using real-valued double-precision operators.
/*! The Epetra_SerialDenseOperator class is a pure virtual class (specifies interface only) that 
    enable the use of real-valued double-precision operators. It is currently implemented by the
    Epetra_SerialDenseMatrix, Epetra_SerialDenseSolver and Epetra_SerialDenseSVD classes.

   
*/    

class Epetra_SerialDenseOperator {
      
 public:

   //! @name Destructor
  //@{ 
    //! Destructor
    virtual ~Epetra_SerialDenseOperator() {};
  //@}
  
  //! @name Atribute set methods
  //@{ 

    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
	does not support transpose use, this method should return a value of -1.
      
    \param In
	   UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.

    \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does not support transpose.
  */
    virtual int SetUseTranspose(bool UseTranspose) = 0;
  //@}
  
  //! @name Mathematical functions
  //@{ 

    //! Returns the result of a Epetra_SerialDenseOperator applied to a Epetra_SerialDenseMatrix X in Y.
    /*! 
    \param In
	   X - A Epetra_SerialDenseMatrix to multiply with operator.
    \param Out
	   Y -A Epetra_SerialDenseMatrix containing result.

    \return Integer error code, set to 0 if successful.
  */
    virtual int Apply(const Epetra_SerialDenseMatrix& X, Epetra_SerialDenseMatrix& Y) = 0;

    //! Returns the result of a Epetra_SerialDenseOperator inverse applied to an Epetra_SerialDenseMatrix X in Y.
    /*! 
    \param In
	   X - A Epetra_SerialDenseMatrix to solve for.
    \param Out
	   Y -A Epetra_SerialDenseMatrix containing result.

    \return Integer error code, set to 0 if successful.

  */
    virtual int ApplyInverse(const Epetra_SerialDenseMatrix & X, Epetra_SerialDenseMatrix & Y) = 0;

    //! Returns the infinity norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_\infty\f$ such that
       \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].

       \warning This method must not be called unless HasNormInf() returns true.
    */ 
    virtual double NormInf() const = 0;
  //@}
  
  //! @name Atribute access functions
  //@{ 

    //! Returns a character string describing the operator
    virtual const char * Label() const = 0;

    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const = 0;

    //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
    virtual bool HasNormInf() const = 0;

    //! Returns the row dimension of operator
    virtual int RowDim() const = 0;

    //! Returns the column dimension of operator
    virtual int ColDim() const = 0;
  //@}

};

#endif /* _EPETRA_OPERATOR_H_ */
