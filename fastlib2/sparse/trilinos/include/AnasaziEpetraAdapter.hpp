// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
//                 Copyright (2004) Sandia Corporation
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
// @HEADER

/*! \file AnasaziEpetraAdapter.hpp
  \brief Declarations of Anasazi multi-vector and operator classes using Epetra_MultiVector and Epetra_Operator classes
*/

#ifndef ANASAZI_EPETRA_ADAPTER_HPP
#define ANASAZI_EPETRA_ADAPTER_HPP

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziMultiVec.hpp"
#include "AnasaziOperator.hpp"

#include "Teuchos_SerialDenseMatrix.hpp"
#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"
#include "Epetra_Map.h"
#include "Epetra_LocalMap.h"

namespace Anasazi {

  //! @name Epetra Adapter Exceptions
  //@{

  /** \brief EpetraMultiVecFailure is thrown when a return value from an Epetra
   * call on an Epetra_MultiVector is non-zero.
   */
  class EpetraMultiVecFailure : public AnasaziError {public:
    EpetraMultiVecFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  /** \brief EpetraOpFailure is thrown when a return value from an Epetra
   * call on an Epetra_Operator is non-zero.
   */
  class EpetraOpFailure : public AnasaziError {public:
    EpetraOpFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  //@}
  
  ///////////////////////////////////////////////////////////////
  //
  //--------template class AnasaziEpetraMultiVec-----------------
  //
  ///////////////////////////////////////////////////////////////
  
  /*! 
    \brief Basic adapter class for Anasazi::MultiVec that uses Epetra_MultiVector.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */
  class EpetraMultiVec : public MultiVec<double>, public Epetra_MultiVector {
  public:
    //! @name Constructors/Destructors
    //@{ 

    //! Basic EpetraMultiVec constructor.
    /*! @param Map [in] An Epetra_LocalMap, Epetra_Map or Epetra_BlockMap.
      @param numvecs [in] Number of vectors in multi-vector.

      \returns Pointer to an EpetraMultiVec
    */
    EpetraMultiVec(const Epetra_BlockMap& Map, const int numvecs);

    //! Copy constructor.
    EpetraMultiVec(const Epetra_MultiVector & P_vec);
    
    //! Create multi-vector with values from two dimensional array.
    /*! @param Map [in] An Epetra_LocalMap, Epetra_Map or Epetra_BlockMap
      @param array [in] Pointer to an array of double precision numbers.  The first vector starts at \c array, the
      second at \c array+stride, and so on.  This array is copied.
      @param numvecs [in] Number of vectors in the multi-vector.
      @param stride [in] The stride between vectors in memory of \c array.

      \returns Pointer to an EpetraMultiVec
    */
    EpetraMultiVec(const Epetra_BlockMap& Map, double * array, const int numvecs, const int stride=0);

    //! Create multi-vector from list of vectors in an existing EpetraMultiVec.
    /*! @param CV [in] Enumerated type set to Copy or View.
      @param P_vec [in] An existing fully constructed Epetra_MultiVector.
      @param index [in] A integer vector containing the indices of the vectors to copy out of \c P_vec.

      \returns Pointer to an EpetraMultiVec
    */
    EpetraMultiVec(Epetra_DataAccess CV, const Epetra_MultiVector& P_vec, const std::vector<int>& index);

    //! Destructor
    virtual ~EpetraMultiVec() {};

    //@}

    //! @name Creation methods
    //@{ 

    /*! \brief Creates a new empty EpetraMultiVec containing \c numvecs columns.
      
    \returns Pointer to an EpetraMultiVec
    */
    MultiVec<double> * Clone ( const int numvecs ) const;

    /*! \brief Creates a new EpetraMultiVec and copies contents of \c *this into
      the new vector (deep copy).
      
      \returns Pointer to an EpetraMultiVec
    */
    MultiVec<double> * CloneCopy () const;

    /*! \brief Creates a new EpetraMultiVec and copies the selected contents of \c *this 
      into the new vector (deep copy).  
      
      The copied vectors from \c *this are indicated by the \c index.size() indices in \c index.
      
      \returns Pointer to an EpetraMultiVec
    */
    MultiVec<double> * CloneCopy ( const std::vector<int>& index ) const;
    
    /*! \brief Creates a new EpetraMultiVec that shares the selected contents of \c *this.
      
    The index of the \c numvecs vectors shallow copied from \c *this are indicated by the
    indices given in \c index.
    
    \returns Pointer to an EpetraMultiVec
    */
    MultiVec<double> * CloneView ( const std::vector<int>& index );

    //@}

    //! @name Attribute methods
    //@{ 

    //! Obtain the vector length of *this.
    int GetNumberVecs () const { return NumVectors(); }

    //! Obtain the number of vectors in *this.
    int GetVecLength () const { return GlobalLength(); }

    //@}

    //! @name Update methods
    //@{ 
    /*! \brief Update \c *this with \f$\alpha AB + \beta (*this)\f$.
     */
    void MvTimesMatAddMv ( double alpha, const MultiVec<double>& A, 
                           const Teuchos::SerialDenseMatrix<int,double>& B, 
                           double beta );

    /*! \brief Replace \c *this with \f$\alpha A + \beta B\f$.
     */
    void MvAddMv ( double alpha, const MultiVec<double>& A, 
                   double beta, const MultiVec<double>& B);

    /*! \brief Compute a dense matrix \c B through the matrix-matrix multiply \f$\alpha A^T(*this)\f$.
    */
    void MvTransMv ( double alpha, const MultiVec<double>& A, Teuchos::SerialDenseMatrix<int,double>& B 
#ifdef HAVE_ANASAZI_EXPERIMENTAL
        , ConjType conj = Anasazi::CONJ
#endif
        ) const;
  
    /*! \brief Compute a vector \c b where the components are the individual dot-products, i.e. \f$ b[i] = A[i]^H(this[i])\f$ where \c A[i] is the i-th column of \c A.
    */
    void MvDot ( const MultiVec<double>& A, std::vector<double>* b
#ifdef HAVE_ANASAZI_EXPERIMENTAL
        , ConjType conj = Anasazi::CONJ
#endif
        ) const;

    /*! \brief Scale each element of the vectors in \c *this with \c alpha.
     */
    void MvScale ( double alpha ) { 
      TEST_FOR_EXCEPTION( this->Scale( alpha )!=0, EpetraMultiVecFailure,
          "Anasazi::EpetraMultiVec::MvScale call to Epetra_MultiVector::Scale() returned a nonzero value.");
    }
    
    /*! \brief Scale each element of the \c i-th vector in \c *this with \c alpha[i].
     */
    void MvScale ( const std::vector<double>& alpha );

    //@}
    //! @name Norm method
    //@{ 
    
    /*! \brief Compute the 2-norm of each individual vector of \c *this.  
      Upon return, \c normvec[i] holds the 2-norm of the \c i-th vector of \c *this
    */
    void MvNorm ( std::vector<double>* normvec ) const {
      if ((normvec!=NULL) && ((int)normvec->size() >= GetNumberVecs()) ) {
        TEST_FOR_EXCEPTION( this->Norm2(&(*normvec)[0])!=0, EpetraMultiVecFailure,
            "Anasazi::EpetraMultiVec::MvNorm call to Epetra_MultiVector::Norm2() returned a nonzero value.");
      }
    };
    //@}
    
    //! @name Initialization methods
    //@{ 
    /*! \brief Copy the vectors in \c A to a set of vectors in \c *this.  
      
    The \c numvecs vectors in \c A are copied to a subset of vectors in \c *this
    indicated by the indices given in \c index.
    */
    void SetBlock ( const MultiVec<double>& A, const std::vector<int>& index );

    /*! \brief Fill the vectors in \c *this with random numbers.
     */
    void MvRandom() { 
      TEST_FOR_EXCEPTION( this->Random()!=0, EpetraMultiVecFailure,
          "Anasazi::EpetraMultiVec::MvRandom call to Epetra_MultiVector::Random() returned a nonzero value.");
    };

    /*! \brief Replace each element of the vectors in \c *this with \c alpha.
     */
    void MvInit ( double alpha ) { 
      TEST_FOR_EXCEPTION( this->PutScalar( alpha )!=0, EpetraMultiVecFailure,
          "Anasazi::EpetraMultiVec::MvInit call to Epetra_MultiVector::PutScalar() returned a nonzero value.");
    };
    
    //@}
    //! @name Print method
    //@{ 
    /*! \brief Print \c *this EpetraMultiVec.
     */
    void MvPrint( std::ostream& os ) const { os << *this << std::endl; };
    //@}

  private:
  };
  //-------------------------------------------------------------
  
  ///////////////////////////////////////////////////////////////
  //
  //--------template class AnasaziEpetraOp---------------------
  //
  ///////////////////////////////////////////////////////////////
  
  /*! 
    \brief Basic adapter class for Anasazi::Operator that uses Epetra_Operator.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */
  class EpetraOp : public virtual Operator<double> {
  public:
    //! @name Constructor/Destructor
    //@{ 
    
    //! Basic constructor.  Accepts reference-counted pointer to an Epetra_Operator.
    EpetraOp(const Teuchos::RCP<Epetra_Operator> &Op );
    
    //! Destructor
    ~EpetraOp();
    //@}
    
    //! @name Operator application method
    //@{ 
    
    /*! \brief This method takes the Anasazi::MultiVec \c X and
      applies the operator to it resulting in the Anasazi::MultiVec \c Y.
    */
    void Apply ( const MultiVec<double>& X, MultiVec<double>& Y ) const;
    //@} 
    
  private:
    Teuchos::RCP<Epetra_Operator> Epetra_Op;
  };
  //-------------------------------------------------------------

  ///////////////////////////////////////////////////////////////
  //
  //--------template class AnasaziEpetraGenOp--------------------
  //
  ///////////////////////////////////////////////////////////////
  
  /*! 
    \brief Adapter class for creating an operators often used in solving generalized eigenproblems.

    This class will apply the operation \f$A^{-1}M\f$ [default] or \f$AM\f$, for the \c Apply method of the
    Epetra_Operator / Anasazi::Operator.  The Anasazi::EpetraGenOp operator is useful when spectral 
    transformations are used within eigensolvers.  For instance, \f$A^{-1}M\f$ is a shift and invert 
    spectral transformation commonly used with Anasazi::BlockKrylovSchur to compute the smallest-magnitude
    eigenvalues for the eigenproblem \f$Ax = \lambda Mx\f$.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */

  class EpetraGenOp : public virtual Operator<double>, public virtual Epetra_Operator {
  public:
    //! Basic constructor for applying operator \f$A^{-1}M\f$ [default] or \f$AM\f$.
    /*! If \c isAInverse is true this operator will apply \f$A^{-1}M\f$, else
      it will apply \f$AM\f$.
    */
    EpetraGenOp(const Teuchos::RCP<Epetra_Operator> &AOp, 
                const Teuchos::RCP<Epetra_Operator> &MOp,
                bool isAInverse = true );

    //! Destructor
    ~EpetraGenOp();
    
    //! Apply method [inherited from Anasazi::Operator class]
    /*! This method will apply \f$A^{-1}M\f$ or \f$AM\f$ to \c X, returning \c Y.
     */
    void Apply ( const MultiVec<double>& X, MultiVec<double>& Y ) const; 

    //! Apply method [inherited from Epetra_Operator class]
    /*! This method will apply \f$A^{-1}M\f$ or \f$AM\f$ to \c X, returning \c Y.
     */
    int Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;

    //! Apply inverse method [inherited from Epetra_Operator class]
    /*! This method will apply \f$(A^{-1}M)^{-1}\f$ or \f$(AM)^{-1}\f$ to \c X, returning \c Y.
     */
    int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;

    //! Returns a character string describing the operator.
    const char* Label() const { return "Epetra_Operator applying A^{-1}M"; };
    
    //! Returns the current UseTranspose setting [always false for this operator].
    bool UseTranspose() const { return (false); };

    //! If set true, the transpose of this operator will be applied [not functional for this operator].
    int SetUseTranspose(bool UseTranspose) { return 0; };
    
    //! Returns true if this object can provide an approximate inf-norm [always false for this operator].
    bool HasNormInf() const { return (false); };
    
    //! Returns the infinity norm of the global matrix [not functional for this operator].
    double NormInf() const  { return (-1.0); };
    
    //! Returns the Epetra_Comm communicator associated with this operator.
    const Epetra_Comm& Comm() const { return Epetra_AOp->Comm(); };

    //! Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map& OperatorDomainMap() const { return Epetra_AOp->OperatorDomainMap(); };

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map& OperatorRangeMap() const { return Epetra_AOp->OperatorRangeMap(); };

  private:
    bool isAInverse;
    Teuchos::RCP<Epetra_Operator> Epetra_AOp;
    Teuchos::RCP<Epetra_Operator> Epetra_MOp;
  };
  
  ///////////////////////////////////////////////////////////////
  //
  //--------template class AnasaziEpetraSymOp--------------------
  //
  ///////////////////////////////////////////////////////////////

  /*! 
    \brief Adapter class for creating a symmetric operator from an Epetra_Operator.

    This class will apply the operation \f$A^TA\f$ [default] or \f$AA^T\f$, for the \c Apply method of the
    Epetra_Operator / Anasazi::Operator.  The Anasazi::EpetraSymOp operator is useful when trying to compute
    a few singular values of the operator \f$A\f$.  The singular values are the square-root of the eigenvalues
    of \f$A^TA\f$ and \f$AA^T\f$.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */

  class EpetraSymOp : public virtual Operator<double>, public virtual Epetra_Operator {
  public:
    //! Basic constructor for applying operator \f$A^TA\f$ [default] or \f$AA^T\f$.
    /*! If \c isTrans is false this operator will apply \f$A^TA\f$, else it will apply \f$AA^T\f$.
    */
    EpetraSymOp(const Teuchos::RCP<Epetra_Operator> &Op, bool isTrans = false );

    //! Destructor
    ~EpetraSymOp();
    
    //! Apply method [inherited from Anasazi::Operator class]
    /*! This method will apply \f$A^TA\f$ or \f$AA^T\f$ to \c X, returning \c Y.
     */
    void Apply ( const MultiVec<double>& X, MultiVec<double>& Y ) const; 

    //! Apply method [inherited from Epetra_Operator class]
    /*! This method will apply \f$A^TA\f$ or \f$AA^T\f$ to \c X, returning \c Y.
     */
    int Apply(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;

    //! Apply inverse method [inherited from Epetra_Operator class]
    /*! This method will apply \f$(A^TA)^{-1}\f$ or \f$(AA^T)^{-1}\f$ to \c X, returning \c Y.
      \note This method is only defined if \f$A^{-1}\f$ is defined for the given Epetra_Operator.
     */
    int ApplyInverse(const Epetra_MultiVector &X, Epetra_MultiVector &Y) const;

    //! Returns a character string describing the operator.
    const char* Label() const { return "Epetra_Operator applying A^TA or AA^T"; };
    
    //! Returns the current UseTranspose setting [always false for this operator].
    bool UseTranspose() const { return (false); };

    //! If set true, the transpose of this operator will be applied [not functional for this operator].
    int SetUseTranspose(bool UseTranspose) { return 0; };
    
    //! Returns true if this object can provide an approximate inf-norm [always false for this operator].
    bool HasNormInf() const { return (false); };
    
    //! Returns the infinity norm of the global matrix [not functional for this operator].
    double NormInf() const  { return (-1.0); };
    
    //! Returns the Epetra_Comm communicator associated with this operator.
    const Epetra_Comm& Comm() const { return Epetra_Op->Comm(); };

    //! Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map& OperatorDomainMap() const { return Epetra_Op->OperatorDomainMap(); };

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map& OperatorRangeMap() const { return Epetra_Op->OperatorRangeMap(); };

  private:
    Teuchos::RCP<Epetra_Operator> Epetra_Op;
    bool isTrans_;
  };


  //////////////////////////////////////////////////////////////////
  //
  //--------template class AnasaziEpetraSymMVOp---------------------
  //
  //////////////////////////////////////////////////////////////////

  /*! 
    \brief Adapter class for creating a symmetric operator from an Epetra_MultiVector.

    This class will apply the operation \f$A^TA\f$ [default] or \f$AA^T\f$, for the \c Apply method of the
    Epetra_Operator / Anasazi::Operator.  The Anasazi::EpetraSymMvOp operator is useful when trying to compute
    a few singular values of the Epetra_MultiVector \f$A\f$.  The singular values are the square-root of the 
    eigenvalues of \f$A^TA\f$ and \f$AA^T\f$.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */

  class EpetraSymMVOp : public virtual Operator<double> {
  public:
    //! Basic constructor for applying operator \f$A^TA\f$ [default] or \f$AA^T\f$.
    /*! If \c isTrans is false this operator will apply \f$A^TA\f$, else it will apply \f$AA^T\f$.
    */
    EpetraSymMVOp(const Teuchos::RCP<const Epetra_MultiVector> &MV, 
                  bool isTrans = false );
    
    //! Destructor
    ~EpetraSymMVOp() {};
    
    //! Apply method 
    /*! This method will apply \f$A^TA\f$ or \f$AA^T\f$ to \c X, returning \c Y.
     */
    void Apply ( const MultiVec<double>& X, MultiVec<double>& Y ) const; 

  private:
    Teuchos::RCP<const Epetra_MultiVector> Epetra_MV;
    Teuchos::RCP<const Epetra_Map> MV_localmap;
    Teuchos::RCP<const Epetra_BlockMap> MV_blockmap;
    bool isTrans_;
  };

  //////////////////////////////////////////////////////////////////
  //
  //--------template class AnasaziEpetraWSymMVOp---------------------
  //
  //////////////////////////////////////////////////////////////////

  /*! 
    \brief Adapter class for creating a weighted symmetric operator from an Epetra_MultiVector and Epetra_Operator.

    This class will apply the operation \f$(WA)^T*WA\f$ for the \c Apply method of the
    Anasazi::Operator.  The Anasazi::EpetraWSymMvOp operator is useful when trying to compute
    a few singular values of the Epetra_MultiVector \f$A\f$ under the weighting matrix \f$W\f$.  
    The singular values are the square-root of the eigenvalues of \f$(WA)^T*WA\f$.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */

  class EpetraWSymMVOp : public virtual Operator<double> {
  public:
    //! Basic constructor for applying operator \f$A^TA\f$ [default] or \f$AA^T\f$.
    /*! If \c isTrans is false this operator will apply \f$A^TA\f$, else it will apply \f$AA^T\f$.
    */
    EpetraWSymMVOp(const Teuchos::RCP<const Epetra_MultiVector> &MV, 
                   const Teuchos::RCP<Epetra_Operator> &OP );
    
    //! Destructor
    ~EpetraWSymMVOp() {};
    
    //! Apply method 
    /*! This method will apply \f$(WA)^T*WA\f$ to \c X, returning \c Y.
     */
    void Apply ( const MultiVec<double>& X, MultiVec<double>& Y ) const; 

  private:
    Teuchos::RCP<const Epetra_MultiVector> Epetra_MV;
    Teuchos::RCP<Epetra_Operator> Epetra_OP;
    Teuchos::RCP<Epetra_MultiVector> Epetra_WMV;
    Teuchos::RCP<const Epetra_Map> MV_localmap;
    Teuchos::RCP<const Epetra_BlockMap> MV_blockmap;
  };

  
  
  ////////////////////////////////////////////////////////////////////
  //
  // Implementation of the Anasazi::MultiVecTraits for Epetra::MultiVector.
  //
  ////////////////////////////////////////////////////////////////////

  /*! 
    \brief Template specialization of Anasazi::MultiVecTraits class using the Epetra_MultiVector class.

    This interface will ensure that any Epetra_MultiVector will be accepted by the Anasazi
    templated solvers.  

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */

  template<>
  class MultiVecTraits<double, Epetra_MultiVector>
  {
  public:

    //! @name Creation methods
    //@{ 

    /*! \brief Creates a new empty Epetra_MultiVector containing \c numvecs columns.
      
    \return Reference-counted pointer to the new Epetra_MultiVector.
    */
    static Teuchos::RCP<Epetra_MultiVector> Clone( const Epetra_MultiVector& mv, const int numvecs )
    { return Teuchos::rcp( new Epetra_MultiVector(mv.Map(), numvecs) ); }

    /*! \brief Creates a new Epetra_MultiVector and copies contents of \c mv into the new vector (deep copy).
      
      \return Reference-counted pointer to the new Epetra_MultiVector.
    */
    static Teuchos::RCP<Epetra_MultiVector> CloneCopy( const Epetra_MultiVector& mv )
    { return Teuchos::rcp( new Epetra_MultiVector( mv ) ); }

    /*! \brief Creates a new Epetra_MultiVector and copies the selected contents of \c mv into the new vector (deep copy).  

      The copied vectors from \c mv are indicated by the \c indeX.size() indices in \c index.      
      \return Reference-counted pointer to the new Epetra_MultiVector.
    */
    static Teuchos::RCP<Epetra_MultiVector> CloneCopy( const Epetra_MultiVector& mv, const std::vector<int>& index )
    { 
      std::vector<int>& tmp_index = const_cast<std::vector<int> &>( index );
      return Teuchos::rcp( new Epetra_MultiVector(::Copy, mv, &tmp_index[0], index.size()) ); 
    }

    /*! \brief Creates a new Epetra_MultiVector that shares the selected contents of \c mv (shallow copy).

    The index of the \c numvecs vectors shallow copied from \c mv are indicated by the indices given in \c index.
    \return Reference-counted pointer to the new Epetra_MultiVector.
    */      
    static Teuchos::RCP<Epetra_MultiVector> CloneView( Epetra_MultiVector& mv, const std::vector<int>& index )
    { 
      std::vector<int>& tmp_index = const_cast<std::vector<int> &>( index );
      return Teuchos::rcp( new Epetra_MultiVector(::View, mv, &tmp_index[0], index.size()) ); 
    }

    /*! \brief Creates a new const Epetra_MultiVector that shares the selected contents of \c mv (shallow copy).

    The index of the \c numvecs vectors shallow copied from \c mv are indicated by the indices given in \c index.
    \return Reference-counted pointer to the new const Epetra_MultiVector.
    */      
    static Teuchos::RCP<const Epetra_MultiVector> CloneView( const Epetra_MultiVector& mv, const std::vector<int>& index )
    { 
      std::vector<int>& tmp_index = const_cast<std::vector<int> &>( index );
      return Teuchos::rcp( new Epetra_MultiVector(::View, mv, &tmp_index[0], index.size()) ); 
    }

    //@}

    //! @name Attribute methods
    //@{ 

    //! Obtain the vector length of \c mv.
    static int GetVecLength( const Epetra_MultiVector& mv )
    { return mv.GlobalLength(); }

    //! Obtain the number of vectors in \c mv
    static int GetNumberVecs( const Epetra_MultiVector& mv )
    { return mv.NumVectors(); }
    //@}

    //! @name Update methods
    //@{ 

    /*! \brief Update \c mv with \f$ \alpha AB + \beta mv \f$.
     */
    static void MvTimesMatAddMv( double alpha, const Epetra_MultiVector& A, 
                                 const Teuchos::SerialDenseMatrix<int,double>& B, 
                                 double beta, Epetra_MultiVector& mv )
    { 
      Epetra_LocalMap LocalMap(B.numRows(), 0, mv.Map().Comm());
      Epetra_MultiVector B_Pvec(::Copy, LocalMap, B.values(), B.stride(), B.numCols());

      TEST_FOR_EXCEPTION( mv.Multiply( 'N', 'N', alpha, A, B_Pvec, beta )!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvNorm call to Epetra_MultiVector::Multiply() returned a nonzero value.");
    }

    /*! \brief Replace \c mv with \f$\alpha A + \beta B\f$.
     */
    static void MvAddMv( double alpha, const Epetra_MultiVector& A, double beta, const Epetra_MultiVector& B, Epetra_MultiVector& mv )
    { 
      TEST_FOR_EXCEPTION( mv.Update( alpha, A, beta, B, 0.0 )!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvAddMv call to Epetra_MultiVector::Update() returned a nonzero value.");
    }

    /*! \brief Compute a dense matrix \c B through the matrix-matrix multiply \f$ \alpha A^Tmv \f$.
    */
    static void MvTransMv( double alpha, const Epetra_MultiVector& A, const Epetra_MultiVector& mv, Teuchos::SerialDenseMatrix<int,double>& B
#ifdef HAVE_ANASAZI_EXPERIMENTAL
                          , ConjType conj = Anasazi::CONJ
#endif
                        )
    { 
      Epetra_LocalMap LocalMap(B.numRows(), 0, mv.Map().Comm());
      Epetra_MultiVector B_Pvec(::View, LocalMap, B.values(), B.stride(), B.numCols());
      
      TEST_FOR_EXCEPTION( B_Pvec.Multiply( 'T', 'N', alpha, A, mv, 0.0 )!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvTransMv call to Epetra_MultiVector::Multiply() returned a nonzero value.");
    }
    
    /*! \brief Compute a vector \c b where the components are the individual dot-products of the \c i-th columns of \c A and \c mv, i.e.\f$b[i] = A[i]^Tmv[i]\f$.
     */
    static void MvDot( const Epetra_MultiVector& mv, const Epetra_MultiVector& A, std::vector<double>* b
#ifdef HAVE_ANASAZI_EXPERIMENTAL
                      , ConjType conj = Anasazi::CONJ
#endif
                      )
    {
      TEST_FOR_EXCEPTION( mv.Dot( A, &(*b)[0] )!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvDot call to Epetra_MultiVector::Dot() returned a nonzero value.");     
    }

    //@}
    //! @name Norm method
    //@{ 

    /*! \brief Compute the 2-norm of each individual vector of \c mv.  
      Upon return, \c normvec[i] holds the value of \f$||mv_i||_2\f$, the \c i-th column of \c mv.
    */
    static void MvNorm( const Epetra_MultiVector& mv, std::vector<double>* normvec )
    { 
      TEST_FOR_EXCEPTION( mv.Norm2(&(*normvec)[0])!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvNorm call to Epetra_MultiVector::Norm2() returned a nonzero value."); 
    }

    //@}
    
    //! @name Initialization methods
    //@{ 
    /*! \brief Copy the vectors in \c A to a set of vectors in \c mv indicated by the indices given in \c index.
     */
    static void SetBlock( const Epetra_MultiVector& A, const std::vector<int>& index, Epetra_MultiVector& mv )
    { 
      // Extract the "numvecs" columns of mv indicated by the index vector.
      int numvecs = index.size();
      std::vector<int>& tmp_index = const_cast<std::vector<int> &>( index );
      Epetra_MultiVector temp_vec(::View, mv, &tmp_index[0], numvecs);

      if ( A.NumVectors() != numvecs ) {
        std::vector<int> index2( numvecs );
        for(int i=0; i<numvecs; i++)
          index2[i] = i;
        Epetra_MultiVector A_vec(::View, A, &index2[0], numvecs);      
        TEST_FOR_EXCEPTION( temp_vec.Update( 1.0, A_vec, 0.0, A_vec, 0.0 )!=0, EpetraMultiVecFailure,
            "MultiVecTraits<double, Epetra_MultiVector>::SetBlock call to Epetra_MultiVector::Update() returned a nonzero value."); 
      }
      else {
        TEST_FOR_EXCEPTION( temp_vec.Update( 1.0, A, 0.0, A, 0.0 )!=0, EpetraMultiVecFailure,
            "MultiVecTraits<double, Epetra_MultiVector>::SetBlock call to Epetra_MultiVector::Update() returned a nonzero value.");
      }
    }
    
    /*! \brief Scale each element of the vectors in \c mv with \c alpha.
     */
    static void MvScale ( Epetra_MultiVector& mv, double alpha ) 
    { 
      TEST_FOR_EXCEPTION( mv.Scale( alpha )!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvScale call to Epetra_MultiVector::Scale() returned a nonzero value."); 
    }
    
    /*! \brief Scale each element of the \c i-th vector in \c mv with \c alpha[i].
     */
    static void MvScale ( Epetra_MultiVector& mv, const std::vector<double>& alpha )
    { 
      // Check to make sure the vector is as long as the multivector has columns.
      int numvecs = mv.NumVectors();
      TEST_FOR_EXCEPTION( (int)alpha.size() != numvecs, std::invalid_argument,
                          "MultiVecTraits<double, Epetra_MultiVector>::MvScale(MV mv,vector alpha) alpha argument size was inconsistent with number of vectors in mv.")

      std::vector<int> tmp_index( 1, 0 );
      for (int i=0; i<numvecs; i++) {
        Epetra_MultiVector temp_vec(::View, mv, &tmp_index[0], 1);
        TEST_FOR_EXCEPTION( temp_vec.Scale( alpha[i] )!=0, EpetraMultiVecFailure,
            "MultiVecTraits<double, Epetra_MultiVector>::MvScale call to Epetra_MultiVector::Scale() returned a nonzero value.");
        tmp_index[0]++;
      }
    }

    /*! \brief Replace the vectors in \c mv with random vectors.
     */
    static void MvRandom( Epetra_MultiVector& mv )
    { 
      TEST_FOR_EXCEPTION( mv.Random()!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvRandom call to Epetra_MultiVector::Random() returned a nonzero value.");
    }

    /*! \brief Replace each element of the vectors in \c mv with \c alpha.
     */
    static void MvInit( Epetra_MultiVector& mv, double alpha = Teuchos::ScalarTraits<double>::zero() )
    { 
      TEST_FOR_EXCEPTION( mv.PutScalar(alpha)!=0, EpetraMultiVecFailure,
          "MultiVecTraits<double, Epetra_MultiVector>::MvInit call to Epetra_MultiVector::PutScalar() returned a nonzero value.");
    }
    
    //@}

    //! @name Print method
    //@{ 

    /*! \brief Print the \c mv multi-vector to the \c os output stream.
     */
    static void MvPrint( const Epetra_MultiVector& mv, std::ostream& os )
    { os << mv << std::endl; }

    //@}
  };        

  ////////////////////////////////////////////////////////////////////
  //
  // Implementation of the Anasazi::OperatorTraits for Epetra::Operator.
  //
  ////////////////////////////////////////////////////////////////////

  /*! 
    \brief Template specialization of Anasazi::OperatorTraits class using the Epetra_Operator virtual base class and 
    Epetra_MultiVector class.

    This interface will ensure that any Epetra_Operator and Epetra_MultiVector will be accepted by the Anasazi
    templated solvers.

    \note The Epetra package performs double-precision arithmetic, so the use of Epetra with Anasazi will
    only provide a double-precision eigensolver.
  */

  template <> 
  class OperatorTraits < double, Epetra_MultiVector, Epetra_Operator >
  {
  public:
    
    /*! \brief This method takes the Epetra_MultiVector \c x and
      applies the Epetra_Operator \c Op to it resulting in the Epetra_MultiVector \c y.
    */    
    static void Apply ( const Epetra_Operator& Op, 
                        const Epetra_MultiVector& x, 
                        Epetra_MultiVector& y )
    { 
      TEST_FOR_EXCEPTION( Op.Apply( x, y ) != 0, OperatorError, "Error in Epetra_Operator::Apply()!" );
    }
    
  };
  
} // end of Anasazi namespace 

#endif 
// end of file ANASAZI_EPETRA_ADAPTER_HPP
