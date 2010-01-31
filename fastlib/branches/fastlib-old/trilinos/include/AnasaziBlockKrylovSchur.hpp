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

/*! \file AnasaziBlockKrylovSchur.hpp
  \brief Implementation of a block Krylov-Schur eigensolver.
*/

#ifndef ANASAZI_BLOCK_KRYLOV_SCHUR_HPP
#define ANASAZI_BLOCK_KRYLOV_SCHUR_HPP

#include "AnasaziTypes.hpp"

#include "AnasaziEigensolver.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziOperatorTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include "AnasaziOrthoManager.hpp"

#include "Teuchos_LAPACK.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"

#ifdef HAVE_TEUCHOS_COMPLEX
#if defined(HAVE_COMPLEX)
#define ANSZI_CPLX_CLASS std::complex
#elif  defined(HAVE_COMPLEX_H)
#define ANSZI_CPLX_CLASS ::complex
#endif
#endif

/*!     \class Anasazi::BlockKrylovSchur

  \brief This class implements the block Krylov-Schur iteration,
  for solving linear eigenvalue problems.

  This method is a block version of the iteration presented by G.W. Stewart 
  in "A Krylov-Schur Algorithm for Large Eigenproblems", 
  SIAM J. Matrix Anal. Appl., Vol 23(2001), No. 3, pp. 601-614.

  \ingroup anasazi_solver_framework

  \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, Heidi Thornquist
*/

namespace Anasazi {

  //! @name BlockKrylovSchur Structures 
  //@{ 

  /** \brief Structure to contain pointers to BlockKrylovSchur state variables.
   *
   * This struct is utilized by BlockKrylovSchur::initialize() and BlockKrylovSchur::getState().
   */
  template <class ScalarType, class MulVec>
  struct BlockKrylovSchurState {
    /*! \brief The current dimension of the reduction.
     *
     * This should always be equal to BlockKrylovSchur::getCurSubspaceDim()
     */
    int curDim;
    /*! \brief The current Krylov basis. */
    Teuchos::RCP<const MulVec> V;
    /*! \brief The current Hessenberg matrix. 
     *
     * The \c curDim by \c curDim leading submatrix of H is the 
     * projection of problem->getOperator() by the first \c curDim vectors in V. 
     */
    Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > H;
    /*! \brief The current Schur form reduction of the valid part of H. */
    Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > S;
    /*! \brief The current Schur vectors of the valid part of H. */
    Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > Q;
    BlockKrylovSchurState() : curDim(0), V(Teuchos::null),
                              H(Teuchos::null), S(Teuchos::null),
                              Q(Teuchos::null) {}
  };

  //@}

  //! @name BlockKrylovSchur Exceptions
  //@{ 

  /** \brief BlockKrylovSchurInitFailure is thrown when the BlockKrylovSchur solver is unable to
   * generate an initial iterate in the BlockKrylovSchur::initialize() routine. 
   *
   * This exception is thrown from the BlockKrylovSchur::initialize() method, which is
   * called by the user or from the BlockKrylovSchur::iterate() method if isInitialized()
   * == \c false.
   *
   * In the case that this exception is thrown, 
   * BlockKrylovSchur::isInitialized() will be \c false and the user will need to provide
   * a new initial iterate to the solver.
   *
   */
  class BlockKrylovSchurInitFailure : public AnasaziError {public:
    BlockKrylovSchurInitFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  /** \brief BlockKrylovSchurOrthoFailure is thrown when the orthogonalization manager is
   * unable to generate orthonormal columns from the new basis vectors.
   *
   * This exception is thrown from the BlockKrylovSchur::iterate() method.
   *
   */
  class BlockKrylovSchurOrthoFailure : public AnasaziError {public:
    BlockKrylovSchurOrthoFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};
  
  //@}


  template <class ScalarType, class MV, class OP>
  class BlockKrylovSchur : public Eigensolver<ScalarType,MV,OP> { 
  public:
    //! @name Constructor/Destructor
    //@{ 
    
    /*! \brief %BlockKrylovSchur constructor with eigenproblem, solver utilities, and parameter list of solver options.
     *
     * This constructor takes pointers required by the eigensolver, in addition
     * to a parameter list of options for the eigensolver. These options include the following:
     *   - "Block Size" - an \c int specifying the block size used by the algorithm. This can also be specified using the setBlockSize() method. Default: 1
     *   - "Num Blocks" - an \c int specifying the maximum number of blocks allocated for the solver basis. Default: 3*problem->getNEV()
     *   - "Step Size"  - an \c int specifying how many iterations are performed between computations of eigenvalues and eigenvectors.\n
     *     Note: This parameter is mandatory.
     *   - "Number of Ritz Vectors" - an \c int specifying how many Ritz vectors are computed on calls to getRitzVectors(). Default: 0
     *   - "Print Number of Ritz Values" - an \c int specifying how many Ritz values are printed on calls to currentStatus(). Default: "Block Size"
     */
    BlockKrylovSchur( const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem, 
                   const Teuchos::RCP<SortManager<ScalarType,MV,OP> > &sorter,
                   const Teuchos::RCP<OutputManager<ScalarType> > &printer,
                   const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
                   const Teuchos::RCP<OrthoManager<ScalarType,MV> > &ortho,
                   Teuchos::ParameterList &params 
                 );
    
    //! %BlockKrylovSchur destructor.
    virtual ~BlockKrylovSchur() {};
    //@}


    //! @name Solver methods
    //@{ 
    
    /*! \brief This method performs Block Krylov-Schur iterations until the status
     * test indicates the need to stop or an error occurs (in which case, an
     * exception is thrown).
     *
     * iterate() will first determine whether the solver is inintialized; if
     * not, it will call initialize() using default arguments. After
     * initialization, the solver performs Block Krylov-Schur iterations until the
     * status test evaluates as ::Passed, at which point the method returns to
     * the caller. 
     *
     * The Block Krylov-Schur iteration proceeds as follows:
     * -# The operator problem->getOperator() is applied to the newest \c blockSize vectors in the Krylov basis.
     * -# The resulting vectors are orthogonalized against the auxiliary vectors and the previous basis vectors, and made orthonormal.
     * -# The Hessenberg matrix is updated.
     * -# If we have performed \c stepSize iterations since the last update, update the Ritz values and Ritz residuals.
     *
     * The status test is queried at the beginning of the iteration.
     *
     * Possible exceptions thrown include the BlockKrylovSchurOrthoFailure.
     *
     */
    void iterate();

    /*! \brief Initialize the solver to an iterate, providing a Krylov basis and Hessenberg matrix.
     *
     * The %BlockKrylovSchur eigensolver contains a certain amount of state,
     * consisting of the current Krylov basis and the associated Hessenberg matrix.
     *
     * initialize() gives the user the opportunity to manually set these,
     * although this must be done with caution, abiding by the rules given
     * below. All notions of orthogonality and orthonormality are derived from
     * the inner product specified by the orthogonalization manager.
     *
     * \post 
     * <li>isInitialized() == \c true (see post-conditions of isInitialize())
     *
     * The user has the option of specifying any component of the state using
     * initialize(). However, these arguments are assumed to match the
     * post-conditions specified under isInitialized(). Any necessary component of the
     * state not given to initialize() will be generated.
     *
     * Note, for any pointer in \c newstate which directly points to the multivectors in 
     * the solver, the data is not copied.
     */
    void initialize(BlockKrylovSchurState<ScalarType,MV> state);

    /*! \brief Initialize the solver with the initial vectors from the eigenproblem
     *  or random data.
     */
    void initialize();

    /*! \brief Indicates whether the solver has been initialized or not.
     *
     * \return bool indicating the state of the solver.
     * \post
     * If isInitialized() == \c true:
     *    - the first getCurSubspaceDim() vectors of V are orthogonal to auxiliary vectors and have orthonormal columns
     *    - the principal Hessenberg submatrix of of H contains the Hessenberg matrix associated with V
     */
    bool isInitialized() const { return initialized_; }

    /*! \brief Get the current state of the eigensolver.
     * 
     * The data is only valid if isInitialized() == \c true. 
     *
     * \returns A BlockKrylovSchurState object containing const pointers to the current
     * solver state.
     */
    BlockKrylovSchurState<ScalarType,MV> getState() const {
      BlockKrylovSchurState<ScalarType,MV> state;
      state.curDim = curDim_;
      state.V = V_;
      state.H = H_;
      state.Q = Q_;
      state.S = schurH_;
      return state;
    }
    
    //@}


    //! @name Status methods
    //@{ 

    //! \brief Get the current iteration count.
    int getNumIters() const { return(iter_); }

    //! \brief Reset the iteration count.
    void resetNumIters() { iter_=0; }

    /*! \brief Get the Ritz vectors.
     *
     *  \return A multivector of columns not exceeding the maximum dimension of the subspace
     *  containing the Ritz vectors from the most recent call to computeRitzVectors().
     *
     *  \note To see if the returned Ritz vectors are current, call isRitzVecsCurrent().
     */
    Teuchos::RCP<const MV> getRitzVectors() { return ritzVectors_; }

    /*! \brief Get the Ritz values.
     *
     *  \return A vector of length not exceeding the maximum dimension of the subspace 
     *  containing the Ritz values from the most recent Schur form update.
     *
     *  \note To see if the returned Ritz values are current, call isRitzValsCurrent().
     */
    std::vector<Value<ScalarType> > getRitzValues() { 
      std::vector<Value<ScalarType> > ret = ritzValues_;
      ret.resize(ritzIndex_.size());
      return ret;
    }

    /*! \brief Get the Ritz index vector.
     *
     *  \return A vector of length not exceeding the maximum dimension of the subspace
     *  containing the index vector for the Ritz values and Ritz vectors, if they are computed.
     */ 
    std::vector<int> getRitzIndex() { return ritzIndex_; }

    /*! \brief Get the current residual norms.
     *
     *  \note Block Krylov-Schur cannot provide this so a zero length vector will be returned.
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getResNorms() {
      std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> ret(0);
      return ret;
    }

    /*! \brief Get the current residual 2-norms
     *
     *  \note Block Krylov-Schur cannot provide this so a zero length vector will be returned.
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getRes2Norms() {
      std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> ret(0);
      return ret;
    }

    /*! \brief Get the current Ritz residual 2-norms
     *
     *  \return A vector of length blockSize containing the 2-norms of the Ritz residuals.
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getRitzRes2Norms() {
      std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> ret = ritzResiduals_;
      ret.resize(ritzIndex_.size());
      return ret;
    }

    //@}

    //! @name Accessor routines
    //@{ 

    //! Get a constant reference to the eigenvalue problem.
    const Eigenproblem<ScalarType,MV,OP>& getProblem() const { return(*problem_); };

    /*! \brief Set the blocksize and number of blocks to be used by the
     * iterative solver in solving this eigenproblem.
     *  
     *  Changing either the block size or the number of blocks will reset the
     *  solver to an uninitialized state.
     */
    void setSize(int blockSize, int numBlocks);

    //! \brief Set the blocksize. 
    void setBlockSize(int blockSize);

    //! \brief Set the step size. 
    void setStepSize(int stepSize);

    //! \brief Set the number of Ritz vectors to compute.
    void setNumRitzVectors(int numRitzVecs);

    //! \brief Get the step size. 
    int getStepSize() const { return(stepSize_); }

    //! Get the blocksize to be used by the iterative solver in solving this eigenproblem.
    int getBlockSize() const { return(blockSize_); }

    //! \brief Get the number of Ritz vectors to compute.
    int getNumRitzVectors() const { return(numRitzVecs_); }

    /*! \brief Get the dimension of the search subspace used to generate the current eigenvectors and eigenvalues.
     *
     *  \return An integer specifying the rank of the Krylov subspace currently in use by the eigensolver. If isInitialized() == \c false, 
     *  the return is 0.
     */
    int getCurSubspaceDim() const {
      if (!initialized_) return 0;
      return curDim_;
    }

    //! Get the maximum dimension allocated for the search subspace. 
    int getMaxSubspaceDim() const { return (problem_->isHermitian()?blockSize_*numBlocks_:blockSize_*numBlocks_+1); }


    /*! \brief Set the auxiliary vectors for the solver.
     *
     *  Because the current Krylov subspace cannot be assumed
     *  orthogonal to the new auxiliary vectors, a call to setAuxVecs() will
     *  reset the solver to the uninitialized state. This happens only in the
     *  case where the new auxiliary vectors have a combined dimension of 
     *  greater than zero.
     *
     *  In order to preserve the current state, the user will need to extract
     *  it from the solver using getState(), orthogonalize it against the
     *  new auxiliary vectors, and reinitialize using initialize().
     */
    void setAuxVecs(const Teuchos::Array<Teuchos::RCP<const MV> > &auxvecs);

    //! Get the auxiliary vectors for the solver.
    Teuchos::Array<Teuchos::RCP<const MV> > getAuxVecs() const {return auxVecs_;}

    //@}

    //! @name Output methods
    //@{ 
    
    //! This method requests that the solver print out its current status to screen.
    void currentStatus(std::ostream &os);

    //@}

    //! @name Block-Krylov Schur status routines
    //@{
    
    //! Get the status of the Ritz vectors currently stored in the eigensolver.
    bool isRitzVecsCurrent() const { return ritzVecsCurrent_; }

    //! Get the status of the Ritz values currently stored in the eigensolver.
    bool isRitzValsCurrent() const { return ritzValsCurrent_; }
    
    //! Get the status of the Schur form currently stored in the eigensolver.
    bool isSchurCurrent() const { return schurCurrent_; }
    
    //@}

    //! @name Block-Krylov Schur compute routines
    //@{
    
    //! Compute the Ritz vectors using the current Krylov factorization.
    void computeRitzVectors();

    //! Compute the Ritz values using the current Krylov factorization.
    void computeRitzValues();
    
    //! Compute the Schur form of the projected eigenproblem from the current Krylov factorization.
    void computeSchurForm( const bool sort = true );

    //@}

  private:
    //
    // Convenience typedefs
    //
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename SCT::magnitudeType MagnitudeType;
    typedef typename std::vector<ScalarType>::iterator STiter;
    typedef typename std::vector<MagnitudeType>::iterator MTiter;
    const MagnitudeType MT_ONE;  
    const MagnitudeType MT_ZERO; 
    const MagnitudeType NANVAL;
    const ScalarType ST_ONE;
    const ScalarType ST_ZERO;
    //
    // Internal structs
    //
    struct CheckList {
      bool checkV;
      bool checkArn;
      bool checkAux;
      CheckList() : checkV(false), checkArn(false), checkAux(false) {};
    };
    //
    // Internal methods
    //
    std::string accuracyCheck(const CheckList &chk, const std::string &where) const;
    void sortSchurForm( Teuchos::SerialDenseMatrix<int,ScalarType>& H,
                        Teuchos::SerialDenseMatrix<int,ScalarType>& Q,
                        std::vector<int>& order );
    //
    // Classes inputed through constructor that define the eigenproblem to be solved.
    //
    const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> >     problem_;
    const Teuchos::RCP<SortManager<ScalarType,MV,OP> >      sm_;
    const Teuchos::RCP<OutputManager<ScalarType> >          om_;
    const Teuchos::RCP<StatusTest<ScalarType,MV,OP> >       tester_;
    const Teuchos::RCP<OrthoManager<ScalarType,MV> >        orthman_;
    //
    // Information obtained from the eigenproblem
    //
    Teuchos::RCP<const OP> Op_;
    //
    // Internal timers
    //
    Teuchos::RCP<Teuchos::Time> timerOp_, timerSortRitzVal_,
                                        timerCompSF_, timerSortSF_,
                                        timerCompRitzVec_, timerOrtho_;
    //
    // Counters
    //
    int count_ApplyOp_;

    //
    // Algorithmic parameters.
    //
    // blockSize_ is the solver block size; it controls the number of eigenvectors that 
    // we compute, the number of residual vectors that we compute, and therefore the number
    // of vectors added to the basis on each iteration.
    int blockSize_;
    // numBlocks_ is the size of the allocated space for the Krylov basis, in blocks.
    int numBlocks_; 
    // stepSize_ dictates how many iterations are performed before eigenvectors and eigenvalues
    // are computed again
    int stepSize_;
    
    // 
    // Current solver state
    //
    // initialized_ specifies that the basis vectors have been initialized and the iterate() routine
    // is capable of running; _initialize is controlled  by the initialize() member method
    // For the implications of the state of initialized_, please see documentation for initialize()
    bool initialized_;
    //
    // curDim_ reflects how much of the current basis is valid 
    // NOTE: for Hermitian, 0 <= curDim_ <= blockSize_*numBlocks_
    //   for non-Hermitian, 0 <= curDim_ <= blockSize_*numBlocks_ + 1
    // this also tells us how many of the values in _theta are valid Ritz values
    int curDim_;
    //
    // State Multivecs
    Teuchos::RCP<MV> ritzVectors_, V_;
    int numRitzVecs_;
    //
    // Projected matrices
    // H_ : Projected matrix from the Krylov-Schur factorization AV = VH + FB^T
    //
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > H_;
    // 
    // Schur form of Projected matrices (these are only updated when the Ritz values/vectors are updated).
    // schurH_: Schur form reduction of H
    // Q_: Schur vectors of H
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > schurH_;
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > Q_;
    // 
    // Auxiliary vectors
    Teuchos::Array<Teuchos::RCP<const MV> > auxVecs_;
    int numAuxVecs_;
    //
    // Number of iterations that have been performed.
    int iter_;
    //
    // State flags
    bool ritzVecsCurrent_, ritzValsCurrent_, schurCurrent_;
    // 
    // Current eigenvalues, residual norms
    std::vector<Value<ScalarType> > ritzValues_;
    std::vector<MagnitudeType> ritzResiduals_;
    // 
    // Current index vector for Ritz values and vectors
    std::vector<int> ritzIndex_;  // computed by BKS
    std::vector<int> ritzOrder_;  // returned from sort manager
    //
    // Number of Ritz pairs to be printed upon output, if possible
    int numRitzPrint_;
  };


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Helper function for correctly storing the Ritz values when the eigenproblem is non-Hermitian
  // This allows us to use template specialization to compute the right index vector and correctly
  // handle complex-conjugate pairs.
  template<class ScalarType>
  void sortRitzValues( const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>& rRV,
                       const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>& iRV,
                       std::vector<Value<ScalarType> >* RV, std::vector<int>* RO, std::vector<int>* RI )
  {
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    MagnitudeType MT_ZERO = Teuchos::ScalarTraits<MagnitudeType>::zero();

    int curDim = (int)rRV.size();
    int i = 0;

    // Clear the current index.
    RI->clear();

    // Place the Ritz values from rRV and iRV into the RV container.
    while( i < curDim ) {
      if ( iRV[i] != MT_ZERO ) {
        //
        // We will have this situation for real-valued, non-Hermitian matrices.
        (*RV)[i].set(rRV[i], iRV[i]);
        (*RV)[i+1].set(rRV[i+1], iRV[i+1]);
        
        // Make sure that complex conjugate pairs have their positive imaginary part first.
        if ( (*RV)[i].imagpart < MT_ZERO ) {
          // The negative imaginary part is first, so swap the order of the ritzValues and ritzOrders.
          Anasazi::Value<ScalarType> tmp_ritz( (*RV)[i] );
          (*RV)[i] = (*RV)[i+1];
          (*RV)[i+1] = tmp_ritz;
          
          int tmp_order = (*RO)[i];
          (*RO)[i] = (*RO)[i+1];
          (*RO)[i+1] = tmp_order;
          
        }
        RI->push_back(1); RI->push_back(-1);
        i = i+2;
      } else {
        //
        // The Ritz value is not complex.
        (*RV)[i].set(rRV[i], MT_ZERO);
        RI->push_back(0);
        i++;
      }
    }
  }
  
#ifdef HAVE_TEUCHOS_COMPLEX
  // Template specialization for the complex scalar type.
  void sortRitzValues( const std::vector<double>& rRV, 
                       const std::vector<double>& iRV,
                       std::vector<Value<ANSZI_CPLX_CLASS<double> > >* RV, 
                       std::vector<int>* RO, std::vector<int>* RI )
  {
    int curDim = (int)rRV.size();
    int i = 0;

    // Clear the current index.
    RI->clear();

    // Place the Ritz values from rRV and iRV into the RV container.
    while( i < curDim ) {
      (*RV)[i].set(rRV[i], iRV[i]);
      RI->push_back(0);
      i++;
    }    
  }
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Helper function for correctly scaling the eigenvectors of the projected eigenproblem.
  // This allows us to use template specialization to compute the right scaling so the
  // Ritz residuals are correct.
  template<class ScalarType>
  void scaleRitzVectors( const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>& iRV,
                         Teuchos::SerialDenseMatrix<int, ScalarType>* S )
  {
    ScalarType ST_ONE = Teuchos::ScalarTraits<ScalarType>::one();
    
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    MagnitudeType MT_ZERO = Teuchos::ScalarTraits<MagnitudeType>::zero();

    Teuchos::LAPACK<int,MagnitudeType> lapack_mag;
    Teuchos::BLAS<int,ScalarType> blas;
    
    int i = 0, curDim = S->numRows();
    ScalarType temp;
    ScalarType* s_ptr = S->values();
    while( i < curDim ) {
      if ( iRV[i] != MT_ZERO ) {
        temp = lapack_mag.LAPY2( blas.NRM2( curDim, s_ptr+i*curDim, 1 ), 
                                 blas.NRM2( curDim, s_ptr+(i+1)*curDim, 1 ) );
        blas.SCAL( curDim, ST_ONE/temp, s_ptr+i*curDim, 1 );
        blas.SCAL( curDim, ST_ONE/temp, s_ptr+(i+1)*curDim, 1 );
        i = i+2;
      } else {
        temp = blas.NRM2( curDim, s_ptr+i*curDim, 1 );
        blas.SCAL( curDim, ST_ONE/temp, s_ptr+i*curDim, 1 );
        i++;
      }
    }
  }

#ifdef HAVE_TEUCHOS_COMPLEX
  // Template specialization for the complex scalar type.
  void scaleRitzVectors( const std::vector<double>& iRV,
                         Teuchos::SerialDenseMatrix<int, ANSZI_CPLX_CLASS<double> >* S )
  {
    typedef ANSZI_CPLX_CLASS<double> ST;
    ST ST_ONE = Teuchos::ScalarTraits<ST>::one();
    
    Teuchos::BLAS<int,ST> blas;
    
    int i = 0, curDim = S->numRows();
    ST temp;
    ST* s_ptr = S->values();
    while( i < curDim ) {
      temp = blas.NRM2( curDim, s_ptr+i*curDim, 1 );
      blas.SCAL( curDim, ST_ONE/temp, s_ptr+i*curDim, 1 );
      i++;
    }
  }
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Helper function for correctly computing the Ritz residuals of the projected eigenproblem.
  // This allows us to use template specialization to ensure the Ritz residuals are correct.
  template<class ScalarType>
  void computeRitzResiduals( const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>& iRV,
                             const Teuchos::SerialDenseMatrix<int, ScalarType>& S,
                             std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType>* RR
                             )
  {
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    MagnitudeType MT_ZERO = Teuchos::ScalarTraits<MagnitudeType>::zero();
    
    Teuchos::LAPACK<int,MagnitudeType> lapack_mag;
    Teuchos::BLAS<int,ScalarType> blas;
    
    int i = 0;
    int s_stride = S.stride();
    int s_rows = S.numRows();
    int s_cols = S.numCols();
    ScalarType* s_ptr = S.values();

    while( i < s_cols ) {
      if ( iRV[i] != MT_ZERO ) {
        (*RR)[i] = lapack_mag.LAPY2( blas.NRM2(s_rows, s_ptr + i*s_stride, 1),
                                     blas.NRM2(s_rows, s_ptr + (i+1)*s_stride, 1) );
        (*RR)[i+1] = (*RR)[i];
        i = i+2;
      } else {
        (*RR)[i] = blas.NRM2(s_rows, s_ptr + i*s_stride, 1);
        i++;
      }
    }          
  }

#ifdef HAVE_TEUCHOS_COMPLEX
  // Template specialization for the complex scalar type.
  void computeRitzResiduals( const std::vector<double>& iRV,
                             const Teuchos::SerialDenseMatrix<int, ANSZI_CPLX_CLASS<double> >& S,
                             std::vector<double>* RR
                             )
  {
    Teuchos::BLAS<int,ANSZI_CPLX_CLASS<double> > blas;
    
    int s_stride = S.stride();
    int s_rows = S.numRows();
    int s_cols = S.numCols();
    ANSZI_CPLX_CLASS<double>* s_ptr = S.values();

    for (int i=0; i<s_cols; ++i ) {
      (*RR)[i] = blas.NRM2(s_rows, s_ptr + i*s_stride, 1);
    }
  }          
  
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor
  template <class ScalarType, class MV, class OP>
  BlockKrylovSchur<ScalarType,MV,OP>::BlockKrylovSchur(
        const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem, 
        const Teuchos::RCP<SortManager<ScalarType,MV,OP> > &sorter,
        const Teuchos::RCP<OutputManager<ScalarType> > &printer,
        const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
        const Teuchos::RCP<OrthoManager<ScalarType,MV> > &ortho,
        Teuchos::ParameterList &params
        ) :
    MT_ONE(Teuchos::ScalarTraits<MagnitudeType>::one()),
    MT_ZERO(Teuchos::ScalarTraits<MagnitudeType>::zero()),
    NANVAL(Teuchos::ScalarTraits<MagnitudeType>::nan()),
    ST_ONE(Teuchos::ScalarTraits<ScalarType>::one()),
    ST_ZERO(Teuchos::ScalarTraits<ScalarType>::zero()),
    // problem, tools
    problem_(problem), 
    sm_(sorter),
    om_(printer),
    tester_(tester),
    orthman_(ortho),
    // timers, counters
    timerOp_(Teuchos::TimeMonitor::getNewTimer("Operation Op*x")),
    timerSortRitzVal_(Teuchos::TimeMonitor::getNewTimer("Sorting Ritz values")),
    timerCompSF_(Teuchos::TimeMonitor::getNewTimer("Computing Schur form")),
    timerSortSF_(Teuchos::TimeMonitor::getNewTimer("Sorting Schur form")),
    timerCompRitzVec_(Teuchos::TimeMonitor::getNewTimer("Computing Ritz vectors")),
    timerOrtho_(Teuchos::TimeMonitor::getNewTimer("Orthogonalization")),
    count_ApplyOp_(0),
    // internal data
    blockSize_(0),
    numBlocks_(0),
    stepSize_(0),
    initialized_(false),
    curDim_(0),
    numRitzVecs_(0),
    auxVecs_( Teuchos::Array<Teuchos::RCP<const MV> >(0) ), 
    numAuxVecs_(0),
    iter_(0),
    ritzVecsCurrent_(false),
    ritzValsCurrent_(false),
    schurCurrent_(false),
    numRitzPrint_(0)
  {     
    TEST_FOR_EXCEPTION(problem_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: user specified null problem pointer.");
    TEST_FOR_EXCEPTION(problem_->isProblemSet() == false, std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: user specified problem is not set.");
    TEST_FOR_EXCEPTION(sorter == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: user specified null sort manager pointer.");
    TEST_FOR_EXCEPTION(printer == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: user specified null output manager pointer.");
    TEST_FOR_EXCEPTION(tester == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: user specified null status test pointer.");
    TEST_FOR_EXCEPTION(ortho == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: user specified null ortho manager pointer.");

    // Get problem operator
    Op_ = problem_->getOperator();

    // get the step size
    TEST_FOR_EXCEPTION(!params.isParameter("Step Size"), std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::constructor: mandatory parameter 'Step Size' is not specified.");
    int ss = params.get("Step Size",numBlocks_);
    setStepSize(ss);

    // set the block size and allocate data
    int bs = params.get("Block Size", 1);
    int nb = params.get("Num Blocks", 3*problem_->getNEV());
    setSize(bs,nb);

    // get the number of Ritz vectors to compute and allocate data.
    // --> if this parameter is not specified in the parameter list, then it's assumed that no Ritz vectors will be computed.
    int numRitzVecs = params.get("Number of Ritz Vectors", 0);
    setNumRitzVectors( numRitzVecs );

    // get the number of Ritz values to print out when currentStatus is called.
    numRitzPrint_ = params.get("Print Number of Ritz Values", bs);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the block size
  // This simply calls setSize(), modifying the block size while retaining the number of blocks.
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::setBlockSize (int blockSize) 
  {
    setSize(blockSize,numBlocks_);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the step size.
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::setStepSize (int stepSize)
  {
    TEST_FOR_EXCEPTION(stepSize <= 0, std::invalid_argument, "Anasazi::BlockKrylovSchur::setStepSize(): new step size must be positive and non-zero.");
    stepSize_ = stepSize;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the number of Ritz vectors to compute.
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::setNumRitzVectors (int numRitzVecs)
  {
    // This routine only allocates space; it doesn't not perform any computation
    // any change in size will invalidate the state of the solver.

    TEST_FOR_EXCEPTION(numRitzVecs < 0, std::invalid_argument, "Anasazi::BlockKrylovSchur::setNumRitzVectors(): number of Ritz vectors to compute must be positive.");

    // Check to see if the number of requested Ritz vectors has changed.
    if (numRitzVecs != numRitzVecs_) {
      if (numRitzVecs) {
        ritzVectors_ = Teuchos::null;
        ritzVectors_ = MVT::Clone(*V_, numRitzVecs);
      } else {
        ritzVectors_ = Teuchos::null;
      }
      numRitzVecs_ = numRitzVecs;
      ritzVecsCurrent_ = false;
    }      
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the block size and make necessary adjustments.
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::setSize (int blockSize, int numBlocks) 
  {
    // This routine only allocates space; it doesn't not perform any computation
    // any change in size will invalidate the state of the solver.

    TEST_FOR_EXCEPTION(numBlocks <= 0 || blockSize <= 0, std::invalid_argument, "Anasazi::BlockKrylovSchur::setSize was passed a non-positive argument.");
    TEST_FOR_EXCEPTION(numBlocks < 3, std::invalid_argument, "Anasazi::BlockKrylovSchur::setSize(): numBlocks must be at least three.");
    if (blockSize == blockSize_ && numBlocks == numBlocks_) {
      // do nothing
      return;
    }

    blockSize_ = blockSize;
    numBlocks_ = numBlocks;

    Teuchos::RCP<const MV> tmp;
    // grab some Multivector to Clone
    // in practice, getInitVec() should always provide this, but it is possible to use a 
    // Eigenproblem with nothing in getInitVec() by manually initializing with initialize(); 
    // in case of that strange scenario, we will try to Clone from V_; first resort to getInitVec(), 
    // because we would like to clear the storage associated with V_ so we have room for the new V_
    if (problem_->getInitVec() != Teuchos::null) {
      tmp = problem_->getInitVec();
    }
    else {
      tmp = V_;
      TEST_FOR_EXCEPTION(tmp == Teuchos::null,std::invalid_argument,
          "Anasazi::BlockKrylovSchur::setSize(): eigenproblem did not specify initial vectors to clone from.");
    }


    //////////////////////////////////
    // blockSize*numBlocks dependent
    //
    int newsd;
    if (problem_->isHermitian()) {
      newsd = blockSize_*numBlocks_;
    } else {
      newsd = blockSize_*numBlocks_+1;
    }
    // check that new size is valid
    TEST_FOR_EXCEPTION(newsd > MVT::GetVecLength(*tmp),std::invalid_argument,
        "Anasazi::BlockKrylovSchur::setSize(): maximum basis size is larger than problem dimension.");

    ritzValues_.resize(newsd);
    ritzResiduals_.resize(newsd,MT_ONE);
    ritzOrder_.resize(newsd);
    V_ = Teuchos::null;
    V_ = MVT::Clone(*tmp,newsd+blockSize_);
    H_ = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(newsd+blockSize_,newsd) );
    Q_ = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(newsd,newsd) );

    initialized_ = false;
    curDim_ = 0;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the auxiliary vectors
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::setAuxVecs(const Teuchos::Array<Teuchos::RCP<const MV> > &auxvecs) {
    typedef typename Teuchos::Array<Teuchos::RCP<const MV> >::iterator tarcpmv;

    // set new auxiliary vectors
    auxVecs_ = auxvecs;
    
    if (om_->isVerbosity( Debug ) ) {
      // Check almost everything here
      CheckList chk;
      chk.checkAux = true;
      om_->print( Debug, accuracyCheck(chk, ": in setAuxVecs()") );
    }

    numAuxVecs_ = 0;
    for (tarcpmv i=auxVecs_.begin(); i != auxVecs_.end(); i++) {
      numAuxVecs_ += MVT::GetNumberVecs(**i);
    }
    
    // If the solver has been initialized, X and P are not necessarily orthogonal to new auxiliary vectors
    if (numAuxVecs_ > 0 && initialized_) {
      initialized_ = false;
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /* Initialize the state of the solver
   * 
   * POST-CONDITIONS:
   *
   * V_ is orthonormal, orthogonal to auxVecs_, for first curDim_ vectors
   *
   */

  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::initialize(BlockKrylovSchurState<ScalarType,MV> newstate)
  {
    // NOTE: memory has been allocated by setBlockSize(). Use SetBlock below; do not Clone

    std::vector<int> bsind(blockSize_);
    for (int i=0; i<blockSize_; i++) bsind[i] = i;

    // in BlockKrylovSchur, V and H are required.  
    // if either doesn't exist, then we will start with the initial vector.
    //
    // inconsistent multivectors widths and lengths will not be tolerated, and
    // will be treated with exceptions.
    //
    std::string errstr("Anasazi::BlockKrylovSchur::initialize(): specified multivectors must have a consistent length and width.");

    // set up V,H: if the user doesn't specify both of these these, 
    // we will start over with the initial vector.
    if (newstate.V != Teuchos::null && newstate.H != Teuchos::null) {

      // initialize V_,H_, and curDim_

      TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.V) != MVT::GetVecLength(*V_),
                          std::invalid_argument, errstr );
      if (newstate.V != V_) {
        TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.V) < blockSize_,
            std::invalid_argument, errstr );
        TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.V) > getMaxSubspaceDim(),
            std::invalid_argument, errstr );
      }
      TEST_FOR_EXCEPTION( newstate.curDim > getMaxSubspaceDim(),
                          std::invalid_argument, errstr );

      curDim_ = newstate.curDim;
      int lclDim = MVT::GetNumberVecs(*newstate.V);

      // check size of H
      TEST_FOR_EXCEPTION(newstate.H->numRows() < curDim_ || newstate.H->numCols() < curDim_, std::invalid_argument, errstr);
      
      if (curDim_ == 0 && lclDim > blockSize_) {
        om_->stream(Warnings) << "Anasazi::BlockKrylovSchur::initialize(): the solver was initialized with a kernel of " << lclDim << std::endl
                                  << "The block size however is only " << blockSize_ << std::endl
                                  << "The last " << lclDim - blockSize_ << " vectors of the kernel will be overwritten on the first call to iterate()." << std::endl;
      }


      // copy basis vectors from newstate into V
      if (newstate.V != V_) {
        std::vector<int> nevind(lclDim);
        for (int i=0; i<lclDim; i++) nevind[i] = i;
        MVT::SetBlock(*newstate.V,nevind,*V_);
      }

      // put data into H_, make sure old information is not still hanging around.
      if (newstate.H != H_) {
        H_->putScalar( ST_ZERO );
        Teuchos::SerialDenseMatrix<int,ScalarType> newH(Teuchos::View,*newstate.H,curDim_+blockSize_,curDim_);
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > lclH;
        lclH = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::View,*H_,curDim_+blockSize_,curDim_) );
        lclH->assign(newH);

        // done with local pointers
        lclH = Teuchos::null;
      }

    }
    else {
      // user did not specify a basis V
      // get vectors from problem or generate something, projectAndNormalize, call initialize() recursively
      Teuchos::RCP<const MV> ivec = problem_->getInitVec();
      TEST_FOR_EXCEPTION(ivec == Teuchos::null,std::invalid_argument,
                         "Anasazi::BlockKrylovSchur::initialize(): eigenproblem did not specify initial vectors to clone from.");

      int lclDim = MVT::GetNumberVecs(*ivec);
      bool userand = false;
      if (lclDim < blockSize_) {
        // we need at least blockSize_ vectors
        // use a random multivec
        userand = true;
      }
              
      if (userand) {
        // make an index
        std::vector<int> dimind2(lclDim);
        for (int i=0; i<lclDim; i++) { dimind2[i] = i; }

        // alloc newV as a view of the first block of V
        Teuchos::RCP<MV> newV1 = MVT::CloneView(*V_,dimind2);

        // copy the initial vectors into the first lclDim vectors of V
        MVT::SetBlock(*ivec,dimind2,*newV1);

        // resize / reinitialize the index vector        
        dimind2.resize(blockSize_-lclDim);
        for (int i=0; i<blockSize_-lclDim; i++) { dimind2[i] = lclDim + i; }

        // initialize the rest of the vectors with random vectors
        Teuchos::RCP<MV> newV2 = MVT::CloneView(*V_,dimind2);
        MVT::MvRandom(*newV2);
      }
      else {
        // alloc newV as a view of the first block of V
        Teuchos::RCP<MV> newV1 = MVT::CloneView(*V_,bsind);
       
        // get a view of the first block of initial vectors
        Teuchos::RCP<const MV> ivecV = MVT::CloneView(*ivec,bsind);
 
        // assign ivec to first part of newV
        MVT::SetBlock(*ivecV,bsind,*newV1);
      }

      // get pointer into first block of V
      Teuchos::RCP<MV> newV = MVT::CloneView(*V_,bsind);

      // remove auxVecs from newV and normalize newV
      if (auxVecs_.size() > 0) {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );
        
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > dummy;
        int rank = orthman_->projectAndNormalize(*newV,dummy,Teuchos::null,auxVecs_);
        TEST_FOR_EXCEPTION( rank != blockSize_,BlockKrylovSchurInitFailure,
                            "Anasazi::BlockKrylovSchur::initialize(): couldn't generate initial basis of full rank." );
      }
      else {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );

        int rank = orthman_->normalize(*newV,Teuchos::null);
        TEST_FOR_EXCEPTION( rank != blockSize_,BlockKrylovSchurInitFailure,
                            "Anasazi::BlockKrylovSchur::initialize(): couldn't generate initial basis of full rank." );
      }

      // set curDim
      curDim_ = 0;

      // clear pointer
      newV = Teuchos::null;
    }

    // The Ritz vectors/values and Schur form are no longer current.
    ritzVecsCurrent_ = false;
    ritzValsCurrent_ = false;
    schurCurrent_ = false;

    // the solver is initialized
    initialized_ = true;

    if (om_->isVerbosity( Debug ) ) {
      // Check almost everything here
      CheckList chk;
      chk.checkV = true;
      chk.checkArn = true;
      chk.checkAux = true;
      om_->print( Debug, accuracyCheck(chk, ": after initialize()") );
    }

    // Print information on current status
    if (om_->isVerbosity(Debug)) {
      currentStatus( om_->stream(Debug) );
    }
    else if (om_->isVerbosity(IterationDetails)) {
      currentStatus( om_->stream(IterationDetails) );
    }
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // initialize the solver with default state
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::initialize()
  {
    BlockKrylovSchurState<ScalarType,MV> empty;
    initialize(empty);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform BlockKrylovSchur iterations until the StatusTest tells us to stop.
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::iterate()
  {
    //
    // Allocate/initialize data structures
    //
    if (initialized_ == false) {
      initialize();
    }

    // Compute the current search dimension. 
    // If the problem is non-Hermitian and the blocksize is one, let the solver use the extra vector.
    int searchDim = blockSize_*numBlocks_;
    if (problem_->isHermitian() == false) {
      searchDim++;
    } 

    ////////////////////////////////////////////////////////////////
    // iterate until the status test tells us to stop.
    //
    // also break if our basis is full
    //
    while (tester_->checkStatus(this) != Passed && curDim_+blockSize_ <= searchDim) {

      iter_++;

      // F can be found at the curDim_ block, but the next block is at curDim_ + blockSize_.
      int lclDim = curDim_ + blockSize_; 

      // Get the current part of the basis.
      std::vector<int> curind(blockSize_);
      for (int i=0; i<blockSize_; i++) { curind[i] = lclDim + i; }
      Teuchos::RCP<MV> Vnext = MVT::CloneView(*V_,curind);

      // Get a view of the previous vectors
      // this is used for orthogonalization and for computing V^H K H
      for (int i=0; i<blockSize_; i++) { curind[i] = curDim_ + i; }
      Teuchos::RCP<MV> Vprev = MVT::CloneView(*V_,curind);

      // Compute the next vector in the Krylov basis:  Vnext = Op*Vprev
      {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply(*Op_,*Vprev,*Vnext);
        count_ApplyOp_ += blockSize_;
      }
      Vprev = Teuchos::null;
      
      // Remove all previous Krylov-Schur basis vectors and auxVecs from Vnext
      {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );
        
        // Get a view of all the previous vectors
        std::vector<int> prevind(lclDim);
        for (int i=0; i<lclDim; i++) { prevind[i] = i; }
        Vprev = MVT::CloneView(*V_,prevind);
        Teuchos::Array<Teuchos::RCP<const MV> > AVprev(1, Vprev);
        
        // Get a view of the part of the Hessenberg matrix needed to hold the ortho coeffs.
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> >
          subH = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>
                               ( Teuchos::View,*H_,lclDim,blockSize_,0,curDim_ ) );
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > AsubH;
        AsubH.append( subH );
        
        // Add the auxiliary vectors to the current basis vectors if any exist
        if (auxVecs_.size() > 0) {
          for (unsigned int i=0; i<auxVecs_.size(); i++) {
            AVprev.append( auxVecs_[i] );
            AsubH.append( Teuchos::null );
          }
        }
        
        // Get a view of the part of the Hessenberg matrix needed to hold the norm coeffs.
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> >
          subR = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>
                               ( Teuchos::View,*H_,blockSize_,blockSize_,lclDim,curDim_ ) );
        int rank = orthman_->projectAndNormalize(*Vnext,AsubH,subR,AVprev);
        TEST_FOR_EXCEPTION(rank != blockSize_,BlockKrylovSchurOrthoFailure,
                           "Anasazi::BlockKrylovSchur::iterate(): couldn't generate basis of full rank.");
      }
      //
      // V has been extended, and H has been extended. 
      //
      // Update basis dim and release all pointers.
      Vnext = Teuchos::null;
      curDim_ += blockSize_;
      // The Ritz vectors/values and Schur form are no longer current.
      ritzVecsCurrent_ = false;
      ritzValsCurrent_ = false;
      schurCurrent_ = false;
      //
      // Update Ritz values and residuals if needed
      if (!(iter_%stepSize_)) {
        computeRitzValues();
      }
      
      // When required, monitor some orthogonalities
      if (om_->isVerbosity( Debug ) ) {
        // Check almost everything here
        CheckList chk;
        chk.checkV = true;
        chk.checkArn = true;
        om_->print( Debug, accuracyCheck(chk, ": after local update") );
      }
      else if (om_->isVerbosity( OrthoDetails ) ) {
        CheckList chk;
        chk.checkV = true;
        om_->print( OrthoDetails, accuracyCheck(chk, ": after local update") );
      }
      
      // Print information on current iteration
      if (om_->isVerbosity(Debug)) {
        currentStatus( om_->stream(Debug) );
      }
      else if (om_->isVerbosity(IterationDetails)) {
        currentStatus( om_->stream(IterationDetails) );
      }
      
    } // end while (statusTest == false)
    
  } // end of iterate()


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Check accuracy, orthogonality, and other debugging stuff
  // 
  // bools specify which tests we want to run (instead of running more than we actually care about)
  //
  // checkV : V orthonormal
  //          orthogonal to auxvecs
  // checkAux: check that auxiliary vectors are actually orthonormal
  //
  // checkArn: check the Arnoldi factorization
  //
  // NOTE:  This method needs to check the current dimension of the subspace, since it is possible to
  //        call this method when curDim_ = 0 (after initialization).
  template <class ScalarType, class MV, class OP>
  std::string BlockKrylovSchur<ScalarType,MV,OP>::accuracyCheck( const CheckList &chk, const std::string &where ) const 
  {
    std::stringstream os;
    os.precision(2);
    os.setf(std::ios::scientific, std::ios::floatfield);
    MagnitudeType tmp;

    os << " Debugging checks: iteration " << iter_ << where << std::endl;

    // index vectors for V and F
    std::vector<int> lclind(curDim_);
    for (int i=0; i<curDim_; i++) lclind[i] = i;
    std::vector<int> bsind(blockSize_);
    for (int i=0; i<blockSize_; i++) { bsind[i] = curDim_ + i; }
    
    Teuchos::RCP<MV> lclV,lclF,lclAV;
    if (curDim_)
      lclV = MVT::CloneView(*V_,lclind);
    lclF = MVT::CloneView(*V_,bsind);

    if (chk.checkV) {
      if (curDim_) {
        tmp = orthman_->orthonormError(*lclV);
        os << " >> Error in V^H M V == I  : " << tmp << std::endl;
      }
      tmp = orthman_->orthonormError(*lclF);
      os << " >> Error in F^H M F == I  : " << tmp << std::endl;
      if (curDim_) {
        tmp = orthman_->orthogError(*lclV,*lclF);
        os << " >> Error in V^H M F == 0  : " << tmp << std::endl;
      }
      for (unsigned int i=0; i<auxVecs_.size(); i++) {
        if (curDim_) {
          tmp = orthman_->orthogError(*lclV,*auxVecs_[i]);
          os << " >> Error in V^H M Aux[" << i << "] == 0 : " << tmp << std::endl;
        }
        tmp = orthman_->orthogError(*lclF,*auxVecs_[i]);
        os << " >> Error in F^H M Aux[" << i << "] == 0 : " << tmp << std::endl;
      }
    }
    
    if (chk.checkArn) {

      if (curDim_) {
        // Compute AV      
        Teuchos::RCP<MV> lclAV = MVT::Clone(*V_,curDim_);
        {
          Teuchos::TimeMonitor lcltimer( *timerOp_ );
          OPT::Apply(*Op_,*lclV,*lclAV);
        }
        
        // Compute AV - VH
        Teuchos::SerialDenseMatrix<int,ScalarType> subH(Teuchos::View,*H_,curDim_,curDim_);
        MVT::MvTimesMatAddMv( -ST_ONE, *lclV, subH, ST_ONE, *lclAV );
        
        // Compute FB_k^T - (AV-VH)
        Teuchos::SerialDenseMatrix<int,ScalarType> curB(Teuchos::View,*H_,
                                                        blockSize_,curDim_, curDim_ );
        MVT::MvTimesMatAddMv( -ST_ONE, *lclF, curB, ST_ONE, *lclAV );
        
        // Compute || FE_k^T - (AV-VH) ||
        std::vector<MagnitudeType> arnNorms( curDim_ );
        orthman_->norm( *lclAV, &arnNorms );
        
        for (int i=0; i<curDim_; i++) {        
        os << " >> Error in Krylov-Schur factorization (R = AV-VS-FB^H), ||R[" << i << "]|| : " << arnNorms[i] << std::endl;
        }
      }
    }

    if (chk.checkAux) {
      for (unsigned int i=0; i<auxVecs_.size(); i++) {
        tmp = orthman_->orthonormError(*auxVecs_[i]);
        os << " >> Error in Aux[" << i << "]^H M Aux[" << i << "] == I : " << tmp << std::endl;
        for (unsigned int j=i+1; j<auxVecs_.size(); j++) {
          tmp = orthman_->orthogError(*auxVecs_[i],*auxVecs_[j]);
          os << " >> Error in Aux[" << i << "]^H M Aux[" << j << "] == 0 : " << tmp << std::endl;
        }
      }
    }

    os << std::endl;

    return os.str();
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /* Get the current approximate eigenvalues, i.e. Ritz values.
   * 
   * POST-CONDITIONS:
   *
   * ritzValues_ contains Ritz w.r.t. V, H
   * Q_ contains the Schur vectors w.r.t. H
   * schurH_ contains the Schur matrix w.r.t. H
   * ritzOrder_ contains the current ordering from sort manager
   */

  template <class ScalarType, class MV, class OP>  
  void BlockKrylovSchur<ScalarType,MV,OP>::computeRitzValues()
  {
    // Can only call this if the solver is initialized
    if (initialized_) {

      // This just updates the Ritz values and residuals.
      // --> ritzValsCurrent_ will be set to 'true' by this method.
      if (!ritzValsCurrent_) {
        // Compute the current Ritz values, through computing the Schur form
        //   without updating the current projection matrix or sorting the Schur form.
        computeSchurForm( false );
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /* Get the current approximate eigenvectors, i.e. Ritz vectors.
   * 
   * POST-CONDITIONS:
   *
   * ritzValues_ contains Ritz w.r.t. V, H
   * ritzVectors_ is first blockSize_ Ritz vectors w.r.t. V, H
   * Q_ contains the Schur vectors w.r.t. H
   * schurH_ contains the Schur matrix w.r.t. H
   * ritzOrder_ contains the current ordering from sort manager
   */

  template <class ScalarType, class MV, class OP>  
  void BlockKrylovSchur<ScalarType,MV,OP>::computeRitzVectors()
  {
    Teuchos::TimeMonitor LocalTimer(*timerCompRitzVec_);

    TEST_FOR_EXCEPTION(numRitzVecs_==0, std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::computeRitzVectors(): no Ritz vectors were required from this solver.");

    TEST_FOR_EXCEPTION(curDim_ < numRitzVecs_, std::invalid_argument,
                       "Anasazi::BlockKrylovSchur::computeRitzVectors(): the current subspace is not large enough to compute the number of requested Ritz vectors.");


    // Check to see if the current subspace dimension is non-trivial and the solver is initialized
    if (curDim_ && initialized_) {

      // Check to see if the Ritz vectors are current.
      if (!ritzVecsCurrent_) {
        
        // Check to see if the Schur factorization of H (schurH_, Q) is current and sorted.
        if (!schurCurrent_) {
          // Compute the Schur factorization of the current H, which will not directly change H,
          // the factorization will be sorted and placed in (schurH_, Q)
          computeSchurForm( true );
        }
        
        // After the Schur form is computed, then the Ritz values are current.
        // Thus, I can check the Ritz index vector to see if I have enough space for the Ritz vectors requested.
        TEST_FOR_EXCEPTION(ritzIndex_[numRitzVecs_-1]==1, std::logic_error,
                           "Anasazi::BlockKrylovSchur::computeRitzVectors(): the number of required Ritz vectors splits a complex conjugate pair.");

        Teuchos::LAPACK<int,ScalarType> lapack;
        Teuchos::LAPACK<int,MagnitudeType> lapack_mag;

        // Compute the Ritz vectors.
        // --> For a Hermitian problem this is simply the current basis times the first numRitzVecs_ Schur vectors
        //     
        // --> For a non-Hermitian problem, this involves solving the projected eigenproblem, then
        //     placing the product of the current basis times the first numRitzVecs_ Schur vectors times the
        //     eigenvectors of interest into the Ritz vectors.

        // Get a view of the current Krylov-Schur basis vectors and Schur vectors
        std::vector<int> curind( curDim_ );
        for (int i=0; i<curDim_; i++) { curind[i] = i; }
        Teuchos::RCP<MV> Vtemp = MVT::CloneView( *V_, curind );
        if (problem_->isHermitian()) {
          // Get a view into the current Schur vectors
          Teuchos::SerialDenseMatrix<int,ScalarType> subQ( Teuchos::View, *Q_, curDim_, numRitzVecs_ );

          // Compute the current Ritz vectors      
          MVT::MvTimesMatAddMv( ST_ONE, *Vtemp, subQ, ST_ZERO, *ritzVectors_ );
          
        } else {

          // Get a view into the current Schur vectors.
          Teuchos::SerialDenseMatrix<int,ScalarType> subQ( Teuchos::View, *Q_, curDim_, curDim_ );
          
          // Get a set of work vectors to hold the current Ritz vectors.
          Teuchos::RCP<MV> tmpritzVectors_ = MVT::Clone( *V_, curDim_ );

          // Compute the current Krylov-Schur vectors.
          MVT::MvTimesMatAddMv( ST_ONE, *Vtemp, subQ, ST_ZERO, *tmpritzVectors_ );          

          //  Now compute the eigenvectors of the Schur form
          //  Reset the dense matrix and compute the eigenvalues of the Schur form.
          //
          // Allocate the work space. This space will be used below for calls to:
          // * TREVC (requires 3*N for real, 2*N for complex) 

          int lwork = 3*curDim_;
          std::vector<ScalarType> work( lwork );
          std::vector<MagnitudeType> rwork( curDim_ );
          char side = 'R';
          int mm, info = 0; 
          const int ldvl = 1;
          ScalarType vl[ ldvl ];
          Teuchos::SerialDenseMatrix<int,ScalarType> copyQ( Teuchos::Copy, *Q_, curDim_, curDim_ );
          lapack.TREVC( side, curDim_, schurH_->values(), schurH_->stride(), vl, ldvl,
                        copyQ.values(), copyQ.stride(), curDim_, &mm, &work[0], &rwork[0], &info );
          TEST_FOR_EXCEPTION(info != 0, std::logic_error,
                             "Anasazi::BlockKrylovSchur::computeRitzVectors(): TREVC returned info != 0.");

          // Get a view into the eigenvectors of the Schur form
          Teuchos::SerialDenseMatrix<int,ScalarType> subCopyQ( Teuchos::View, copyQ, curDim_, numRitzVecs_ );
          
          // Convert back to Ritz vectors of the operator.
          std::vector<int> curind( (numRitzVecs_) );
          for (int i=0; i<(int)curind.size(); i++) { curind[i] = i; }

          Teuchos::RCP<MV> view_ritzVectors = MVT::CloneView( *ritzVectors_, curind );
          MVT::MvTimesMatAddMv( ST_ONE, *tmpritzVectors_, subCopyQ, ST_ZERO, *view_ritzVectors );

          // Compute the norm of the new Ritz vectors
          std::vector<MagnitudeType> ritzNrm( numRitzVecs_ );
          MVT::MvNorm( *view_ritzVectors, &ritzNrm );

          // Release memory used to compute Ritz vectors before scaling the current vectors.
          tmpritzVectors_ = Teuchos::null;
          view_ritzVectors = Teuchos::null;
          
          // Scale the Ritz vectors to have Euclidean norm.
          ScalarType ritzScale = ST_ONE;
          for (int i=0; i<numRitzVecs_; i++) {
            
            // If this is a conjugate pair then normalize by the real and imaginary parts.
            if (ritzIndex_[i] == 1 ) {
              ritzScale = ST_ONE/lapack_mag.LAPY2(ritzNrm[i],ritzNrm[i+1]);
              std::vector<int> newind(2);
              newind[0] = i; newind[1] = i+1;
              tmpritzVectors_ = MVT::CloneCopy( *ritzVectors_, newind );
              view_ritzVectors = MVT::CloneView( *ritzVectors_, newind );
              MVT::MvAddMv( ritzScale, *tmpritzVectors_, ST_ZERO, *tmpritzVectors_, *view_ritzVectors );

              // Increment counter for imaginary part
              i++;
            } else {

              // This is a real Ritz value, normalize the vector
              std::vector<int> newind(1);
              newind[0] = i;
              tmpritzVectors_ = MVT::CloneCopy( *ritzVectors_, newind );
              view_ritzVectors = MVT::CloneView( *ritzVectors_, newind );
              MVT::MvAddMv( ST_ONE/ritzNrm[i], *tmpritzVectors_, ST_ZERO, *tmpritzVectors_, *view_ritzVectors );
            }              
          }
          
        } // if (problem_->isHermitian()) 
        
        // The current Ritz vectors have been computed.
        ritzVecsCurrent_ = true;
        
      } // if (!ritzVecsCurrent_)      
    } // if (curDim_)    
  } // computeRitzVectors()
  

  //////////////////////////////////////////////////////////////////////////////////////////////////
  /* Get the current approximate eigenvalues, i.e. Ritz values.
   * 
   * POST-CONDITIONS:
   *
   * ritzValues_ contains Ritz w.r.t. V, H
   * Q_ contains the Schur vectors w.r.t. H
   * schurH_ contains the Schur matrix w.r.t. H
   * ritzOrder_ contains the current ordering from sort manager
   * schurCurrent_ = true if sort = true; i.e. the Schur form is sorted according to the index
   *  vector returned by the sort manager.
   */
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::computeSchurForm( const bool sort )
  {
    // local timer
    Teuchos::TimeMonitor LocalTimer(*timerCompSF_);

    // Check to see if the dimension of the factorization is greater than zero.
    if (curDim_) {

      // Check to see if the Schur factorization is current.
      if (!schurCurrent_) {
        
        // Check to see if the Ritz values are current
        // --> If they are then the Schur factorization is current but not sorted.
        if (!ritzValsCurrent_) {
          Teuchos::LAPACK<int,ScalarType> lapack; 
          Teuchos::LAPACK<int,MagnitudeType> lapack_mag;
          Teuchos::BLAS<int,ScalarType> blas;
          Teuchos::BLAS<int,MagnitudeType> blas_mag;
          
          // Get a view into Q, the storage for H's Schur vectors.
          Teuchos::SerialDenseMatrix<int,ScalarType> subQ( Teuchos::View, *Q_, curDim_, curDim_ );
          
          // Get a copy of H to compute/sort the Schur form.
          schurH_ = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( Teuchos::Copy, *H_, curDim_, curDim_ ) );
          //
          //---------------------------------------------------
          // Compute the Schur factorization of subH
          // ---> Use driver GEES to first reduce to upper Hessenberg 
          //         form and then compute Schur form, outputting Ritz values
          //---------------------------------------------------
          //
          // Allocate the work space. This space will be used below for calls to:
          // * GEES  (requires 3*N for real, 2*N for complex)
          // * TREVC (requires 3*N for real, 2*N for complex) 
          // * TREXC (requires N for real, none for complex)
          // Furthermore, GEES requires a real array of length curDim_ (for complex datatypes)
          //
          int lwork = 3*curDim_;
          std::vector<ScalarType> work( lwork );
          std::vector<MagnitudeType> rwork( curDim_ );
          std::vector<MagnitudeType> tmp_rRitzValues( curDim_ );
          std::vector<MagnitudeType> tmp_iRitzValues( curDim_ );
          std::vector<int> bwork( curDim_ );
          int info = 0, sdim = 0; 
          char jobvs = 'V';
          lapack.GEES( jobvs,curDim_, schurH_->values(), schurH_->stride(), &sdim, &tmp_rRitzValues[0],
                       &tmp_iRitzValues[0], subQ.values(), subQ.stride(), &work[0], lwork, 
                       &rwork[0], &bwork[0], &info );
          
          TEST_FOR_EXCEPTION(info != 0, std::logic_error,
                             "Anasazi::BlockKrylovSchur::computeSchurForm(): GEES returned info != 0.");
          //
          //---------------------------------------------------
          // Use the Krylov-Schur factorization to compute the current Ritz residuals 
          // for ALL the eigenvalues estimates (Ritz values)
          //           || Ax - x\theta || = || U_m+1*B_m+1^H*Q*s || 
          //                              = || B_m+1^H*Q*s ||
          //
          // where U_m+1 is the current Krylov-Schur basis, Q are the Schur vectors, and x = U_m+1*Q*s
          // NOTE: This means that s = e_i if the problem is hermitian, else the eigenvectors
          //       of the Schur form need to be computed.
          //
          // First compute H_{m+1,m}*B_m^T, then determine what 's' is.
          //---------------------------------------------------
          //
          // Get current B_m+1
          Teuchos::SerialDenseMatrix<int,ScalarType> curB(Teuchos::View, *H_,
                                                          blockSize_, curDim_, curDim_ );
          //
          // Compute B_m+1^H*Q
          Teuchos::SerialDenseMatrix<int,ScalarType> subB( blockSize_, curDim_ );
          blas.GEMM( Teuchos::NO_TRANS, Teuchos::NO_TRANS, blockSize_, curDim_, curDim_, ST_ONE, 
                     curB.values(), curB.stride(), subQ.values(), subQ.stride(), 
                     ST_ZERO, subB.values(), subB.stride() );
          //
          // Determine what 's' is and compute Ritz residuals.
          //
          ScalarType* b_ptr = subB.values();
          if (problem_->isHermitian()) {
            //
            // 's' is the i-th canonical basis vector.
            //
            for (int i=0; i<curDim_ ; i++) {
              ritzResiduals_[i] = blas.NRM2(blockSize_, b_ptr + i*blockSize_, 1);
            }   
          } else {
            //
            //  Compute S: the eigenvectors of the block upper triangular, Schur matrix.
            //
            char side = 'R';
            int mm;
            const int ldvl = 1;
            ScalarType vl[ ldvl ];
            Teuchos::SerialDenseMatrix<int,ScalarType> S( curDim_, curDim_ );
            lapack.TREVC( side, curDim_, schurH_->values(), schurH_->stride(), vl, ldvl,
                          S.values(), S.stride(), curDim_, &mm, &work[0], &rwork[0], &info );
            
            TEST_FOR_EXCEPTION(info != 0, std::logic_error,
                               "Anasazi::BlockKrylovSchur::computeSchurForm(): TREVC returned info != 0.");
            //
            // Scale the eigenvectors so that their Euclidean norms are all one.
            //
            scaleRitzVectors( tmp_iRitzValues, &S );
            //
            // Compute ritzRes = *B_m+1^H*Q*S where the i-th column of S is 's' for the i-th Ritz-value
            //
            Teuchos::SerialDenseMatrix<int,ScalarType> ritzRes( blockSize_, curDim_ );
            blas.GEMM( Teuchos::NO_TRANS, Teuchos::NO_TRANS, blockSize_, curDim_, curDim_, ST_ONE, 
                       subB.values(), subB.stride(), S.values(), S.stride(), 
                       ST_ZERO, ritzRes.values(), ritzRes.stride() );

            /* TO DO:  There's be an incorrect assumption made in the computation of the Ritz residuals.
                       This assumption is that the next vector in the Krylov subspace is Euclidean orthonormal.
                       It may not be normalized using Euclidean norm.
            Teuchos::RCP<MV> ritzResVecs = MVT::Clone( *V_, curDim_ );
            std::vector<int> curind(blockSize_);
            for (int i=0; i<blockSize_; i++) { curind[i] = curDim_ + i; }
            Teuchos::RCP<MV> Vtemp = MVT::CloneView(*V_,curind);     
            
            MVT::MvTimesMatAddMv( ST_ONE, *Vtemp, ritzRes, ST_ZERO, *ritzResVecs );
            std::vector<MagnitudeType> ritzResNrms(curDim_);
            MVT::MvNorm( *ritzResVecs, &ritzResNrms );
            i = 0;
            while( i < curDim_ ) {
              if ( tmp_ritzValues[curDim_+i] != MT_ZERO ) {
                ritzResiduals_[i] = lapack_mag.LAPY2( ritzResNrms[i], ritzResNrms[i+1] );
                ritzResiduals_[i+1] = ritzResiduals_[i];
                i = i+2;
              } else {
                ritzResiduals_[i] = ritzResNrms[i];
                i++;
              }
            }
            */
            //
            // Compute the Ritz residuals for each Ritz value.
            // 
            computeRitzResiduals( tmp_iRitzValues, ritzRes, &ritzResiduals_ );
          }
          //
          // Sort the Ritz values.
          //
          {
            Teuchos::TimeMonitor LocalTimer2(*timerSortRitzVal_);
            int i=0;
            if (problem_->isHermitian()) {
              //
              // Sort using just the real part of the Ritz values.
              sm_->sort( this, curDim_, tmp_rRitzValues, &ritzOrder_ ); // don't catch exception
              ritzIndex_.clear();
              while ( i < curDim_ ) {
                // The Ritz value is not complex.
                ritzValues_[i].set(tmp_rRitzValues[i], MT_ZERO);
                ritzIndex_.push_back(0);
                i++;
              }
            }
            else {
              //
              // Sort using both the real and imaginary parts of the Ritz values.
              sm_->sort( this, curDim_, tmp_rRitzValues, tmp_iRitzValues, &ritzOrder_ );
              sortRitzValues( tmp_rRitzValues, tmp_iRitzValues, &ritzValues_, &ritzOrder_, &ritzIndex_ );
            }
            //
            // Sort the ritzResiduals_ based on the ordering from the Sort Manager.
            std::vector<MagnitudeType> ritz2( curDim_ );
            for (int i=0; i<curDim_; i++) { ritz2[i] = ritzResiduals_[ ritzOrder_[i] ]; }
            blas_mag.COPY( curDim_, &ritz2[0], 1, &ritzResiduals_[0], 1 );
            
            // The Ritz values have now been updated.
            ritzValsCurrent_ = true;
          }

        } // if (!ritzValsCurrent_) ...
        // 
        //---------------------------------------------------
        // * The Ritz values and residuals have been updated at this point.
        // 
        // * The Schur factorization of the projected matrix has been computed,
        //   and is stored in (schurH_, Q_).
        //
        // Now the Schur factorization needs to be sorted.
        //---------------------------------------------------
        //
        // Sort the Schur form using the ordering from the Sort Manager.
        if (sort) {
          sortSchurForm( *schurH_, *Q_, ritzOrder_ );    
          //
          // Indicate the Schur form in (schurH_, Q_) is current and sorted
          schurCurrent_ = true;
        }
      } // if (!schurCurrent_) ...
  
    } // if (curDim_) ...
  
  } // computeSchurForm( ... )
  

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Sort the Schur form of H stored in (H,Q) using the ordering vector.
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::sortSchurForm( Teuchos::SerialDenseMatrix<int,ScalarType>& H,
                                                          Teuchos::SerialDenseMatrix<int,ScalarType>& Q,
                                                          std::vector<int>& order ) 
  {
    // local timer
    Teuchos::TimeMonitor LocalTimer(*timerSortSF_);
    //
    //---------------------------------------------------
    // Reorder real Schur factorization, remember to add one to the indices for the
    // fortran call and determine offset.  The offset is necessary since the TREXC
    // method reorders in a nonsymmetric fashion, thus we use the reordering in
    // a stack-like fashion.  Also take into account conjugate pairs, which may mess
    // up the reordering, since the pair is moved if one of the pair is moved.
    //---------------------------------------------------
    //
    int i = 0, nevtemp = 0;
    char compq = 'V';
    std::vector<int> offset2( curDim_ );
    std::vector<int> order2( curDim_ );

    // LAPACK objects.
    Teuchos::LAPACK<int,ScalarType> lapack; 
    int lwork = 3*curDim_;
    std::vector<ScalarType> work( lwork );

    while (i < curDim_) {
      if ( ritzIndex_[i] != 0 ) { // This is the first value of a complex conjugate pair
        offset2[nevtemp] = 0;
        for (int j=i; j<curDim_; j++) {
          if (order[j] > order[i]) { offset2[nevtemp]++; }
        }
        order2[nevtemp] = order[i];
        i = i+2;
      } else {
        offset2[nevtemp] = 0;
        for (int j=i; j<curDim_; j++) {
          if (order[j] > order[i]) { offset2[nevtemp]++; }
        }
        order2[nevtemp] = order[i];
        i++;
      }
      nevtemp++;
    }
    ScalarType *ptr_h = H.values();
    ScalarType *ptr_q = Q.values();
    int ldh = H.stride(), ldq = Q.stride();
    int info = 0;
    for (i=nevtemp-1; i>=0; i--) {
      lapack.TREXC( compq, curDim_, ptr_h, ldh, ptr_q, ldq, order2[i]+1+offset2[i], 
                    1, &work[0], &info );
      TEST_FOR_EXCEPTION(info != 0, std::logic_error,
                         "Anasazi::BlockKrylovSchur::computeSchurForm(): TREXC returned info != 0.");
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Print the current status of the solver
  template <class ScalarType, class MV, class OP>
  void BlockKrylovSchur<ScalarType,MV,OP>::currentStatus(std::ostream &os) 
  {
    using std::endl;

    os.setf(std::ios::scientific, std::ios::floatfield);
    os.precision(6);
    os <<"================================================================================" << endl;
    os << endl;
    os <<"                         BlockKrylovSchur Solver Status" << endl;
    os << endl;
    os <<"The solver is "<<(initialized_ ? "initialized." : "not initialized.") << endl;
    os <<"The number of iterations performed is " <<iter_<<endl;
    os <<"The block size is         " << blockSize_<<endl;
    os <<"The number of blocks is   " << numBlocks_<<endl;
    os <<"The current basis size is " << curDim_<<endl;
    os <<"The number of auxiliary vectors is    " << numAuxVecs_ << endl;
    os <<"The number of operations Op*x   is "<<count_ApplyOp_<<endl;

    os.setf(std::ios_base::right, std::ios_base::adjustfield);

    os << endl;
    if (initialized_) {
      os <<"CURRENT RITZ VALUES             "<<endl;
      if (ritzIndex_.size() != 0) {
        int numPrint = (curDim_ < numRitzPrint_? curDim_: numRitzPrint_);
        if (problem_->isHermitian()) {
          os << std::setw(20) << "Ritz Value" 
             << std::setw(20) << "Ritz Residual"
             << endl;
          os <<"--------------------------------------------------------------------------------"<<endl;
          for (int i=0; i<numPrint; i++) {
            os << std::setw(20) << ritzValues_[i].realpart 
               << std::setw(20) << ritzResiduals_[i] 
               << endl;
          }
        } else {
          os << std::setw(24) << "Ritz Value" 
             << std::setw(30) << "Ritz Residual"
             << endl;
          os <<"--------------------------------------------------------------------------------"<<endl;
          for (int i=0; i<numPrint; i++) {
            // Print out the real eigenvalue.
            os << std::setw(15) << ritzValues_[i].realpart;
            if (ritzValues_[i].imagpart < MT_ZERO) {
              os << " - i" << std::setw(15) << Teuchos::ScalarTraits<MagnitudeType>::magnitude(ritzValues_[i].imagpart);
            } else {
              os << " + i" << std::setw(15) << ritzValues_[i].imagpart;
            }              
            os << std::setw(20) << ritzResiduals_[i] << endl;
          }
        }
      } else {
        os << std::setw(20) << "[ NONE COMPUTED ]" << endl;
      }
    }
    os << endl;
    os <<"================================================================================" << endl;
    os << endl;
  }
  
} // End of namespace Anasazi

#endif

// End of file AnasaziBlockKrylovSchur.hpp
