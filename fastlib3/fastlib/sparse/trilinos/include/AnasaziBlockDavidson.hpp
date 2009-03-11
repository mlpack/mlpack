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

/*! \file AnasaziBlockDavidson.hpp
  \brief Implementation of the block Davidson method
*/

#ifndef ANASAZI_BLOCK_DAVIDSON_HPP
#define ANASAZI_BLOCK_DAVIDSON_HPP

#include "AnasaziTypes.hpp"

#include "AnasaziEigensolver.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziOperatorTraits.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include "AnasaziMatOrthoManager.hpp"
#include "AnasaziSolverUtils.hpp"

#include "Teuchos_LAPACK.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"

/*!     \class Anasazi::BlockDavidson
  
        \brief This class implements a Block Davidson iteration, a preconditioned iteration for solving linear Hermitian eigenproblems.

        This method is described in <em>A Comparison of Eigensolvers for
        Large-scale 3D Modal Analysis Using AMG-Preconditioned Iterative
        Methods</em>, P. Arbenz, U. L. Hetmaniuk, R. B. Lehoucq, R. S.
        Tuminaro, Internat. J. for Numer. Methods Engrg., 64, pp. 204-236
        (2005)
        
        \ingroup anasazi_solver_framework

        \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, Heidi Thornquist
*/

namespace Anasazi {

  //! @name BlockDavidson Structures 
  //@{ 

  /** \brief Structure to contain pointers to BlockDavidson state variables.
   *
   * This struct is utilized by BlockDavidson::initialize() and BlockDavidson::getState().
   */
  template <class ScalarType, class MV>
  struct BlockDavidsonState {
    /*! \brief The current dimension of the solver.
     *
     * This should always be equal to BlockDavdison::getCurSubspaceDim()
     */
    int curDim;
    /*! \brief The basis for the Krylov space.
     *
     * V has BlockDavidson::getMaxSubspaceDim() vectors, but only the first \c curDim are valid.
     */
    Teuchos::RCP<const MV> V;
    //! The current eigenvectors.
    Teuchos::RCP<const MV> X; 
    //! The image of the current eigenvectors under K.
    Teuchos::RCP<const MV> KX; 
    //! The image of the current eigenvectors under M, or Teuchos::null if M was not specified.
    Teuchos::RCP<const MV> MX;
    //! The current residual vectors
    Teuchos::RCP<const MV> R;
    /*! \brief The current preconditioned residual vectors.
     *
     *  H is a pointer into V, and is only useful when BlockDavidson::iterate() throw a BlockDavidsonOrthoFailure exception.
     */
    Teuchos::RCP<const MV> H;
    //! The current Ritz values. This vector is a copy of the internal 
    Teuchos::RCP<const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> > T;
    /*! \brief The current projected K matrix.
     *
     * KK is of order BlockDavidson::getMaxSubspaceDim(), but only the principal submatrix of order \c curDim is meaningful. It is Hermitian in memory.
     *
     */
    Teuchos::RCP<const Teuchos::SerialDenseMatrix<int,ScalarType> > KK;
    BlockDavidsonState() : curDim(0), V(Teuchos::null),
                           X(Teuchos::null), KX(Teuchos::null), MX(Teuchos::null),
                           R(Teuchos::null), H(Teuchos::null),
                           T(Teuchos::null), KK(Teuchos::null) {}
  };

  //@}

  //! @name BlockDavidson Exceptions
  //@{ 

  /** \brief BlockDavidsonInitFailure is thrown when the BlockDavidson solver is unable to
   * generate an initial iterate in the BlockDavidson::initialize() routine. 
   *
   * This exception is thrown from the BlockDavidson::initialize() method, which is
   * called by the user or from the BlockDavidson::iterate() method if isInitialized()
   * == \c false.
   *
   * In the case that this exception is thrown, 
   * BlockDavidson::isInitialized() will be \c false and the user will need to provide
   * a new initial iterate to the solver.
   *
   */
  class BlockDavidsonInitFailure : public AnasaziError {public:
    BlockDavidsonInitFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  /** \brief BlockDavidsonOrthoFailure is thrown when the orthogonalization manager is
   * unable to orthogonalize the preconditioned residual against (a.k.a. \c H)
   * the current basis (a.k.a. \c V).
   *
   * This exception is thrown from the BlockDavidson::iterate() method.
   *
   */
  class BlockDavidsonOrthoFailure : public AnasaziError {public:
    BlockDavidsonOrthoFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};
  
  //@}


  template <class ScalarType, class MV, class OP>
  class BlockDavidson : public Eigensolver<ScalarType,MV,OP> { 
  public:
    //! @name Constructor/Destructor
    //@{ 
    
    /*! \brief %BlockDavidson constructor with eigenproblem, solver utilities, and parameter list of solver options.
     *
     * This constructor takes pointers required by the eigensolver, in addition
     * to a parameter list of options for the eigensolver. These options include the following:
     *   - "Block Size" - an \c int specifying the block size used by the algorithm. This can also be specified using the setBlockSize() method.
     *   - "Num Blocks" - an \c int specifying the maximum number of blocks allocated for the solver basis.
     */
    BlockDavidson( const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem, 
                   const Teuchos::RCP<SortManager<ScalarType,MV,OP> > &sorter,
                   const Teuchos::RCP<OutputManager<ScalarType> > &printer,
                   const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
                   const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> > &ortho,
                   Teuchos::ParameterList &params 
                 );
    
    //! %Anasazi::BlockDavidson destructor.
    virtual ~BlockDavidson();
    //@}


    //! @name Solver methods
    //@{ 

    /*! \brief This method performs %BlockDavidson iterations until the status
     * test indicates the need to stop or an error occurs (in which case, an 
     * appropriate exception is thrown).
     *
     * iterate() will first determine whether the solver is uninitialized; if
     * not, it will call initialize(). After
     * initialization, the solver performs block Davidson iterations until the
     * status test evaluates as ::Passed, at which point the method returns to
     * the caller. 
     *
     * The block Davidson iteration proceeds as follows:
     * -# The current residual (R) is preconditioned to form H
     * -# H is orthogonalized against the auxiliary vectors and the previous basis vectors, and made orthonormal.
     * -# The current basis is expanded with H and used to project the problem matrix.
     * -# The projected eigenproblem is solved, and the desired eigenvectors and eigenvalues are selected.
     * -# These are used to form the new eigenvector estimates (X).
     * -# The new residual (R) is formed.
     *
     * The status test is queried at the beginning of the iteration.
     *
     * Possible exceptions thrown include std::invalid_argument or
     * one of the BlockDavidson-specific exceptions.
     */
    void iterate();

    /*! \brief Initialize the solver to an iterate, optionally providing the
     * current basis and projected problem matrix, the current Ritz vectors and values,
     * and the current residual.
     *
     * The %BlockDavidson eigensolver contains a certain amount of state,
     * including the current Krylov basis, the current eigenvectors, 
     * the current residual, etc. (see getState())
     *
     * initialize() gives the user the opportunity to manually set these,
     * although this must be done with caution, as the validity of the
     * user input will not be checked.
     *
     * \post 
     * <li>isInitialized() == \c true (see post-conditions of isInitialize())
     *
     * The user has the option of specifying any component of the state using
     * initialize(). However, these arguments are assumed to match the
     * post-conditions specified under isInitialized(). Any component of the
     * state (i.e., KX) not given to initialize() will be generated.
     *
     * Note, for any pointer in \c newstate which directly points to the multivectors in 
     * the solver, the data is not copied.
     */
    void initialize(BlockDavidsonState<ScalarType,MV> newstate);

    /*! \brief Initialize the solver with the initial vectors from the eigenproblem
     *  or random data.
     */
    void initialize();

    /*! \brief Indicates whether the solver has been initialized or not.
     *
     * \return bool indicating the state of the solver.
     * \post
     * If isInitialized() == \c true:
     *    - getCurSubspaceDim() > 0 and is a multiple of getBlockSize()
     *    - the first getCurSubspaceDim() vectors of V are orthogonal to auxiliary vectors and have orthonormal columns
     *    - the principal submatrix of order getCurSubspaceDim() of KK contains the project eigenproblem matrix
     *    - X contains the Ritz vectors with respect to the current Krylov basis
     *    - T contains the Ritz values with respect to the current Krylov basis
     *    - KX == Op*X
     *    - MX == M*X if M != Teuchos::null\n
     *      Otherwise, MX == Teuchos::null
     *    - R contains the residual vectors with respect to X
     */
    bool isInitialized() const;

    /*! \brief Get access to the current state of the eigensolver.
     * 
     * The data is only valid if isInitialized() == \c true. 
     *
     * The data for the preconditioned residual is only meaningful in the
     * scenario that the solver throws a ::BlockDavidsonRitzFailure exception
     * during iterate().
     *
     * \returns A BlockDavidsonState object containing const pointers to the current
     * solver state. Note, these are direct pointers to the multivectors; they are not
     * pointers to views of the multivectors.
     */
    BlockDavidsonState<ScalarType,MV> getState() const;
    
    //@}


    //! @name Status methods
    //@{ 

    //! \brief Get the current iteration count.
    int getNumIters() const;

    //! \brief Reset the iteration count.
    void resetNumIters();

    /*! \brief Get access to the current Ritz vectors.
      
        \return A multivector with getBlockSize() vectors containing 
        the sorted Ritz vectors corresponding to the most significant Ritz values. 
        The i-th vector of the return corresponds to the i-th Ritz vector; there is no need to use
        getRitzIndex().
     */
    Teuchos::RCP<const MV> getRitzVectors();

    /*! \brief Get the Ritz values for the previous iteration.
     *
     *  \return A vector of length getCurSubspaceDim() containing the Ritz values from the
     *  previous projected eigensolve.
     */
    std::vector<Value<ScalarType> > getRitzValues();


    /*! \brief Get the index used for extracting individual Ritz vectors from getRitzVectors().
     *
     * Because BlockDavidson is a Hermitian solver, all Ritz values are real and all Ritz vectors can be represented in a 
     * single column of a multivector. Therefore, getRitzIndex() is not needed when using the output from getRitzVectors().
     *
     * \return An \c int vector of size getCurSubspaceDim() composed of zeros.
     */
    std::vector<int> getRitzIndex();


    /*! \brief Get the current residual norms, computing the norms if they are not up-to-date with the current residual vectors.
     *
     *  \return A vector of length getCurSubspaceDim() containing the norms of the
     *  residuals, with respect to the orthogonalization manager's norm() method.
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getResNorms();


    /*! \brief Get the current residual 2-norms, computing the norms if they are not up-to-date with the current residual vectors.
     *
     *  \return A vector of length getCurSubspaceDim() containing the 2-norms of the
     *  current residuals. 
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getRes2Norms();


    /*! \brief Get the 2-norms of the residuals.
     * 
     * The Ritz residuals are not defined for the %LOBPCG iteration. Hence, this method returns the 
     * 2-norms of the direct residuals, and is equivalent to calling getRes2Norms().
     *
     *  \return A vector of length getBlockSize() containing the 2-norms of the direct residuals.
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getRitzRes2Norms();

    /*! \brief Get the dimension of the search subspace used to generate the current eigenvectors and eigenvalues.
     *
     *  \return An integer specifying the rank of the Krylov subspace currently in use by the eigensolver. If isInitialized() == \c false, 
     *  the return is 0. Otherwise, it will be some strictly positive multiple of getBlockSize().
     */
    int getCurSubspaceDim() const;

    //! Get the maximum dimension allocated for the search subspace. For %BlockDavidson, this always returns numBlocks*blockSize.
    int getMaxSubspaceDim() const;

    //@}


    //! @name Accessor routines from Eigensolver
    //@{ 


    //! Get a constant reference to the eigenvalue problem.
    const Eigenproblem<ScalarType,MV,OP>& getProblem() const;

    /*! \brief Set the blocksize. 
     *
     * This method is required to support the interface provided by Eigensolver. However, the preferred method
     * of setting the allocated size for the BlockDavidson eigensolver is setSize(). In fact, setBlockSize() 
     * simply calls setSize(), maintaining the current number of blocks.
     *
     * The block size determines the number of Ritz vectors and values that are computed on each iteration, thereby
     * determining the increase in the Krylov subspace at each iteration.
     */
    void setBlockSize(int blockSize);

    //! Get the blocksize used by the iterative solver.
    int getBlockSize() const;

    /*! \brief Set the auxiliary vectors for the solver.
     *
     *  Because the current basis V cannot be assumed
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
    Teuchos::Array<Teuchos::RCP<const MV> > getAuxVecs() const;

    //@}

    //! @name BlockDavidson-specific accessor routines
    //@{ 

    /*! \brief Set the blocksize and number of blocks to be used by the
     * iterative solver in solving this eigenproblem.
     *  
     *  Changing either the block size or the number of blocks will reset the
     *  solver to an uninitialized state.
     *
     *  The requested block size must be strictly positive; the number of blocks must be 
     *  greater than one. Invalid arguments will result in a std::invalid_argument exception.
     */
    void setSize(int blockSize, int numBlocks);

    //@}

    //! @name Output methods
    //@{ 

    //! This method requests that the solver print out its current status to the given output stream.
    void currentStatus(std::ostream &os);

    //@}

  private:
    //
    // Convenience typedefs
    //
    typedef SolverUtils<ScalarType,MV,OP> Utils;
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename SCT::magnitudeType MagnitudeType;
    const MagnitudeType ONE;  
    const MagnitudeType ZERO; 
    const MagnitudeType NANVAL;
    //
    // Internal structs
    //
    struct CheckList {
      bool checkV;
      bool checkX, checkMX, checkKX;
      bool checkH, checkMH, checkKH;
      bool checkR, checkQ;
      bool checkKK;
      CheckList() : checkV(false),
                    checkX(false),checkMX(false),checkKX(false),
                    checkH(false),checkMH(false),checkKH(false),
                    checkR(false),checkQ(false),checkKK(false) {};
    };
    //
    // Internal methods
    //
    std::string accuracyCheck(const CheckList &chk, const std::string &where) const;
    //
    // Classes inputed through constructor that define the eigenproblem to be solved.
    //
    const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> >     problem_;
    const Teuchos::RCP<SortManager<ScalarType,MV,OP> >      sm_;
    const Teuchos::RCP<OutputManager<ScalarType> >          om_;
    const Teuchos::RCP<StatusTest<ScalarType,MV,OP> >       tester_;
    const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> >  orthman_;
    //
    // Information obtained from the eigenproblem
    //
    Teuchos::RCP<const OP> Op_;
    Teuchos::RCP<const OP> MOp_;
    Teuchos::RCP<const OP> Prec_;
    bool hasM_;
    //
    // Internal timers
    //
    Teuchos::RCP<Teuchos::Time> timerOp_, timerMOp_, timerPrec_,
                                        timerSortEval_, timerDS_,
                                        timerLocal_, timerCompRes_, 
                                        timerOrtho_, timerInit_;
    //
    // Counters
    //
    int count_ApplyOp_, count_ApplyM_, count_ApplyPrec_;

    //
    // Algorithmic parameters.
    //
    // blockSize_ is the solver block size; it controls the number of eigenvectors that 
    // we compute, the number of residual vectors that we compute, and therefore the number
    // of vectors added to the basis on each iteration.
    int blockSize_;
    // numBlocks_ is the size of the allocated space for the Krylov basis, in blocks.
    int numBlocks_; 
    
    // 
    // Current solver state
    //
    // initialized_ specifies that the basis vectors have been initialized and the iterate() routine
    // is capable of running; _initialize is controlled  by the initialize() member method
    // For the implications of the state of initialized_, please see documentation for initialize()
    bool initialized_;
    //
    // curDim_ reflects how much of the current basis is valid 
    // NOTE: 0 <= curDim_ <= blockSize_*numBlocks_
    // this also tells us how many of the values in theta_ are valid Ritz values
    int curDim_;
    //
    // State Multivecs
    // H_,KH_,MH_ will not own any storage
    // H_ will occasionally point at the current block of vectors in the basis V_
    // MH_,KH_ will occasionally point at MX_,KX_ when they are used as temporary storage
    Teuchos::RCP<MV> X_, KX_, MX_, R_,
                             H_, KH_, MH_,
                             V_;
    //
    // Projected matrices
    //
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > KK_;
    // 
    // auxiliary vectors
    Teuchos::Array<Teuchos::RCP<const MV> > auxVecs_;
    int numAuxVecs_;
    //
    // Number of iterations that have been performed.
    int iter_;
    // 
    // Current eigenvalues, residual norms
    std::vector<MagnitudeType> theta_, Rnorms_, R2norms_;
    // 
    // are the residual norms current with the residual?
    bool Rnorms_current_, R2norms_current_;

  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Implementations
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Constructor
  template <class ScalarType, class MV, class OP>
  BlockDavidson<ScalarType,MV,OP>::BlockDavidson(
        const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem, 
        const Teuchos::RCP<SortManager<ScalarType,MV,OP> > &sorter,
        const Teuchos::RCP<OutputManager<ScalarType> > &printer,
        const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
        const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> > &ortho,
        Teuchos::ParameterList &params
        ) :
    ONE(Teuchos::ScalarTraits<MagnitudeType>::one()),
    ZERO(Teuchos::ScalarTraits<MagnitudeType>::zero()),
    NANVAL(Teuchos::ScalarTraits<MagnitudeType>::nan()),
    // problem, tools
    problem_(problem), 
    sm_(sorter),
    om_(printer),
    tester_(tester),
    orthman_(ortho),
    // timers, counters
    timerOp_(Teuchos::TimeMonitor::getNewTimer("Operation Op*x")),
    timerMOp_(Teuchos::TimeMonitor::getNewTimer("Operation M*x")),
    timerPrec_(Teuchos::TimeMonitor::getNewTimer("Operation Prec*x")),
    timerSortEval_(Teuchos::TimeMonitor::getNewTimer("Sorting eigenvalues")),
    timerDS_(Teuchos::TimeMonitor::getNewTimer("Direct solve")),
    timerLocal_(Teuchos::TimeMonitor::getNewTimer("Local update")),
    timerCompRes_(Teuchos::TimeMonitor::getNewTimer("Computing residuals")),
    timerOrtho_(Teuchos::TimeMonitor::getNewTimer("Orthogonalization")),
    timerInit_(Teuchos::TimeMonitor::getNewTimer("Initialization")),
    count_ApplyOp_(0),
    count_ApplyM_(0),
    count_ApplyPrec_(0),
    // internal data
    blockSize_(0),
    numBlocks_(0),
    initialized_(false),
    curDim_(0),
    auxVecs_( Teuchos::Array<Teuchos::RCP<const MV> >(0) ), 
    numAuxVecs_(0),
    iter_(0),
    Rnorms_current_(false),
    R2norms_current_(false)
  {     
    TEST_FOR_EXCEPTION(problem_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: user passed null problem pointer.");
    TEST_FOR_EXCEPTION(sm_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: user passed null sort manager pointer.");
    TEST_FOR_EXCEPTION(om_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: user passed null output manager pointer.");
    TEST_FOR_EXCEPTION(tester_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: user passed null status test pointer.");
    TEST_FOR_EXCEPTION(orthman_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: user passed null orthogonalization manager pointer.");
    TEST_FOR_EXCEPTION(problem_->isProblemSet() == false, std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: problem is not set.");
    TEST_FOR_EXCEPTION(problem_->isHermitian() == false, std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: problem is not hermitian.");

    // get the problem operators
    Op_   = problem_->getOperator();
    TEST_FOR_EXCEPTION(Op_ == Teuchos::null, std::invalid_argument,
                       "Anasazi::BlockDavidson::constructor: problem provides no operator.");
    MOp_  = problem_->getM();
    Prec_ = problem_->getPrec();
    hasM_ = (MOp_ != Teuchos::null);

    // set the block size and allocate data
    int bs = params.get("Block Size", problem_->getNEV());
    int nb = params.get("Num Blocks", 2);
    setSize(bs,nb);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Destructor
  template <class ScalarType, class MV, class OP>
  BlockDavidson<ScalarType,MV,OP>::~BlockDavidson() {}


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the block size
  // This simply calls setSize(), modifying the block size while retaining the number of blocks.
  template <class ScalarType, class MV, class OP>
  void BlockDavidson<ScalarType,MV,OP>::setBlockSize (int blockSize) 
  {
    setSize(blockSize,numBlocks_);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Return the current auxiliary vectors
  template <class ScalarType, class MV, class OP>
  Teuchos::Array<Teuchos::RCP<const MV> > BlockDavidson<ScalarType,MV,OP>::getAuxVecs() const {
    return auxVecs_;
  }



  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current block size
  template <class ScalarType, class MV, class OP>
  int BlockDavidson<ScalarType,MV,OP>::getBlockSize() const {
    return(blockSize_); 
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return eigenproblem
  template <class ScalarType, class MV, class OP>
  const Eigenproblem<ScalarType,MV,OP>& BlockDavidson<ScalarType,MV,OP>::getProblem() const { 
    return(*problem_); 
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return max subspace dim
  template <class ScalarType, class MV, class OP>
  int BlockDavidson<ScalarType,MV,OP>::getMaxSubspaceDim() const {
    return blockSize_*numBlocks_;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return current subspace dim
  template <class ScalarType, class MV, class OP>
  int BlockDavidson<ScalarType,MV,OP>::getCurSubspaceDim() const {
    if (!initialized_) return 0;
    return curDim_;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return ritz residual 2-norms
  template <class ScalarType, class MV, class OP>
  std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> 
  BlockDavidson<ScalarType,MV,OP>::getRitzRes2Norms() {
    return this->getRes2Norms();
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return ritz index
  template <class ScalarType, class MV, class OP>
  std::vector<int> BlockDavidson<ScalarType,MV,OP>::getRitzIndex() {
    std::vector<int> ret(curDim_,0);
    return ret;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return ritz values
  template <class ScalarType, class MV, class OP>
  std::vector<Value<ScalarType> > BlockDavidson<ScalarType,MV,OP>::getRitzValues() { 
    std::vector<Value<ScalarType> > ret(curDim_);
    for (int i=0; i<curDim_; ++i) {
      ret[i].realpart = theta_[i];
      ret[i].imagpart = ZERO;
    }
    return ret;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return pointer to ritz vectors
  template <class ScalarType, class MV, class OP>
  Teuchos::RCP<const MV> BlockDavidson<ScalarType,MV,OP>::getRitzVectors() {
    return X_;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // reset number of iterations
  template <class ScalarType, class MV, class OP>
  void BlockDavidson<ScalarType,MV,OP>::resetNumIters() { 
    iter_=0; 
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return number of iterations
  template <class ScalarType, class MV, class OP>
  int BlockDavidson<ScalarType,MV,OP>::getNumIters() const { 
    return(iter_); 
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return state pointers
  template <class ScalarType, class MV, class OP>
  BlockDavidsonState<ScalarType,MV> BlockDavidson<ScalarType,MV,OP>::getState() const {
    BlockDavidsonState<ScalarType,MV> state;
    state.curDim = curDim_;
    state.V = V_;
    state.X = X_;
    state.KX = KX_;
    if (hasM_) {
      state.MX = MX_;
    }
    else {
      state.MX = Teuchos::null;
    }
    state.R = R_;
    state.H = H_;
    state.KK = KK_;
    if (curDim_ > 0) {
      state.T = Teuchos::rcp(new std::vector<MagnitudeType>(&theta_[0],&theta_[curDim_]));
    }
    else {
      state.T = Teuchos::rcp(new std::vector<MagnitudeType>(0));
    }
    return state;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Return initialized state
  template <class ScalarType, class MV, class OP>
  bool BlockDavidson<ScalarType,MV,OP>::isInitialized() const { return initialized_; }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the block size and make necessary adjustments.
  template <class ScalarType, class MV, class OP>
  void BlockDavidson<ScalarType,MV,OP>::setSize (int blockSize, int numBlocks) 
  {
    // time spent here counts towards timerInit_
    Teuchos::TimeMonitor lcltimer( *timerInit_ );

    // This routine only allocates space; it doesn't not perform any computation
    // any change in size will invalidate the state of the solver.

    TEST_FOR_EXCEPTION(blockSize < 1, std::invalid_argument, "Anasazi::BlockDavidson::setSize(blocksize,numblocks): blocksize must be strictly positive.");
    TEST_FOR_EXCEPTION(numBlocks < 2, std::invalid_argument, "Anasazi::BlockDavidson::setSize(blocksize,numblocks): numblocks must be greater than one.");
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
    // in case of that strange scenario, we will try to Clone from X_ first, then resort to getInitVec()
    if (X_ != Teuchos::null) { // this is equivalent to blockSize_ > 0
      tmp = X_;
    }
    else {
      tmp = problem_->getInitVec();
      TEST_FOR_EXCEPTION(tmp == Teuchos::null,std::invalid_argument,
                         "Anasazi::BlockDavidson::setSize(): eigenproblem did not specify initial vectors to clone from.");
    }

    TEST_FOR_EXCEPTION(numAuxVecs_+blockSize*numBlocks > MVT::GetVecLength(*tmp),std::invalid_argument,
                       "Anasazi::BlockDavidson::setSize(): max subspace dimension and auxilliary subspace too large.");


    //////////////////////////////////
    // blockSize dependent
    //
    // grow/allocate vectors
    Rnorms_.resize(blockSize_,NANVAL);
    R2norms_.resize(blockSize_,NANVAL);
    //
    // clone multivectors off of tmp
    //
    // free current allocation first, to make room for new allocation
    X_ = Teuchos::null;
    KX_ = Teuchos::null;
    MX_ = Teuchos::null;
    R_ = Teuchos::null;
    V_ = Teuchos::null;

    om_->print(Debug," >> Allocating X_\n");
    X_ = MVT::Clone(*tmp,blockSize_);
    om_->print(Debug," >> Allocating KX_\n");
    KX_ = MVT::Clone(*tmp,blockSize_);
    if (hasM_) {
      om_->print(Debug," >> Allocating MX_\n");
      MX_ = MVT::Clone(*tmp,blockSize_);
    }
    else {
      MX_ = X_;
    }
    om_->print(Debug," >> Allocating R_\n");
    R_ = MVT::Clone(*tmp,blockSize_);

    //////////////////////////////////
    // blockSize*numBlocks dependent
    //
    int newsd = blockSize_*numBlocks_;
    theta_.resize(blockSize_*numBlocks_,NANVAL);
    om_->print(Debug," >> Allocating V_\n");
    V_ = MVT::Clone(*tmp,newsd);
    KK_ = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(newsd,newsd) );

    om_->print(Debug," >> done allocating.\n");

    initialized_ = false;
    curDim_ = 0;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the auxiliary vectors
  template <class ScalarType, class MV, class OP>
  void BlockDavidson<ScalarType,MV,OP>::setAuxVecs(const Teuchos::Array<Teuchos::RCP<const MV> > &auxvecs) {
    typedef typename Teuchos::Array<Teuchos::RCP<const MV> >::iterator tarcpmv;

    // set new auxiliary vectors
    auxVecs_ = auxvecs;
    numAuxVecs_ = 0;
    for (tarcpmv i=auxVecs_.begin(); i != auxVecs_.end(); ++i) {
      numAuxVecs_ += MVT::GetNumberVecs(**i);
    }

    // If the solver has been initialized, V is not necessarily orthogonal to new auxiliary vectors
    if (numAuxVecs_ > 0 && initialized_) {
      initialized_ = false;
    }

    if (om_->isVerbosity( Debug ) ) {
      CheckList chk;
      chk.checkQ = true;
      om_->print( Debug, accuracyCheck(chk, ": in setAuxVecs()") );
    }
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  /* Initialize the state of the solver
   * 
   * POST-CONDITIONS:
   *
   * V_ is orthonormal, orthogonal to auxVecs_, for first curDim_ vectors
   * theta_ contains Ritz w.r.t. V_(1:curDim_)
   * X is Ritz vectors w.r.t. V_(1:curDim_)
   * KX = Op*X
   * MX = M*X if hasM_
   * R = KX - MX*diag(theta_)
   *
   */
  template <class ScalarType, class MV, class OP>
  void BlockDavidson<ScalarType,MV,OP>::initialize(BlockDavidsonState<ScalarType,MV> newstate)
  {
    // NOTE: memory has been allocated by setBlockSize(). Use setBlock below; do not Clone
    // NOTE: Overall time spent in this routine is counted to timerInit_; portions will also be counted towards other primitives

    Teuchos::TimeMonitor lcltimer( *timerInit_ );

    std::vector<int> bsind(blockSize_);
    for (int i=0; i<blockSize_; ++i) bsind[i] = i;

    Teuchos::BLAS<int,ScalarType> blas;

    // in BlockDavidson, V is primary
    // the order of dependence follows like so.
    // --init->               V,KK
    //    --ritz analysis->   theta,X  
    //       --op apply->     KX,MX  
    //          --compute->   R
    // 
    // if the user specifies all data for a level, we will accept it.
    // otherwise, we will generate the whole level, and all subsequent levels.
    //
    // the data members are ordered based on dependence, and the levels are
    // partitioned according to the amount of work required to produce the
    // items in a level.
    //
    // inconsistent multivectors widths and lengths will not be tolerated, and
    // will be treated with exceptions.
    //
    // for multivector pointers in newstate which point directly (as opposed to indirectly, via a view) to 
    // multivectors in the solver, the copy will not be affected.

    // set up V and KK: get them from newstate if user specified them
    // otherwise, set them manually
    Teuchos::RCP<MV> lclV;
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > lclKK;

    if (newstate.V != Teuchos::null && newstate.KK != Teuchos::null) {
      TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.V) != MVT::GetVecLength(*V_), std::invalid_argument, 
                          "Anasazi::BlockDavidson::initialize(newstate): Vector length of V not correct." );
      TEST_FOR_EXCEPTION( newstate.curDim < blockSize_, std::invalid_argument, 
                          "Anasazi::BlockDavidson::initialize(newstate): Rank of new state must be at least blockSize().");
      TEST_FOR_EXCEPTION( newstate.curDim > blockSize_*numBlocks_, std::invalid_argument, 
                          "Anasazi::BlockDavidson::initialize(newstate): Rank of new state must be less than getMaxSubspaceDim().");
      TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.V) < newstate.curDim, std::invalid_argument, 
                          "Anasazi::BlockDavidson::initialize(newstate): Multivector for basis in new state must be as large as specified state rank.");

      curDim_ = newstate.curDim;
      // pick an integral amount
      curDim_ = (int)(curDim_ / blockSize_)*blockSize_;

      TEST_FOR_EXCEPTION( curDim_ != newstate.curDim, std::invalid_argument, 
                          "Anasazi::BlockDavidson::initialize(newstate): Rank of new state must be a multiple of getBlockSize().");

      // check size of KK
      TEST_FOR_EXCEPTION( newstate.KK->numRows() < curDim_ || newstate.KK->numCols() < curDim_, std::invalid_argument, 
                          "Anasazi::BlockDavidson::initialize(newstate): Projected matrix in new state must be as large as specified state rank.");


      // put data in V
      std::vector<int> nevind(curDim_);
      for (int i=0; i<curDim_; ++i) nevind[i] = i;
      if (newstate.V != V_) {
        MVT::SetBlock(*newstate.V,nevind,*V_);
      }
      lclV = MVT::CloneView(*V_,nevind);

      // put data into KK_
      lclKK = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::View,*KK_,curDim_,curDim_) );
      if (newstate.KK != KK_) {
        Teuchos::SerialDenseMatrix<int,ScalarType> newKK(Teuchos::View,*newstate.KK,curDim_,curDim_);
        lclKK->assign(newKK);
      }
      //
      // make lclKK Hermitian in memory (copy the upper half to the lower half)
      for (int j=0; j<curDim_; ++j) {
        for (int i=j+1; i<curDim_; ++i) {
          (*lclKK)(i,j) = SCT::conjugate((*lclKK)(j,i));
        }
      }
    }
    else {
      // user did not specify one of V or KK
      // get vectors from problem or generate something, projectAndNormalize
      Teuchos::RCP<const MV> ivec = problem_->getInitVec();
      TEST_FOR_EXCEPTION(ivec == Teuchos::null,std::invalid_argument,
                         "Anasazi::BlockDavdison::initialize(newstate): Eigenproblem did not specify initial vectors to clone from.");
      // clear newstate so we won't use any data from it below
      newstate.X      = Teuchos::null;
      newstate.MX     = Teuchos::null;
      newstate.KX     = Teuchos::null;
      newstate.R      = Teuchos::null;
      newstate.H      = Teuchos::null;
      newstate.T      = Teuchos::null;
      newstate.KK     = Teuchos::null;
      newstate.V      = Teuchos::null;
      newstate.curDim = 0;

      curDim_ = MVT::GetNumberVecs(*ivec);
      // pick the largest multiple of blockSize_
      curDim_ = (int)(curDim_ / blockSize_)*blockSize_;
      if (curDim_ > blockSize_*numBlocks_) {
        // user specified too many vectors... truncate
        // this produces a full subspace, but that is okay
        curDim_ = blockSize_*numBlocks_;
      }
      bool userand = false;
      if (curDim_ == 0) {
        // we need at least blockSize_ vectors
        // use a random multivec: ignore everything from InitVec
        userand = true;
        curDim_ = blockSize_;
      }

      // get pointers into V,KV,MV
      // tmpVecs will be used below for M*V and K*V (not simultaneously)
      // lclV has curDim vectors
      // if there is space for lclV and tmpVecs in V_, point tmpVecs into V_
      // otherwise, we must allocate space for these products
      //
      // get pointer to first curDim vector in V_
      std::vector<int> dimind(curDim_);
      for (int i=0; i<curDim_; ++i) dimind[i] = i;
      lclV = MVT::CloneView(*V_,dimind);
      if (userand) {
        // generate random vector data
        MVT::MvRandom(*lclV);
      }
      else {
        // assign ivec to first part of lclV
        MVT::SetBlock(*ivec,dimind,*lclV);
      }
      Teuchos::RCP<MV> tmpVecs; 
      if (curDim_*2 <= blockSize_*numBlocks_) {
        // partition V_ = [lclV tmpVecs _leftover_]
        std::vector<int> block2(curDim_);
        for (int i=0; i<curDim_; ++i) block2[i] = curDim_+i;
        tmpVecs = MVT::CloneView(*V_,block2);
      }
      else {
        // allocate space for tmpVecs
        tmpVecs = MVT::Clone(*V_,curDim_);
      }

      // compute M*lclV if hasM_
      if (hasM_) {
        Teuchos::TimeMonitor lcltimer( *timerMOp_ );
        OPT::Apply(*MOp_,*lclV,*tmpVecs);
        count_ApplyM_ += curDim_;
      }

      // remove auxVecs from lclV and normalize it
      if (auxVecs_.size() > 0) {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );

        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > dummy;
        int rank = orthman_->projectAndNormalizeMat(*lclV,tmpVecs,dummy,Teuchos::null,auxVecs_);
        TEST_FOR_EXCEPTION(rank != curDim_,BlockDavidsonInitFailure,
                           "Anasazi::BlockDavidson::initialize(): Couldn't generate initial basis of full rank.");
      }
      else {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );

        int rank = orthman_->normalizeMat(*lclV,tmpVecs,Teuchos::null);
        TEST_FOR_EXCEPTION(rank != curDim_,BlockDavidsonInitFailure,
                           "Anasazi::BlockDavidson::initialize(): Couldn't generate initial basis of full rank.");
      }

      // compute K*lclV: we are re-using tmpVecs to store the result
      {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply(*Op_,*lclV,*tmpVecs);
        count_ApplyOp_ += curDim_;
      }

      // generate KK
      lclKK = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::View,*KK_,curDim_,curDim_) );
      MVT::MvTransMv(ONE,*lclV,*tmpVecs,*lclKK);

      // clear tmpVecs
      tmpVecs = Teuchos::null;
    }

    // X,theta require Ritz analysis; if we have to generate one of these, we might as well generate both
    if (newstate.X != Teuchos::null && newstate.T != Teuchos::null) {
      TEST_FOR_EXCEPTION(MVT::GetNumberVecs(*newstate.X) != blockSize_ || MVT::GetVecLength(*newstate.X) != MVT::GetVecLength(*X_),
                          std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): Size of X must be consistent with block size and length of V.");
      TEST_FOR_EXCEPTION((signed int)(newstate.T->size()) != curDim_,
                          std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): Size of T must be consistent with dimension of V.");

      if (newstate.X != X_) {
        MVT::SetBlock(*newstate.X,bsind,*X_);
      }

      std::copy(newstate.T->begin(),newstate.T->end(),theta_.begin());
    }
    else {
      // compute ritz vecs/vals
      Teuchos::SerialDenseMatrix<int,ScalarType> S(curDim_,curDim_);
      {
        Teuchos::TimeMonitor lcltimer( *timerDS_ );
        int rank = curDim_;
        Utils::directSolver(curDim_, *lclKK, Teuchos::null, S, theta_, rank, 10);
        // we want all ritz values back
        TEST_FOR_EXCEPTION(rank != curDim_,BlockDavidsonInitFailure,
                           "Anasazi::BlockDavidson::initialize(newstate): Not enough Ritz vectors to initialize algorithm.");
      }
      // sort ritz pairs
      {
        Teuchos::TimeMonitor lcltimer( *timerSortEval_ );

        std::vector<int> order(curDim_);
        //
        // sort the first curDim_ values in theta_
        sm_->sort( this, curDim_, theta_, &order );   // don't catch exception
        //
        // apply the same ordering to the primitive ritz vectors
        Utils::permuteVectors(order,S);
      }

      // compute eigenvectors
      Teuchos::SerialDenseMatrix<int,ScalarType> S1(Teuchos::View,S,curDim_,blockSize_);
      {
        Teuchos::TimeMonitor lcltimer( *timerLocal_ );

        // X <- lclV*S
        MVT::MvTimesMatAddMv( ONE, *lclV, S1, ZERO, *X_ );
      }
      // we generated theta,X so we don't want to use the user's KX,MX
      newstate.KX = Teuchos::null;
      newstate.MX = Teuchos::null;
    }

    // done with local pointers
    lclV = Teuchos::null;
    lclKK = Teuchos::null;

    // set up KX
    if ( newstate.KX != Teuchos::null ) {
      TEST_FOR_EXCEPTION(MVT::GetNumberVecs(*newstate.KX) != blockSize_,
                          std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): vector length of newstate.KX not correct." );
      TEST_FOR_EXCEPTION(MVT::GetVecLength(*newstate.KX) != MVT::GetVecLength(*X_),
                          std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): newstate.KX must have at least block size vectors." );
      if (newstate.KX != KX_) {
        MVT::SetBlock(*newstate.KX,bsind,*KX_);
      }
    }
    else {
      // generate KX
      {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply(*Op_,*X_,*KX_);
        count_ApplyOp_ += blockSize_;
      }
      // we generated KX; we will generate R as well
      newstate.R = Teuchos::null;
    }

    // set up MX
    if (hasM_) {
      if ( newstate.MX != Teuchos::null ) {
        TEST_FOR_EXCEPTION(MVT::GetNumberVecs(*newstate.MX) != blockSize_,
                            std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): vector length of newstate.MX not correct." );
        TEST_FOR_EXCEPTION(MVT::GetVecLength(*newstate.MX) != MVT::GetVecLength(*X_),
                            std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): newstate.MX must have at least block size vectors." );
        if (newstate.MX != MX_) {
          MVT::SetBlock(*newstate.MX,bsind,*MX_);
        }
      }
      else {
        // generate MX
        {
          Teuchos::TimeMonitor lcltimer( *timerOp_ );
          OPT::Apply(*MOp_,*X_,*MX_);
          count_ApplyOp_ += blockSize_;
        }
        // we generated MX; we will generate R as well
        newstate.R = Teuchos::null;
      }
    }
    else {
      // the assignment MX_==X_ would be redundant; take advantage of this opportunity to debug a little
      TEST_FOR_EXCEPTION(MX_ != X_, std::logic_error, "Anasazi::BlockDavidson::initialize(): solver invariant not satisfied (MX==X).");
    }

    // set up R
    if (newstate.R != Teuchos::null) {
      TEST_FOR_EXCEPTION(MVT::GetNumberVecs(*newstate.R) != blockSize_,
                          std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): vector length of newstate.R not correct." );
      TEST_FOR_EXCEPTION(MVT::GetVecLength(*newstate.R) != MVT::GetVecLength(*X_),
                          std::invalid_argument, "Anasazi::BlockDavidson::initialize(newstate): newstate.R must have at least block size vectors." );
      if (newstate.R != R_) {
        MVT::SetBlock(*newstate.R,bsind,*R_);
      }
    }
    else {
      Teuchos::TimeMonitor lcltimer( *timerCompRes_ );
      
      // form R <- KX - MX*T
      MVT::MvAddMv(ZERO,*KX_,ONE,*KX_,*R_);
      Teuchos::SerialDenseMatrix<int,ScalarType> T(blockSize_,blockSize_);
      T.putScalar(ZERO);
      for (int i=0; i<blockSize_; ++i) T(i,i) = theta_[i];
      MVT::MvTimesMatAddMv(-ONE,*MX_,T,ONE,*R_);

    }

    // R has been updated; mark the norms as out-of-date
    Rnorms_current_ = false;
    R2norms_current_ = false;

    // finally, we are initialized
    initialized_ = true;

    if (om_->isVerbosity( Debug ) ) {
      // Check almost everything here
      CheckList chk;
      chk.checkV = true;
      chk.checkX = true;
      chk.checkKX = true;
      chk.checkMX = true;
      chk.checkR = true;
      chk.checkQ = true;
      chk.checkKK = true;
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
  void BlockDavidson<ScalarType,MV,OP>::initialize()
  {
    BlockDavidsonState<ScalarType,MV> empty;
    initialize(empty);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform BlockDavidson iterations until the StatusTest tells us to stop.
  template <class ScalarType, class MV, class OP>
  void BlockDavidson<ScalarType,MV,OP>::iterate ()
  {
    //
    // Initialize solver state
    if (initialized_ == false) {
      initialize();
    }

    // as a data member, this would be redundant and require synchronization with
    // blockSize_ and numBlocks_; we'll use a constant here.
    const int searchDim = blockSize_*numBlocks_;

    Teuchos::BLAS<int,ScalarType> blas;

    //
    // The projected matrices are part of the state, but the eigenvectors are defined locally.
    //    S = Local eigenvectors         (size: searchDim * searchDim
    Teuchos::SerialDenseMatrix<int,ScalarType> S( searchDim, searchDim );


    ////////////////////////////////////////////////////////////////
    // iterate until the status test tells us to stop.
    // also break if our basis is full
    while (tester_->checkStatus(this) != Passed && curDim_ < searchDim) {

      // Print information on current iteration
      if (om_->isVerbosity(Debug)) {
        currentStatus( om_->stream(Debug) );
      }
      else if (om_->isVerbosity(IterationDetails)) {
        currentStatus( om_->stream(IterationDetails) );
      }

      ++iter_;

      // get the current part of the basis
      std::vector<int> curind(blockSize_);
      for (int i=0; i<blockSize_; ++i) curind[i] = curDim_ + i;
      H_ = MVT::CloneView(*V_,curind);
      
      // Apply the preconditioner on the residuals: H <- Prec*R
      // H = Prec*R
      if (Prec_ != Teuchos::null) {
        Teuchos::TimeMonitor lcltimer( *timerPrec_ );
        OPT::Apply( *Prec_, *R_, *H_ );   // don't catch the exception
        count_ApplyPrec_ += blockSize_;
      }
      else {
        std::vector<int> bsind(blockSize_);
        for (int i=0; i<blockSize_; ++i) { bsind[i] = i; }
        MVT::SetBlock(*R_,bsind,*H_);
      }

      // Apply the mass matrix on H
      if (hasM_) {
        // use memory at MX_ for temporary storage
        MH_ = MX_;
        Teuchos::TimeMonitor lcltimer( *timerMOp_ );
        OPT::Apply( *MOp_, *H_, *MH_);    // don't catch the exception
        count_ApplyM_ += blockSize_;
      }
      else  {
        MH_ = H_;
      }

      // Get a view of the previous vectors
      // this is used for orthogonalization and for computing V^H K H
      std::vector<int> prevind(curDim_);
      for (int i=0; i<curDim_; ++i) prevind[i] = i;
      Teuchos::RCP<MV> Vprev = MVT::CloneView(*V_,prevind);

      // Orthogonalize H against the previous vectors and the auxiliary vectors, and normalize
      {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );

        Teuchos::Array<Teuchos::RCP<const MV> > against = auxVecs_;
        against.push_back(Vprev);
        int rank = orthman_->projectAndNormalizeMat(*H_,MH_,
                                            Teuchos::tuple<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > >(Teuchos::null),
                                            Teuchos::null,against);
        TEST_FOR_EXCEPTION(rank != blockSize_,BlockDavidsonOrthoFailure,
                           "Anasazi::BlockDavidson::iterate(): unable to compute orthonormal basis for H.");
      }

      // Apply the stiffness matrix to H
      {
        // use memory at KX_ for temporary storage
        KH_ = KX_;
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply( *Op_, *H_, *KH_);    // don't catch the exception
        count_ApplyOp_ += blockSize_;
      }

      if (om_->isVerbosity( Debug ) ) {
        CheckList chk;
        chk.checkH = true;
        chk.checkMH = true;
        chk.checkKH = true;
        om_->print( Debug, accuracyCheck(chk, ": after ortho H") );
      }
      else if (om_->isVerbosity( OrthoDetails ) ) {
        CheckList chk;
        chk.checkH = true;
        chk.checkMH = true;
        chk.checkKH = true;
        om_->print( OrthoDetails, accuracyCheck(chk,": after ortho H") );
      }

      // compute next part of the projected matrices
      // this this in two parts
      Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > nextKK;
      // Vprev*K*H
      nextKK = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::View,*KK_,curDim_,blockSize_,0,curDim_) );
      MVT::MvTransMv(ONE,*Vprev,*KH_,*nextKK);
      // H*K*H
      nextKK = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::View,*KK_,blockSize_,blockSize_,curDim_,curDim_) );
      MVT::MvTransMv(ONE,*H_,*KH_,*nextKK);
      // 
      // make sure that KK_ is Hermitian in memory
      nextKK = Teuchos::null;
      for (int i=curDim_; i<curDim_+blockSize_; ++i) {
        for (int j=0; j<i; ++j) {
          (*KK_)(i,j) = SCT::conjugate((*KK_)(j,i));
        }
      }

      // V has been extended, and KK has been extended. Update basis dim and release all pointers.
      curDim_ += blockSize_;
      H_ = KH_ = MH_ = Teuchos::null;
      Vprev = Teuchos::null;

      if (om_->isVerbosity( Debug ) ) {
        CheckList chk;
        chk.checkKK = true;
        om_->print( Debug, accuracyCheck(chk, ": after expanding KK") );
      }

      // Get pointer to complete basis
      curind.resize(curDim_);
      for (int i=0; i<curDim_; ++i) curind[i] = i;
      Teuchos::RCP<const MV> curV = MVT::CloneView(*V_,curind);

      // Perform spectral decomposition
      {
        Teuchos::TimeMonitor lcltimer(*timerDS_);
        int nevlocal = curDim_;
        int info = Utils::directSolver(curDim_,*KK_,Teuchos::null,S,theta_,nevlocal,10);
        TEST_FOR_EXCEPTION(info != 0,std::logic_error,"Anasazi::BlockDavidson::iterate(): direct solve returned error code.");
        // we did not ask directSolver to perform deflation, so nevLocal better be curDim_
        TEST_FOR_EXCEPTION(nevlocal != curDim_,std::logic_error,"Anasazi::BlockDavidson::iterate(): direct solve did not compute all eigenvectors."); // this should never happen
      }

      // Sort ritz pairs
      { 
        Teuchos::TimeMonitor lcltimer( *timerSortEval_ );

        std::vector<int> order(curDim_);
        // 
        // sort the first curDim_ values in theta_
        sm_->sort( this, curDim_, theta_, &order );   // don't catch exception
        //
        // apply the same ordering to the primitive ritz vectors
        Teuchos::SerialDenseMatrix<int,ScalarType> curS(Teuchos::View,S,curDim_,curDim_);
        Utils::permuteVectors(order,curS);
      }

      // Create a view matrix of the first blockSize_ vectors
      Teuchos::SerialDenseMatrix<int,ScalarType> S1( Teuchos::View, S, curDim_, blockSize_ );

      // Compute the new Ritz vectors
      {
        Teuchos::TimeMonitor lcltimer( *timerLocal_ );
        MVT::MvTimesMatAddMv(ONE,*curV,S1,ZERO,*X_);
      }

      // Apply the stiffness matrix for the Ritz vectors
      {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply( *Op_, *X_, *KX_);    // don't catch the exception
        count_ApplyOp_ += blockSize_;
      }
      // Apply the mass matrix for the Ritz vectors
      if (hasM_) {
        Teuchos::TimeMonitor lcltimer( *timerMOp_ );
        OPT::Apply(*MOp_,*X_,*MX_);
        count_ApplyM_ += blockSize_;
      }
      else {
        MX_ = X_;
      }

      // Compute the residual
      // R = KX - MX*diag(theta)
      {
        Teuchos::TimeMonitor lcltimer( *timerCompRes_ );
        
        MVT::MvAddMv( ONE, *KX_, ZERO, *KX_, *R_ );
        Teuchos::SerialDenseMatrix<int,ScalarType> T( blockSize_, blockSize_ );
        for (int i = 0; i < blockSize_; ++i) {
          T(i,i) = theta_[i];
        }
        MVT::MvTimesMatAddMv( -ONE, *MX_, T, ONE, *R_ );
      }

      // R has been updated; mark the norms as out-of-date
      Rnorms_current_ = false;
      R2norms_current_ = false;


      // When required, monitor some orthogonalities
      if (om_->isVerbosity( Debug ) ) {
        // Check almost everything here
        CheckList chk;
        chk.checkV = true;
        chk.checkX = true;
        chk.checkKX = true;
        chk.checkMX = true;
        chk.checkR = true;
        om_->print( Debug, accuracyCheck(chk, ": after local update") );
      }
      else if (om_->isVerbosity( OrthoDetails )) {
        CheckList chk;
        chk.checkX = true;
        chk.checkKX = true;
        chk.checkMX = true;
        chk.checkR = true;
        om_->print( OrthoDetails, accuracyCheck(chk, ": after local update") );
      }
    } // end while (statusTest == false)

  } // end of iterate()



  //////////////////////////////////////////////////////////////////////////////////////////////////
  // compute/return residual M-norms
  template <class ScalarType, class MV, class OP>
  std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> 
  BlockDavidson<ScalarType,MV,OP>::getResNorms() {
    if (Rnorms_current_ == false) {
      // Update the residual norms
      orthman_->norm(*R_,&Rnorms_);
      Rnorms_current_ = true;
    }
    return Rnorms_;
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // compute/return residual 2-norms
  template <class ScalarType, class MV, class OP>
  std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> 
  BlockDavidson<ScalarType,MV,OP>::getRes2Norms() {
    if (R2norms_current_ == false) {
      // Update the residual 2-norms 
      MVT::MvNorm(*R_,&R2norms_);
      R2norms_current_ = true;
    }
    return R2norms_;
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Check accuracy, orthogonality, and other debugging stuff
  // 
  // bools specify which tests we want to run (instead of running more than we actually care about)
  //
  // we don't bother checking the following because they are computed explicitly:
  //    H == Prec*R
  //   KH == K*H
  //
  // 
  // checkV : V orthonormal
  //          orthogonal to auxvecs
  // checkX : X orthonormal
  //          orthogonal to auxvecs
  // checkMX: check MX == M*X
  // checkKX: check KX == K*X
  // checkH : H orthonormal 
  //          orthogonal to V and H and auxvecs
  // checkMH: check MH == M*H
  // checkR : check R orthogonal to X
  // checkQ : check that auxiliary vectors are actually orthonormal
  // checkKK: check that KK is symmetric in memory 
  //
  // TODO: 
  //  add checkTheta 
  //
  template <class ScalarType, class MV, class OP>
  std::string BlockDavidson<ScalarType,MV,OP>::accuracyCheck( const CheckList &chk, const std::string &where ) const 
  {
    using std::endl;

    std::stringstream os;
    os.precision(2);
    os.setf(std::ios::scientific, std::ios::floatfield);
    MagnitudeType tmp;

    os << " Debugging checks: iteration " << iter_ << where << endl;

    // V and friends
    std::vector<int> lclind(curDim_);
    for (int i=0; i<curDim_; ++i) lclind[i] = i;
    Teuchos::RCP<MV> lclV,lclKV;
    if (initialized_) {
      lclV = MVT::CloneView(*V_,lclind);
    }
    if (chk.checkV && initialized_) {
      tmp = orthman_->orthonormError(*lclV);
      os << " >> Error in V^H M V == I  : " << tmp << endl;
      for (unsigned int i=0; i<auxVecs_.size(); ++i) {
        tmp = orthman_->orthogError(*lclV,*auxVecs_[i]);
        os << " >> Error in V^H M Q[" << i << "] == 0 : " << tmp << endl;
      }
      Teuchos::SerialDenseMatrix<int,ScalarType> curKK(curDim_,curDim_);
      Teuchos::RCP<MV> lclKV = MVT::Clone(*V_,curDim_);
      OPT::Apply(*Op_,*lclV,*lclKV);
      MVT::MvTransMv(ONE,*lclV,*lclKV,curKK);
      Teuchos::SerialDenseMatrix<int,ScalarType> subKK(Teuchos::View,*KK_,curDim_,curDim_);
      curKK -= subKK;
      // dup the lower tri part
      for (int j=0; j<curDim_; ++j) {
        for (int i=j+1; i<curDim_; ++i) {
          curKK(i,j) = curKK(j,i);
        }
      }
      os << " >> Error in V^H K V == KK : " << curKK.normFrobenius() << endl;
    }

    // X and friends
    if (chk.checkX && initialized_) {
      tmp = orthman_->orthonormError(*X_);
      os << " >> Error in X^H M X == I  : " << tmp << endl;
      for (unsigned int i=0; i<auxVecs_.size(); ++i) {
        tmp = orthman_->orthogError(*X_,*auxVecs_[i]);
        os << " >> Error in X^H M Q[" << i << "] == 0 : " << tmp << endl;
      }
    }
    if (chk.checkMX && hasM_ && initialized_) {
      tmp = Utils::errorEquality(*X_, *MX_, MOp_);
      os << " >> Error in MX == M*X     : " << tmp << endl;
    }
    if (chk.checkKX && initialized_) {
      tmp = Utils::errorEquality(*X_, *KX_, Op_);
      os << " >> Error in KX == K*X     : " << tmp << endl;
    }

    // H and friends
    if (chk.checkH && initialized_) {
      tmp = orthman_->orthonormError(*H_);
      os << " >> Error in H^H M H == I  : " << tmp << endl;
      tmp = orthman_->orthogError(*H_,*lclV);
      os << " >> Error in H^H M V == 0  : " << tmp << endl;
      tmp = orthman_->orthogError(*H_,*X_);
      os << " >> Error in H^H M X == 0  : " << tmp << endl;
      for (unsigned int i=0; i<auxVecs_.size(); ++i) {
        tmp = orthman_->orthogError(*H_,*auxVecs_[i]);
        os << " >> Error in H^H M Q[" << i << "] == 0 : " << tmp << endl;
      }
    }
    if (chk.checkKH && initialized_) {
      tmp = Utils::errorEquality(*H_, *KH_, Op_);
      os << " >> Error in KH == K*H     : " << tmp << endl;
    }
    if (chk.checkMH && hasM_ && initialized_) {
      tmp = Utils::errorEquality(*H_, *MH_, MOp_);
      os << " >> Error in MH == M*H     : " << tmp << endl;
    }

    // R: this is not M-orthogonality, but standard Euclidean orthogonality
    if (chk.checkR && initialized_) {
      Teuchos::SerialDenseMatrix<int,ScalarType> xTx(blockSize_,blockSize_);
      MVT::MvTransMv(ONE,*X_,*R_,xTx);
      tmp = xTx.normFrobenius();
      os << " >> Error in X^H R == 0    : " << tmp << endl;
    }

    // KK
    if (chk.checkKK && initialized_) {
      Teuchos::SerialDenseMatrix<int,ScalarType> tmp(curDim_,curDim_), lclKK(Teuchos::View,*KK_,curDim_,curDim_);
      for (int j=0; j<curDim_; ++j) {
        for (int i=0; i<curDim_; ++i) {
          tmp(i,j) = lclKK(i,j) - SCT::conjugate(lclKK(j,i));
        }
      }
      os << " >> Error in KK - KK^H == 0 : " << tmp.normFrobenius() << endl;
    }

    // Q
    if (chk.checkQ) {
      for (unsigned int i=0; i<auxVecs_.size(); ++i) {
        tmp = orthman_->orthonormError(*auxVecs_[i]);
        os << " >> Error in Q[" << i << "]^H M Q[" << i << "] == I : " << tmp << endl;
        for (unsigned int j=i+1; j<auxVecs_.size(); ++j) {
          tmp = orthman_->orthogError(*auxVecs_[i],*auxVecs_[j]);
          os << " >> Error in Q[" << i << "]^H M Q[" << j << "] == 0 : " << tmp << endl;
        }
      }
    }

    os << endl;

    return os.str();
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Print the current status of the solver
  template <class ScalarType, class MV, class OP>
  void 
  BlockDavidson<ScalarType,MV,OP>::currentStatus(std::ostream &os) 
  {
    using std::endl;

    os.setf(std::ios::scientific, std::ios::floatfield);
    os.precision(6);
    os <<endl;
    os <<"================================================================================" << endl;
    os << endl;
    os <<"                          BlockDavidson Solver Status" << endl;
    os << endl;
    os <<"The solver is "<<(initialized_ ? "initialized." : "not initialized.") << endl;
    os <<"The number of iterations performed is " <<iter_<<endl;
    os <<"The block size is         " << blockSize_<<endl;
    os <<"The number of blocks is   " << numBlocks_<<endl;
    os <<"The current basis size is " << curDim_<<endl;
    os <<"The number of auxiliary vectors is    " << numAuxVecs_ << endl;
    os <<"The number of operations Op*x   is "<<count_ApplyOp_<<endl;
    os <<"The number of operations M*x    is "<<count_ApplyM_<<endl;
    os <<"The number of operations Prec*x is "<<count_ApplyPrec_<<endl;

    os.setf(std::ios_base::right, std::ios_base::adjustfield);

    if (initialized_) {
      os << endl;
      os <<"CURRENT EIGENVALUE ESTIMATES             "<<endl;
      os << std::setw(20) << "Eigenvalue" 
         << std::setw(20) << "Residual(M)"
         << std::setw(20) << "Residual(2)"
         << endl;
      os <<"--------------------------------------------------------------------------------"<<endl;
      for (int i=0; i<blockSize_; ++i) {
        os << std::setw(20) << theta_[i];
        if (Rnorms_current_) os << std::setw(20) << Rnorms_[i];
        else os << std::setw(20) << "not current";
        if (R2norms_current_) os << std::setw(20) << R2norms_[i];
        else os << std::setw(20) << "not current";
        os << endl;
      }
    }
    os <<"================================================================================" << endl;
    os << endl;
  }
  
} // End of namespace Anasazi

#endif

// End of file AnasaziBlockDavidson.hpp
