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


/*! \file AnasaziLOBPCG.hpp
  \brief Implementation of the locally-optimal block preconditioned conjugate gradient (LOBPCG) method
*/

/*
    LOBPCG contains local storage of up to 10*blockSize_ vectors, representing 10 entities
      X,H,P,R
      KX,KH,KP  (product of K and the above)
      MX,MH,MP  (product of M and the above, not allocated if we don't have an M matrix)
    If full orthogonalization is enabled, one extra multivector of blockSize_ vectors is required to 
    compute the local update of X and P.
    
    A solver is bound to an eigenproblem at declaration.
    Other solver parameters (e.g., block size, auxiliary vectors) can be changed dynamically.
    
    The orthogonalization manager is used to project away from the auxiliary vectors.
    If full orthogonalization is enabled, the orthogonalization manager is also used to construct an M orthonormal basis.
    The orthogonalization manager is subclass of MatOrthoManager, which LOBPCG assumes to be defined by the M inner product.
    LOBPCG will not work correctly if the orthomanager uses a different inner product.
 */


#ifndef ANASAZI_LOBPCG_HPP
#define ANASAZI_LOBPCG_HPP

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

/*!     \class Anasazi::LOBPCG

        \brief This class provides the Locally Optimal Block Preconditioned Conjugate Gradient (%LOBPCG) iteration, a preconditioned iteration for solving linear Hermitian eigenproblems.

        This implementation is a modification of the one found in 
        A. Knyazev, "Toward the optimal preconditioned eigensolver:
        Locally optimal block preconditioner conjugate gradient method",
        SIAM J. Sci. Comput., vol 23, n 2, pp. 517-541.

        The modification consists of the orthogonalization steps recommended in
        U. Hetmaniuk and R. Lehoucq, "Basis Selection in LOBPCG", Journal of Computational Physics. 

        These modifcation are referred to as full orthogonalization, and consist of also conducting
        the local optimization using an orthonormal basis.

        \ingroup anasazi_solver_framework

        \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, Heidi Thornquist
*/

namespace Anasazi {

  //! @name LOBPCG Structures
  //@{ 

  /** \brief Structure to contain pointers to Anasazi state variables.
   *
   * This struct is utilized by LOBPCG::initialize() and LOBPCG::getState().
   */
  template <class ScalarType, class MultiVector>
  struct LOBPCGState {
    //! The current test basis.
    Teuchos::RCP<const MultiVector> V; 
    //! The image of the current test basis under K.
    Teuchos::RCP<const MultiVector> KV; 
    //! The image of the current test basis under M, or Teuchos::null if M was not specified.
    Teuchos::RCP<const MultiVector> MV;

    //! The current eigenvectors.
    Teuchos::RCP<const MultiVector> X; 
    //! The image of the current eigenvectors under K.
    Teuchos::RCP<const MultiVector> KX; 
    //! The image of the current eigenvectors under M, or Teuchos::null if M was not specified.
    Teuchos::RCP<const MultiVector> MX;

    //! The current search direction.
    Teuchos::RCP<const MultiVector> P; 
    //! The image of the current search direction under K.
    Teuchos::RCP<const MultiVector> KP; 
    //! The image of the current search direction under M, or Teuchos::null if M was not specified.
    Teuchos::RCP<const MultiVector> MP;

    /*! \brief The current preconditioned residual vectors.
     *
     *  H is only useful when LOBPCG::iterate() throw a LOBPCGRitzFailure exception.
     */
    Teuchos::RCP<const MultiVector> H; 
    //! The image of the current preconditioned residual vectors under K.
    Teuchos::RCP<const MultiVector> KH; 
    //! The image of the current preconditioned residual vectors under M, or Teuchos::null if M was not specified.
    Teuchos::RCP<const MultiVector> MH;

    //! The current residual vectors.
    Teuchos::RCP<const MultiVector> R;

    //! The current Ritz values.
    Teuchos::RCP<const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> > T;

    LOBPCGState() : 
      V(Teuchos::null),KV(Teuchos::null),MV(Teuchos::null),
      X(Teuchos::null),KX(Teuchos::null),MX(Teuchos::null),
      P(Teuchos::null),KP(Teuchos::null),MP(Teuchos::null),
      H(Teuchos::null),KH(Teuchos::null),MH(Teuchos::null),
      R(Teuchos::null),T(Teuchos::null) {};
  };

  //@}

  //! @name LOBPCG Exceptions
  //@{ 

  /** \brief LOBPCGRitzFailure is thrown when the LOBPCG solver is unable to
   *  continue a call to LOBPCG::iterate() due to a failure of the algorithm.
   *
   *  This signals that the Rayleigh-Ritz analysis over the subspace \c
   *  colsp([X H P]) detected ill-conditioning of the projected mass matrix
   *  and the inability to generate a set of orthogonal eigenvectors for 
   *  the projected problem.
   *
   *  This exception is only thrown from the LOBPCG::iterate() routine. After
   *  catching this exception, the user can recover the subspace via
   *  LOBPCG::getState(). This information can be used to restart the solver.
   *
   */
  class LOBPCGRitzFailure : public AnasaziError {public:
    LOBPCGRitzFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  /** \brief LOBPCGInitFailure is thrown when the LOBPCG solver is unable to
   * generate an initial iterate in the LOBPCG::initialize() routine. 
   *
   * This exception is thrown from the LOBPCG::initialize() method, which is
   * called by the user or from the LOBPCG::iterate() method when isInitialized()
   * == \c false.
   *
   * In the case that this exception is thrown, LOBPCG::hasP() and
   * LOBPCG::isInitialized() will be \c false and the user will need to provide
   * a new initial iterate to the solver.
   *
   */
  class LOBPCGInitFailure : public AnasaziError {public:
    LOBPCGInitFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  /** \brief LOBPCGOrthoFailure is thrown when an orthogonalization attempt 
   * fails.
   *
   * This is thrown in one of two scenarstd::ios. After preconditioning the residual,
   * the orthogonalization manager is asked to orthogonalize the preconditioned
   * residual (H) against the auxiliary vectors. If full orthogonalization
   * is enabled, H is also orthogonalized against X and P and normalized.
   *
   * The second scenario involves the generation of new X and P from the
   * basis [X H P]. When full orthogonalization is enabled, an attempt is
   * made to select coefficients for X and P so that they will be
   * mutually orthogonal and orthonormal.
   *
   * If either of these attempts fail, the solver throws an LOBPCGOrthoFailure
   * exception.
   */
  class LOBPCGOrthoFailure : public AnasaziError {public:
    LOBPCGOrthoFailure(const std::string& what_arg) : AnasaziError(what_arg)
    {}};

  //@}


  template <class ScalarType, class MV, class OP>
  class LOBPCG : public Eigensolver<ScalarType,MV,OP> { 
  public:
    
    //! @name Constructor/Destructor
    //@{ 
    
    /*! \brief %LOBPCG constructor with eigenproblem, solver utilities, and parameter list of solver options.
     *
     * This constructor takes pointers required by the eigensolver, in addition
     * to a parameter list of options for the eigensolver. These options include the following:
     *   - "Block Size" - an \c int specifying the block size used by the algorithm. This can also be specified using the setBlockSize() method.
     *   - "Full Ortho" - a \c bool specifying whether the solver should employ a full orthogonalization technique. This can also be specified using the setFullOrtho() method.
     */
    LOBPCG( const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem, 
            const Teuchos::RCP<SortManager<ScalarType,MV,OP> > &sorter,
            const Teuchos::RCP<OutputManager<ScalarType> > &printer,
            const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &tester,
            const Teuchos::RCP<MatOrthoManager<ScalarType,MV,OP> > &ortho,
            Teuchos::ParameterList &params 
          );
    
    //! %LOBPCG destructor
    virtual ~LOBPCG() {};

    //@}

    //! @name Solver methods
    //@{

    /*! \brief This method performs %LOBPCG iterations until the status test
     * indicates the need to stop or an error occurs (in which case, an
     * exception is thrown).
     *
     * iterate() will first determine whether the solver is initialized; if
     * not, it will call initialize() using default arguments.  After
     * initialization, the solver performs %LOBPCG iterations until the status
     * test evaluates as Passed, at which point the method returns to the
     * caller.
     *
     * The %LOBPCG iteration proceeds as follows:
     * -# The current residual (R) is preconditioned to form H
     * -# H is orthogonalized against the auxiliary vectors and, if full orthogonalization\n
     *    is enabled, against X and P. 
     * -# The basis [X H P] is used to project the problem matrices.
     * -# The projected eigenproblem is solved, and the desired eigenvectors and eigenvalues are selected.
     * -# These are used to form the new eigenvector estimates (X) and the search directions (P).\n
     *    If full orthogonalization is enabled, these are generated to be mutually orthogonal and with orthonormal columns.
     * -# The new residual (R) is formed.
     *
     * The status test is queried at the beginning of the iteration.
     *
     * Possible exceptions thrown include std::logic_error, std::invalid_argument or
     * one of the LOBPCG-specific exceptions.
     *
    */
    void iterate();

    /*! \brief Initialize the solver to an iterate, optionally providing the
     * Ritz values, residual, and search direction.
     *
     * \note LOBPCGState contains fields V, KV and MV: These are ignored by initialize()
     *
     * The %LOBPCG eigensolver contains a certain amount of state relating to
     * the current iterate, including the current residual, the current search
     * direction, and the images of these spaces under the eigenproblem operators.
     *
     * initialize() gives the user the opportunity to manually set these,
     * although this must be done with caution, abiding by the rules
     * given below. All notions of orthogonality and orthonormality are derived
     * from the inner product specified by the orthogonalization manager.
     *
     * \post 
     *   - isInitialized() == true (see post-conditions of isInitialize())
     *   - If newstate.P != Teuchos::null, hasP() == true.\n
     *     Otherwise, hasP() == false
     *
     * The user has the option of specifying any component of the state using
     * initialize(). However, these arguments are assumed to match the
     * post-conditions specified under isInitialized(). Any component of the
     * state (i.e., KX) not given to initialize() will be generated.
     *
     */
    void initialize(LOBPCGState<ScalarType,MV> newstate);

    /*! \brief Initialize the solver with the initial vectors from the eigenproblem
     *  or random data.
     */
    void initialize();

    /*! \brief Indicates whether the solver has been initialized or not.
     *
     * \return bool indicating the state of the solver.
     * \post
     * If isInitialized() == \c true:
     *   - X is orthogonal to auxiliary vectors and has orthonormal columns
     *   - KX == Op*X
     *   - MX == M*X if M != Teuchos::null\n
     *     Otherwise, MX == Teuchos::null
     *   - getRitzValues() returns the sorted Ritz values with respect to X
     *   - getResNorms(), getRes2Norms(), getRitzResNorms() are correct
     *   - If hasP() == \c true,
     *      - P orthogonal to auxiliary vectors
     *      - If getFullOrtho() == \c true,
     *        - P is orthogonal to X and has orthonormal columns
     *      - KP == Op*P
     *      - MP == M*P if M != Teuchos::null\n
     *        Otherwise, MP == Teuchos::null
     */
    bool isInitialized() const;

    /*! \brief Get the current state of the eigensolver.
     * 
     * The data is only valid if isInitialized() == \c true. The
     * data for the search directions P is only meaningful if hasP() == \c
     * true. Finally, the data for the preconditioned residual (H) is only meaningful in the situation where
     * the solver throws an ::LOBPCGRitzFailure exception during iterate().
     *
     * \returns An LOBPCGState object containing const views to the current
     * solver state.
     */
    LOBPCGState<ScalarType,MV> getState() const;

    //@}

    //! @name Status methods
    //@{

    //! \brief Get the current iteration count.
    int getNumIters() const;

    //! \brief Reset the iteration count.
    void resetNumIters();

    /*! \brief Get the Ritz vectors from the previous iteration.
      
        \return A multivector with getBlockSize() vectors containing 
        the sorted Ritz vectors corresponding to the most significant Ritz values.
        The i-th vector of the return corresponds to the i-th Ritz vector; there is no need to use
        getRitzIndex().
     */
    Teuchos::RCP<const MV> getRitzVectors();

    /*! \brief Get the Ritz values from the previous iteration.
     *
     *  \return A vector of length getCurSubspaceDim() containing the Ritz values from the
     *  previous projected eigensolve.
     */
    std::vector<Value<ScalarType> > getRitzValues();

    /*! \brief Get the index used for extracting Ritz vectors from getRitzVectors().
     *
     * Because BlockDavidson is a Hermitian solver, all Ritz values are real and all Ritz vectors can be represented in a 
     * single column of a multivector. Therefore, getRitzIndex() is not needed when using the output from getRitzVectors().
     *
     * \return An \c int vector of size getCurSubspaceDim() composed of zeros.
     */
    std::vector<int> getRitzIndex();


    /*! \brief Get the current residual norms
     *
     *  \return A vector of length getBlockSize() containing the norms of the
     *  residuals, with respect to the orthogonalization manager norm() method.
     */
    std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> getResNorms();


    /*! \brief Get the current residual 2-norms
     *
     *  \return A vector of length getBlockSize() containing the 2-norms of the
     *  residuals. 
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
     *  %LOBPCG employs a sequential subspace iteration, maintaining a fixed-rank basis, as opposed to an expanding subspace
     *  mechanism employed by Krylov-subspace solvers like BlockKrylovSchur and BlockDavidson.
     *  
     *  \return An integer specifying the rank of the subspace generated by the eigensolver. If isInitialized() == \c false, 
     *  the return is 0. Otherwise, the return will be 2*getBlockSize() or 3*getBlockSize().
     */
    int getCurSubspaceDim() const;

    /*! \brief Get the maximum dimension allocated for the search subspace. For %LOBPCG, this always returns 3*getBlockSize(), the dimension of the 
     *   subspace colspan([X H P]).
     */
    int getMaxSubspaceDim() const;

    //@}

    //!  @name Accessor routines from Eigensolver
    //@{


    //! Get a constant reference to the eigenvalue problem.
    const Eigenproblem<ScalarType,MV,OP>& getProblem() const;


    /*! \brief Set the blocksize to be used by the iterative solver in solving
     * this eigenproblem.
     *  
     *  If the block size is reduced, then the new iterate (and residual and
     *  search direction) are chosen as the subset of the current iterate
     *  preferred by the sort manager.  Otherwise, the solver state is set to
     *  uninitialized.
     */
    void setBlockSize(int blockSize);


    //! Get the blocksize to be used by the iterative solver in solving this eigenproblem.
    int getBlockSize() const;


    /*! \brief Set the auxiliary vectors for the solver.
     *
     *  Because the current iterate X and search direction P cannot be assumed
     *  orthogonal to the new auxiliary vectors, a call to setAuxVecs() with a
     *  non-empty argument will reset the solver to the uninitialized state.
     *
     *  In order to preserve the current state, the user will need to extract
     *  it from the solver using getState(), orthogonalize it against the new
     *  auxiliary vectors, and manually reinitialize the solver using
     *  initialize().
     */
    void setAuxVecs(const Teuchos::Array<Teuchos::RCP<const MV> > &auxvecs);

    //! Get the current auxiliary vectors.
    Teuchos::Array<Teuchos::RCP<const MV> > getAuxVecs() const;

    //@}

    //!  @name %LOBPCG-specific accessor routines
    //@{

    /*! \brief Instruct the LOBPCG iteration to use full orthogonality.
     *
     *  If the getFullOrtho() == \c false and isInitialized() == \c true and hasP() == \c true, then
     *  P will be invalidated by setting full orthogonalization to \c true.
     */
    void setFullOrtho(bool fullOrtho);

    //! Determine if the LOBPCG iteration is using full orthogonality.
    bool getFullOrtho() const;
    
    //! Indicates whether the search direction given by getState() is valid.
    bool hasP();

    //@}
    
    //!  @name Output methods
    //@{

    //! This method requests that the solver print out its current status to screen.
    void currentStatus(std::ostream &os);

    //@}

  private:
    //
    //
    //
    void setupViews();
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
      bool checkX, checkMX, checkKX;
      bool checkH, checkMH;
      bool checkP, checkMP, checkKP;
      bool checkR, checkQ;
      CheckList() : checkX(false),checkMX(false),checkKX(false),
                    checkH(false),checkMH(false),
                    checkP(false),checkMP(false),checkKP(false),
                    checkR(false),checkQ(false) {};
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
                                timerSort_, 
                                timerLocalProj_, timerDS_,
                                timerLocalUpdate_, timerCompRes_,
                                timerOrtho_, timerInit_;
    //
    // Counters
    //
    // Number of operator applications
    int count_ApplyOp_, count_ApplyM_, count_ApplyPrec_;

    //
    // Algorithmic parameters.
    //
    // blockSize_ is the solver block size
    int blockSize_;
    //
    // fullOrtho_ dictates whether the orthogonalization procedures specified by Hetmaniuk and Lehoucq should
    // be activated (see citations at the top of this file)
    bool fullOrtho_;

    //
    // Current solver state
    //
    // initialized_ specifies that the basis vectors have been initialized and the iterate() routine
    // is capable of running; _initialize is controlled  by the initialize() member method
    // For the implications of the state of initialized_, please see documentation for initialize()
    bool initialized_;
    //
    // nevLocal_ reflects how much of the current basis is valid (0 <= nevLocal_ <= 3*blockSize_)
    // this tells us how many of the values in theta_ are valid Ritz values
    int nevLocal_;
    //
    // hasP_ tells us whether there is valid data in P (and KP,MP)
    bool hasP_;
    //
    // State Multivecs
    // V_, KV_ MV_  and  R_ are primary pointers to allocated multivectors
    // the rest are multivector views into V_, KV_ and MV_
    Teuchos::RCP<MV> V_, KV_, MV_, R_;
    Teuchos::RCP<MV> X_, KX_, MX_,
                     H_, KH_, MH_,
                     P_, KP_, MP_;

    //
    // if fullOrtho_ == true, then we must produce the following on every iteration:
    // [newX newP] = [X H P] [CX;CP]
    // the structure of [CX;CP] when using full orthogonalization does not allow us to 
    // do this in situ, and R_ does not have enough storage for newX and newP. therefore, 
    // we must allocate additional storage for this.
    // otherwise, when not using full orthogonalization, the structure
    // [newX newP] = [X H P] [CX1  0 ]
    //                       [CX2 CP2]  allows us to work using only R as work space
    //                       [CX3 CP3] 
    Teuchos::RCP<MV> tmpmvec_;        
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
  // Constructor
  template <class ScalarType, class MV, class OP>
  LOBPCG<ScalarType,MV,OP>::LOBPCG(
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
    timerSort_(Teuchos::TimeMonitor::getNewTimer("Sorting eigenvalues")),
    timerLocalProj_(Teuchos::TimeMonitor::getNewTimer("Local projection")),
    timerDS_(Teuchos::TimeMonitor::getNewTimer("Direct solve")),
    timerLocalUpdate_(Teuchos::TimeMonitor::getNewTimer("Local update")),
    timerCompRes_(Teuchos::TimeMonitor::getNewTimer("Computing residuals")),
    timerOrtho_(Teuchos::TimeMonitor::getNewTimer("Orthogonalization")),
    timerInit_(Teuchos::TimeMonitor::getNewTimer("Initialization")),
    count_ApplyOp_(0),
    count_ApplyM_(0),
    count_ApplyPrec_(0),
    // internal data
    blockSize_(0),
    fullOrtho_(params.get("Full Ortho", true)),
    initialized_(false),
    nevLocal_(0),
    hasP_(false),
    auxVecs_( Teuchos::Array<Teuchos::RCP<const MV> >(0) ), 
    numAuxVecs_(0),
    iter_(0),
    Rnorms_current_(false),
    R2norms_current_(false)
  {     
    TEST_FOR_EXCEPTION(problem_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: user passed null problem pointer.");
    TEST_FOR_EXCEPTION(sm_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: user passed null sort manager pointer.");
    TEST_FOR_EXCEPTION(om_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: user passed null output manager pointer.");
    TEST_FOR_EXCEPTION(tester_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: user passed null status test pointer.");
    TEST_FOR_EXCEPTION(orthman_ == Teuchos::null,std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: user passed null orthogonalization manager pointer.");
    TEST_FOR_EXCEPTION(problem_->isProblemSet() == false, std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: problem is not set.");
    TEST_FOR_EXCEPTION(problem_->isHermitian() == false, std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: problem is not Hermitian; LOBPCG requires Hermitian problem.");

    // get the problem operators
    Op_   = problem_->getOperator();
    TEST_FOR_EXCEPTION(Op_ == Teuchos::null, std::invalid_argument,
                       "Anasazi::LOBPCG::constructor: problem provides no operator.");
    MOp_  = problem_->getM();
    Prec_ = problem_->getPrec();
    hasM_ = (MOp_ != Teuchos::null);

    // set the block size and allocate data
    int bs = params.get("Block Size", problem_->getNEV());
    setBlockSize(bs);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the block size and make necessary adjustments.
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::setBlockSize (int newBS) 
  {
    // time spent here counts towards timerInit_
    Teuchos::TimeMonitor lcltimer( *timerInit_ );

    // This routine only allocates space; it doesn't not perform any computation
    // if size is decreased, take the first newBS vectors of all and leave state as is
    // otherwise, grow/allocate space and set solver to unitialized

    Teuchos::RCP<const MV> tmp;
    // grab some Multivector to Clone
    // in practice, getInitVec() should always provide this, but it is possible to use a 
    // Eigenproblem with nothing in getInitVec() by manually initializing with initialize(); 
    // in case of that strange scenario, we will try to Clone from R_ because it is smaller 
    // than V_, and we don't want to keep V_ around longer than necessary
    if (blockSize_ > 0) {
      tmp = R_;
    }
    else {
      tmp = problem_->getInitVec();
      TEST_FOR_EXCEPTION(tmp == Teuchos::null,std::logic_error,
                         "Anasazi::LOBPCG::setBlockSize(): eigenproblem did not specify initial vectors to clone from.");
    }
    
    TEST_FOR_EXCEPTION(newBS <= 0 || newBS > MVT::GetVecLength(*tmp), std::invalid_argument, "Anasazi::LOBPCG::setBlockSize(): block size must be strictly positive.");
    if (newBS == blockSize_) {
      // do nothing
      return;
    }
    else if (newBS < blockSize_ && initialized_) {
      //
      // shrink vectors

      // release views so we can modify the bases
      X_ = Teuchos::null;
      KX_ = Teuchos::null;
      MX_ = Teuchos::null;
      H_ = Teuchos::null;
      KH_ = Teuchos::null;
      MH_ = Teuchos::null;
      P_ = Teuchos::null;
      KP_ = Teuchos::null;
      MP_ = Teuchos::null;

      // make new indices vectors
      std::vector<int> newind(newBS), oldind(newBS);
      for (int i=0; i<newBS; i++) {
        newind[i] = i;
        oldind[i] = i;
      }

      Teuchos::RCP<MV> newV, newMV, newKV, newR, src;
      // allocate R and newV
      newR = MVT::Clone(*tmp,newBS);
      newV = MVT::Clone(*tmp,newBS*3);
      newKV = MVT::Clone(*tmp,newBS*3);
      if (hasM_) {
        newMV = MVT::Clone(*tmp,newBS*3);
      }

      //
      // if we are initialized, we want to pull the data from V_ into newV:
      //           bs  |  bs  | bs 
      // newV  = [newX | **** |newP ]
      // newKV = [newKX| **** |newKP]
      // newMV = [newMX| **** |newMP]
      // where 
      //          oldbs   |  oldbs  |   oldbs   
      //  V_ = [newX  *** | ******* | newP  ***]
      // KV_ = [newKX *** | ******* | newKP ***]
      // MV_ = [newMX *** | ******* | newMP ***]
      //
      // we don't care to copy the data corresponding to H
      // we will not copy the M data if !hasM_, because it doesn't exist
      //

      // these are shrink operations which preserve their data
      theta_.resize(3*newBS);
      Rnorms_.resize(newBS);
      R2norms_.resize(newBS);

      // copy residual vectors: oldind,newind currently contains [0,...,newBS-1]
      src = MVT::CloneView(*R_,newind);
      MVT::SetBlock(*src,newind,*newR);
      // free old memory and point to new memory
      R_ = newR;

      // copy in order: newX newKX newMX, then newP newKP newMP
      // for  X: [0,bs-1] <-- [0,bs-1] 
      src = MVT::CloneView(*V_,oldind);
      MVT::SetBlock(*src,newind,*newV);
      src = MVT::CloneView(*KV_,oldind);
      MVT::SetBlock(*src,newind,*newKV);
      if (hasM_) {
        src = MVT::CloneView(*MV_,oldind);
        MVT::SetBlock(*src,newind,*newMV);
      }
      // for  P: [2*bs, 3*bs-1] <-- [2*oldbs, 2*oldbs+bs-1] 
      for (int i=0; i<newBS; i++) {
        newind[i] += 2*newBS;
        oldind[i] += 2*blockSize_;
      }
      src = MVT::CloneView(*V_,oldind);
      MVT::SetBlock(*src,newind,*newV);
      src = MVT::CloneView(*KV_,oldind);
      MVT::SetBlock(*src,newind,*newKV);
      if (hasM_) {
        src = MVT::CloneView(*MV_,oldind);
        MVT::SetBlock(*src,newind,*newMV);
      }

      // release temp view
      src = Teuchos::null;

      // release old allocations and point at new memory
      V_ = newV;
      KV_ = newKV;
      if (hasM_) {
        MV_ = newMV;
      }
      else {
        MV_ = V_;
      }
    }
    else {  
      // newBS > blockSize_  or  not initialized
      // this is also the scenario for our initial call to setBlockSize(), in the constructor
      initialized_ = false;
      hasP_ = false;

      // release views
      X_ = Teuchos::null;
      KX_ = Teuchos::null;
      MX_ = Teuchos::null;
      H_ = Teuchos::null;
      KH_ = Teuchos::null;
      MH_ = Teuchos::null;
      P_ = Teuchos::null;
      KP_ = Teuchos::null;
      MP_ = Teuchos::null;

      // free allocated storage
      R_ = Teuchos::null;
      V_ = Teuchos::null;

      // allocate scalar vectors
      theta_.resize(3*newBS,NANVAL);
      Rnorms_.resize(newBS,NANVAL);
      R2norms_.resize(newBS,NANVAL);
      
      // clone multivectors off of tmp
      R_ = MVT::Clone(*tmp,newBS);
      V_ = MVT::Clone(*tmp,newBS*3);
      KV_ = MVT::Clone(*tmp,newBS*3);
      if (hasM_) {
        MV_ = MVT::Clone(*tmp,newBS*3);
      }
      else {
        MV_ = V_;
      }
    }

    // allocate tmp space
    tmpmvec_ = Teuchos::null;
    if (fullOrtho_) {
      tmpmvec_ = MVT::Clone(*tmp,newBS);
    }

    // set new block size
    blockSize_ = newBS;

    // setup new views
    setupViews();
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Setup views into V,KV,MV
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::setupViews() 
  {
    std::vector<int> ind(blockSize_);

    for (int i=0; i<blockSize_; i++) {
      ind[i] = i;
    }
    X_  = MVT::CloneView(*V_,ind);
    KX_ = MVT::CloneView(*KV_,ind);
    if (hasM_) {
      MX_ = MVT::CloneView(*MV_,ind);
    }
    else {
      MX_ = X_;
    }

    for (int i=0; i<blockSize_; i++) {
      ind[i] += blockSize_;
    }
    H_  = MVT::CloneView(*V_,ind);
    KH_ = MVT::CloneView(*KV_,ind);
    if (hasM_) {
      MH_ = MVT::CloneView(*MV_,ind);
    }
    else {
      MH_ = H_;
    }

    for (int i=0; i<blockSize_; i++) {
      ind[i] += blockSize_;
    }
    P_  = MVT::CloneView(*V_,ind);
    KP_ = MVT::CloneView(*KV_,ind);
    if (hasM_) {
      MP_ = MVT::CloneView(*MV_,ind);
    }
    else {
      MP_ = P_;
    }
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Set the auxiliary vectors
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::setAuxVecs(const Teuchos::Array<Teuchos::RCP<const MV> > &auxvecs) {
    typedef typename Teuchos::Array<Teuchos::RCP<const MV> >::iterator tarcpmv;

    // set new auxiliary vectors
    auxVecs_ = auxvecs;
    
    numAuxVecs_ = 0;
    for (tarcpmv i=auxVecs_.begin(); i != auxVecs_.end(); i++) {
      numAuxVecs_ += MVT::GetNumberVecs(**i);
    }
    
    // If the solver has been initialized, X and P are not necessarily orthogonal to new auxiliary vectors
    if (numAuxVecs_ > 0 && initialized_) {
      initialized_ = false;
      hasP_ = false;
    }

    if (om_->isVerbosity( Debug ) ) {
      // Check almost everything here
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
   * initialized_ == true
   * X is orthonormal, orthogonal to auxVecs_
   * KX = Op*X
   * MX = M*X if hasM_
   * theta_ contains Ritz values of X
   * R = KX - MX*diag(theta_)
   * if hasP() == true,
   *   P orthogonal to auxVecs_
   *   if fullOrtho_ == true,
   *     P orthonormal and orthogonal to X
   *   KP = Op*P
   *   MP = M*P
   */
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::initialize(LOBPCGState<ScalarType,MV> newstate)
  {
    // NOTE: memory has been allocated by setBlockSize(). Use SetBlock below; do not Clone
    // NOTE: Overall time spent in this routine is counted to timerInit_; portions will also be counted towards other primitives

    Teuchos::TimeMonitor lcltimer( *timerInit_ );

    std::vector<int> bsind(blockSize_);
    for (int i=0; i<blockSize_; i++) bsind[i] = i;

    // in LOBPCG, X (the subspace iterate) is primary
    // the order of dependence follows like so.
    // --init->                 X
    //    --op apply->          MX,KX
    //       --ritz analysis->  theta
    //          --optional->    P,MP,KP
    // 
    // if the user specifies all data for a level, we will accept it.
    // otherwise, we will generate the whole level, and all subsequent levels.
    //
    // the data members are ordered based on dependence, and the levels are
    // partitioned according to the amount of work required to produce the
    // items in a level.
    //
    // inconsitent multivectors widths and lengths will not be tolerated, and
    // will be treated with exceptions.

    // set up X, KX, MX: get them from "state" if user specified them

    //----------------------------------------
    // set up X, MX, KX
    //----------------------------------------
    if (newstate.X != Teuchos::null) {
      TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.X) != MVT::GetVecLength(*X_),
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.X not correct." );
      // newstate.X must have blockSize_ vectors; any more will be ignored
      TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.X) < blockSize_,
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.X must have at least block size vectors.");

      // put X data in X_
      MVT::SetBlock(*newstate.X,bsind,*X_);

      // put MX data in MX_
      if (hasM_) {
        if (newstate.MX != Teuchos::null) {
          TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.MX) != MVT::GetVecLength(*MX_),
                              std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.MX not correct." );
          // newstate.MX must have blockSize_ vectors; any more will be ignored
          TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.MX) < blockSize_,
                              std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.MX must have at least block size vectors.");
          MVT::SetBlock(*newstate.MX,bsind,*MX_);
        }
        else {
          // user didn't specify MX, compute it
          {
            Teuchos::TimeMonitor lcltimer( *timerMOp_ );
            OPT::Apply(*MOp_,*X_,*MX_);
            count_ApplyM_ += blockSize_;
          }
          // we generated MX; we will generate R as well
          newstate.R = Teuchos::null;
        }
      }
  
      // put data in KX
      if (newstate.KX != Teuchos::null) {
        TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.KX) != MVT::GetVecLength(*KX_),
                            std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.KX not correct." );
        // newstate.KX must have blockSize_ vectors; any more will be ignored
        TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.KX) < blockSize_,
                            std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.KX must have at least block size vectors.");
        MVT::SetBlock(*newstate.KX,bsind,*KX_);
      }
      else {
        // user didn't specify KX, compute it
        {
          Teuchos::TimeMonitor lcltimer( *timerOp_ );
          OPT::Apply(*Op_,*X_,*KX_);
          count_ApplyOp_ += blockSize_;
        }
        // we generated KX; we will generate R as well
        newstate.R = Teuchos::null;
      }
    }
    else {
      // user did not specify X
      // we will initialize X, compute KX and MX, and compute R
      //
      // clear state so we won't use any data from it below
      newstate.P  = Teuchos::null;
      newstate.KP = Teuchos::null;
      newstate.MP = Teuchos::null;
      newstate.R  = Teuchos::null;
      newstate.T  = Teuchos::null;

      // generate a basis and projectAndNormalize
      Teuchos::RCP<const MV> ivec = problem_->getInitVec();
      TEST_FOR_EXCEPTION(ivec == Teuchos::null,std::logic_error,
                         "Anasazi::LOBPCG::initialize(): Eigenproblem did not specify initial vectors to clone from.");

      int initSize = MVT::GetNumberVecs(*ivec);
      if (initSize > blockSize_) {
        // we need only the first blockSize_ vectors from ivec; get a view of them
        initSize = blockSize_;
        std::vector<int> ind(blockSize_);
        for (int i=0; i<blockSize_; i++) ind[i] = i;
        ivec = MVT::CloneView(*ivec,ind);
      }

      // assign ivec to first part of X_ 
      if (initSize > 0) {
        std::vector<int> ind(initSize);
        for (int i=0; i<initSize; i++) ind[i] = i;
        MVT::SetBlock(*ivec,ind,*X_);
      }
      // fill the rest of X_ with random
      if (blockSize_ > initSize) {
        std::vector<int> ind(blockSize_ - initSize);
        for (int i=0; i<blockSize_ - initSize; i++) ind[i] = initSize + i;
        Teuchos::RCP<MV> rX = MVT::CloneView(*X_,ind);
        MVT::MvRandom(*rX);
        rX = Teuchos::null;
      }

      // put data in MX
      if (hasM_) {
        Teuchos::TimeMonitor lcltimer( *timerMOp_ );
        OPT::Apply(*MOp_,*X_,*MX_);
        count_ApplyM_ += blockSize_;
      }
  
      // remove auxVecs from X_ and normalize it
      if (numAuxVecs_ > 0) {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );
        Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > dummy;
        int rank = orthman_->projectAndNormalizeMat(*X_,MX_,dummy,Teuchos::null,auxVecs_);
        TEST_FOR_EXCEPTION(rank != blockSize_, LOBPCGInitFailure,
                           "Anasazi::LOBPCG::initialize(): Couldn't generate initial basis of full rank.");
      }
      else {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );
        int rank = orthman_->normalizeMat(*X_,MX_,Teuchos::null);
        TEST_FOR_EXCEPTION(rank != blockSize_, LOBPCGInitFailure,
                           "Anasazi::LOBPCG::initialize(): Couldn't generate initial basis of full rank.");
      }

      // put data in KX
      {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply(*Op_,*X_,*KX_);
        count_ApplyOp_ += blockSize_;
      }
    } // end if (newstate.X != Teuchos::null)


    //----------------------------------------
    // set up Ritz values
    //----------------------------------------
    theta_.resize(3*blockSize_,NANVAL);
    if (newstate.T != Teuchos::null) {
      TEST_FOR_EXCEPTION( (signed int)(newstate.T->size()) < blockSize_,
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.T must contain at least block size Ritz values.");
      for (int i=0; i<blockSize_; i++) {
        theta_[i] = (*newstate.T)[i];
      }
    }
    else {
      // get ritz vecs/vals
      Teuchos::SerialDenseMatrix<int,ScalarType> KK(blockSize_,blockSize_),
                                                 MM(blockSize_,blockSize_),
                                                  S(blockSize_,blockSize_);
      {
        Teuchos::TimeMonitor lcltimer( *timerLocalProj_ );
        // project K
        MVT::MvTransMv(ONE,*X_,*KX_,KK);
        // project M
        MVT::MvTransMv(ONE,*X_,*MX_,MM);
        nevLocal_ = blockSize_;
      }

      // solve the projected problem
      {
        Teuchos::TimeMonitor lcltimer( *timerDS_ );
        Utils::directSolver(blockSize_, KK, Teuchos::rcp(&MM,false), S, theta_, nevLocal_, 1);
        TEST_FOR_EXCEPTION(nevLocal_ != blockSize_,LOBPCGInitFailure,
                           "Anasazi::LOBPCG::initialize(): Initial Ritz analysis did not produce enough Ritz pairs to initialize algorithm.");
      }

      // We only have blockSize_ ritz pairs, ergo we do not need to select.
      // However, we still require them to be ordered correctly
      {
        Teuchos::TimeMonitor lcltimer( *timerSort_ );

        std::vector<int> order(blockSize_);
        // 
        // sort the first blockSize_ values in theta_
        sm_->sort( this, blockSize_, theta_, &order );   // don't catch exception
        //
        // apply the same ordering to the primitive ritz vectors
        Utils::permuteVectors(order,S);
      }

      // update the solution, use R for storage
      {
        Teuchos::TimeMonitor lcltimer( *timerLocalUpdate_ );
        // X <- X*S
        MVT::MvAddMv( ONE, *X_, ZERO, *X_, *R_ );        
        MVT::MvTimesMatAddMv( ONE, *R_, S, ZERO, *X_ );
        // KX <- KX*S
        MVT::MvAddMv( ONE, *KX_, ZERO, *KX_, *R_ );        
        MVT::MvTimesMatAddMv( ONE, *R_, S, ZERO, *KX_ );
        if (hasM_) {
          // MX <- MX*S
          MVT::MvAddMv( ONE, *MX_, ZERO, *MX_, *R_ );        
          MVT::MvTimesMatAddMv( ONE, *R_, S, ZERO, *MX_ );
        }
      }
    }

    //----------------------------------------
    // compute R
    //----------------------------------------
    if (newstate.R != Teuchos::null) {
      TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.R) != MVT::GetVecLength(*R_),
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.R not correct." );
      TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.R) < blockSize_,
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.R must have blockSize number of vectors." );
      MVT::SetBlock(*newstate.R,bsind,*R_);
    }
    else {
      Teuchos::TimeMonitor lcltimer( *timerCompRes_ );
      // form R <- KX - MX*T
      MVT::MvAddMv(ZERO,*KX_,ONE,*KX_,*R_);
      Teuchos::SerialDenseMatrix<int,ScalarType> T(blockSize_,blockSize_);
      for (int i=0; i<blockSize_; i++) T(i,i) = theta_[i];
      MVT::MvTimesMatAddMv(-ONE,*MX_,T,ONE,*R_);
    }

    // R has been updated; mark the norms as out-of-date
    Rnorms_current_ = false;
    R2norms_current_ = false;
  
    // put data in P,KP,MP: P is not used to set theta
    if (newstate.P != Teuchos::null) {
      TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.P) < blockSize_ ,
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.P must have blockSize number of vectors." );
      TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.P) != MVT::GetVecLength(*P_),
                          std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.P not correct." );
      hasP_ = true;

      // set P_
      MVT::SetBlock(*newstate.P,bsind,*P_);

      // set/compute KP_
      if (newstate.KP != Teuchos::null) {
        TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.KP) < blockSize_,
                            std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.KP must have blockSize number of vectors." );
        TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.KP) != MVT::GetVecLength(*KP_),
                            std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.KP not correct." );
        MVT::SetBlock(*newstate.KP,bsind,*KP_);
      }
      else {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply(*Op_,*P_,*KP_);
        count_ApplyOp_ += blockSize_;
      }

      // set/compute MP_
      if (hasM_) {
        if (newstate.MP != Teuchos::null) {
          TEST_FOR_EXCEPTION( MVT::GetNumberVecs(*newstate.MP) < blockSize_,
                              std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): newstate.MP must have blockSize number of vectors." );
          TEST_FOR_EXCEPTION( MVT::GetVecLength(*newstate.MP) != MVT::GetVecLength(*MP_),
                              std::invalid_argument, "Anasazi::LOBPCG::initialize(newstate): vector length of newstate.MP not correct." );
          MVT::SetBlock(*newstate.MP,bsind,*MP_);
        }
        else {
          Teuchos::TimeMonitor lcltimer( *timerMOp_ );
          OPT::Apply(*MOp_,*P_,*MP_);
          count_ApplyM_ += blockSize_;
        }
      }
    }
    else {
      hasP_ = false;
    }

    // finally, we are initialized
    initialized_ = true;

    if (om_->isVerbosity( Debug ) ) {
      // Check almost everything here
      CheckList chk;
      chk.checkX = true;
      chk.checkKX = true;
      chk.checkMX = true;
      chk.checkP = true;
      chk.checkKP = true;
      chk.checkMP = true;
      chk.checkR = true;
      chk.checkQ = true;
      om_->print( Debug, accuracyCheck(chk, ": after initialize()") );
    }

  }

  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::initialize()
  {
    LOBPCGState<ScalarType,MV> empty;
    initialize(empty);
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Instruct the solver to use full orthogonalization
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::setFullOrtho (bool fullOrtho)
  {
    if ( fullOrtho_ == true || initialized_ == false || fullOrtho == fullOrtho_ ) {
      // state is already orthogonalized or solver is not initialized
      fullOrtho_ = fullOrtho;
    }
    else {
      // solver is initialized, state is not fully orthogonalized, and user has requested full orthogonalization
      // ergo, we must throw away data in P
      fullOrtho_ = true;
      hasP_ = false;
    }

    // the user has called setFullOrtho, so the class has been instantiated
    // ergo, the data has already been allocated, i.e., setBlockSize() has been called
    // if it is already allocated, it should be the proper size
    if (fullOrtho_ && tmpmvec_ == Teuchos::null) {
      // allocated the workspace
      tmpmvec_ = MVT::Clone(*X_,blockSize_);
    }
    else if (fullOrtho_==false) {
      // free the workspace
      tmpmvec_ = Teuchos::null;
    }
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform LOBPCG iterations until the StatusTest tells us to stop.
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::iterate () 
  {
    //
    // Allocate/initialize data structures
    //
    if (initialized_ == false) {
      initialize();
    }

    //
    // Miscellaneous definitions
    const int oneBlock    =   blockSize_;
    const int twoBlocks   = 2*blockSize_;
    const int threeBlocks = 3*blockSize_;

    std::vector<int> indblock1(blockSize_), indblock2(blockSize_), indblock3(blockSize_);
    for (int i=0; i<blockSize_; i++) {
      indblock1[i] = i;
      indblock2[i] = i +  blockSize_;
      indblock3[i] = i + 2*blockSize_;
    }

    //
    // Define dense projected/local matrices
    //   KK = Local stiffness matrix               (size: 3*blockSize_ x 3*blockSize_)
    //   MM = Local mass matrix                    (size: 3*blockSize_ x 3*blockSize_)
    //    S = Local eigenvectors                   (size: 3*blockSize_ x 3*blockSize_)
    Teuchos::SerialDenseMatrix<int,ScalarType> KK( threeBlocks, threeBlocks ), 
                                               MM( threeBlocks, threeBlocks ),
                                                S( threeBlocks, threeBlocks );

    while (tester_->checkStatus(this) != Passed) {

      // Print information on current status
      if (om_->isVerbosity(Debug)) {
        currentStatus( om_->stream(Debug) );
      }
      else if (om_->isVerbosity(IterationDetails)) {
        currentStatus( om_->stream(IterationDetails) );
      }

      // increment iteration counter
      iter_++;

      // Apply the preconditioner on the residuals: H <- Prec*R
      if (Prec_ != Teuchos::null) {
        Teuchos::TimeMonitor lcltimer( *timerPrec_ );
        OPT::Apply( *Prec_, *R_, *H_ );   // don't catch the exception
        count_ApplyPrec_ += blockSize_;
      }
      else {
        MVT::MvAddMv(ONE,*R_,ZERO,*R_,*H_);
      }

      // Apply the mass matrix on H
      if (hasM_) {
        Teuchos::TimeMonitor lcltimer( *timerMOp_ );
        OPT::Apply( *MOp_, *H_, *MH_);    // don't catch the exception
        count_ApplyM_ += blockSize_;
      }

      // orthogonalize H against the auxiliary vectors
      // optionally: orthogonalize H against X and P ([X P] is already orthonormal)
      Teuchos::Array<Teuchos::RCP<const MV> > Q;
      Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > C = 
        Teuchos::tuple<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > >(Teuchos::null);
      Q = auxVecs_;
      if (fullOrtho_) {
        // X and P are not contiguous, so there is not much point in putting them under 
        // a single multivector view
        Q.push_back(X_);
        if (hasP_) {
          Q.push_back(P_);
        }
      }
      {
        Teuchos::TimeMonitor lcltimer( *timerOrtho_ );
        int rank = orthman_->projectAndNormalizeMat(*H_,MH_,C,Teuchos::null,Q);
        // our views are currently in place; it is safe to throw an exception
        TEST_FOR_EXCEPTION(rank != blockSize_,LOBPCGOrthoFailure,
                           "Anasazi::LOBPCG::iterate(): unable to compute orthonormal basis for H.");
      }

      if (om_->isVerbosity( Debug ) ) {
        CheckList chk;
        chk.checkH = true;
        chk.checkMH = true;
        om_->print( Debug, accuracyCheck(chk, ": after ortho H") );
      }
      else if (om_->isVerbosity( OrthoDetails ) ) {
        CheckList chk;
        chk.checkH = true;
        chk.checkMH = true;
        om_->print( OrthoDetails, accuracyCheck(chk,": after ortho H") );
      }

      // Apply the stiffness matrix to H
      {
        Teuchos::TimeMonitor lcltimer( *timerOp_ );
        OPT::Apply( *Op_, *H_, *KH_);   // don't catch the exception
        count_ApplyOp_ += blockSize_;
      }

      if (hasP_) {
        nevLocal_ = threeBlocks;
      }
      else {
        nevLocal_ = twoBlocks;
      }

      //
      // we need bases: [X H P] and [H P] (only need the latter if fullOrtho == false)
      // we need to perform the following operations:
      //    X' [KX KH KP]
      //    X' [MX MH MP]
      //    H' [KH KP]
      //    H' [MH MP]
      //    P' [KP]
      //    P' [MP]
      //    [X H P] CX
      //    { [X H P] CP    if  fullOrtho
      //    {   [H P] CP    if !fullOrtho
      //
      // since M[X H P] is potentially the same memory as [X H P], and 
      // because we are not allowed to have overlapping non-const views of 
      // a multivector, we will now abandon our non-const views in favor of 
      // const views
      //
      X_ = Teuchos::null;
      KX_ = Teuchos::null;
      MX_ = Teuchos::null;
      H_ = Teuchos::null;
      KH_ = Teuchos::null;
      MH_ = Teuchos::null;
      P_ = Teuchos::null;
      KP_ = Teuchos::null;
      MP_ = Teuchos::null;
      Teuchos::RCP<const MV> cX, cH, cXHP, cHP, cK_XHP, cK_HP, cM_XHP, cM_HP, cP, cK_P, cM_P;
      {
        cX = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(V_),indblock1);
        cH = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(V_),indblock2);

        std::vector<int> indXHP(nevLocal_);
        for (int i=0; i<nevLocal_; i++) {
          indXHP[i] = i;
        }
        cXHP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(V_),indXHP);
        cK_XHP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(KV_),indXHP);
        if (hasM_) {
          cM_XHP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(MV_),indXHP);
        }
        else {
          cM_XHP = cXHP;
        }

        std::vector<int> indHP(nevLocal_-blockSize_);
        for (int i=blockSize_; i<nevLocal_; i++) {
          indHP[i-blockSize_] = i;
        }
        cHP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(V_),indHP);
        cK_HP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(KV_),indHP);
        if (hasM_) {
          cM_HP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(MV_),indHP);
        }
        else {
          cM_HP = cHP;
        }

        if (nevLocal_ == threeBlocks) {
          cP = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(V_),indblock3);
          cK_P = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(KV_),indblock3);
          if (hasM_) {
            cM_P = MVT::CloneView(*Teuchos::rcp_implicit_cast<const MV>(MV_),indblock3);
          }
          else {
            cM_P = cP;
          }
        }
      }

      //
      //----------------------------------------
      // Form "local" mass and stiffness matrices
      //----------------------------------------
      {
        // We will form only the block upper triangular part of 
        // [X H P]' K [X H P]  and  [X H P]' M [X H P]
        // Get the necessary views into KK and MM:
        //      [--K1--]        [--M1--]
        // KK = [  -K2-]   MM = [  -M2-]
        //      [    K3]        [    M3]
        // 
        // It is okay to declare a zero-area view of a Teuchos::SerialDenseMatrix
        //
        Teuchos::SerialDenseMatrix<int,ScalarType> 
          K1(Teuchos::View,KK,blockSize_,nevLocal_             ,0*blockSize_,0*blockSize_),
          K2(Teuchos::View,KK,blockSize_,nevLocal_-1*blockSize_,1*blockSize_,1*blockSize_),
          K3(Teuchos::View,KK,blockSize_,nevLocal_-2*blockSize_,2*blockSize_,2*blockSize_),
          M1(Teuchos::View,MM,blockSize_,nevLocal_             ,0*blockSize_,0*blockSize_),
          M2(Teuchos::View,MM,blockSize_,nevLocal_-1*blockSize_,1*blockSize_,1*blockSize_),
          M3(Teuchos::View,MM,blockSize_,nevLocal_-2*blockSize_,2*blockSize_,2*blockSize_);
        {
          Teuchos::TimeMonitor lcltimer( *timerLocalProj_ );
          MVT::MvTransMv( ONE, *cX, *cK_XHP, K1 );
          MVT::MvTransMv( ONE, *cX, *cM_XHP, M1 );
          MVT::MvTransMv( ONE, *cH, *cK_HP , K2 );
          MVT::MvTransMv( ONE, *cH, *cM_HP , M2 );
          if (nevLocal_ == threeBlocks) {
            MVT::MvTransMv( ONE, *cP, *cK_P, K3 );
            MVT::MvTransMv( ONE, *cP, *cM_P, M3 );
          }
        }
      }
      // below, we only need bases [X H P] and [H P] and friends
      // furthermore, we only need [H P] and friends if fullOrtho == false
      // clear the others now
      cX   = Teuchos::null;
      cH   = Teuchos::null;
      cP   = Teuchos::null;
      cK_P = Teuchos::null;
      cM_P = Teuchos::null;
      if (fullOrtho_ == true) {
        cHP = Teuchos::null;
        cK_HP = Teuchos::null;
        cM_HP = Teuchos::null;
      }

      //
      //---------------------------------------------------
      // Perform a spectral decomposition of (KK,MM)
      //---------------------------------------------------
      //
      // Get pointers to relevant part of KK, MM and S for Rayleigh-Ritz analysis
      Teuchos::SerialDenseMatrix<int,ScalarType> lclKK(Teuchos::View,KK,nevLocal_,nevLocal_), 
                                                 lclMM(Teuchos::View,MM,nevLocal_,nevLocal_),
                                                  lclS(Teuchos::View, S,nevLocal_,nevLocal_);
      {
        Teuchos::TimeMonitor lcltimer( *timerDS_ );
        int localSize = nevLocal_;
        Utils::directSolver(localSize, lclKK, Teuchos::rcp(&lclMM,false), lclS, theta_, nevLocal_, 0);
        // localSize tells directSolver() how big KK,MM are
        // however, directSolver() may choose to use only the principle submatrices of KK,MM 
        // because of loss of MM-orthogonality in the projected eigenvectors
        // nevLocal_ tells us how much it used, telling us the effective localSize 
        // (i.e., how much of KK,MM used by directSolver)
        // we will not tolerate any indefiniteness, and will throw an exception if it was 
        // detected by directSolver
        //
        if (nevLocal_ != localSize) {
          // before throwing the exception, and thereby leaving iterate(), setup the views again
          // first, clear the const views
          cXHP   = Teuchos::null;
          cK_XHP = Teuchos::null;
          cM_XHP = Teuchos::null;
          cHP    = Teuchos::null;
          cK_HP  = Teuchos::null;
          cM_HP  = Teuchos::null;
          setupViews();
        }
        TEST_FOR_EXCEPTION(nevLocal_ != localSize, LOBPCGRitzFailure, 
            "Anasazi::LOBPCG::iterate(): indefiniteness detected in projected mass matrix." );
      }

      //
      //---------------------------------------------------
      // Sort the ritz values using the sort manager
      //---------------------------------------------------
      Teuchos::LAPACK<int,ScalarType> lapack;
      Teuchos::BLAS<int,ScalarType> blas;
      {
        Teuchos::TimeMonitor lcltimer( *timerSort_ );

        std::vector<int> order(nevLocal_);
        // 
        // Sort the first nevLocal_ values in theta_
        sm_->sort( this, nevLocal_, theta_, &order );   // don't catch exception
        //
        // Sort the primitive ritz vectors
        Utils::permuteVectors(order,lclS);
      }

      //
      //----------------------------------------
      // Compute coefficients for X and P under [X H P]
      //----------------------------------------
      // Before computing X,P, optionally perform orthogonalization per Hetmaniuk,Lehoucq paper
      // CX will be the coefficients of [X,H,P] for new X, CP for new P
      // The paper suggests orthogonalizing CP against CX and orthonormalizing CP, w.r.t. MM
      // Here, we will also orthonormalize CX.
      // This is accomplished using the Cholesky factorization of [CX CP]^H lclMM [CX CP]
      Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > CX, CP;
      if (fullOrtho_) {
        // build orthonormal basis for (  0  ) that is MM orthogonal to ( S11 )
        //                             ( S21 )                          ( S21 )
        //                             ( S31 )                          ( S31 )
        // Do this using Cholesky factorization of ( S11  0  )
        //                                         ( S21 S21 )
        //                                         ( S31 S31 )
        //           ( S11  0  )
        // Build C = ( S21 S21 )
        //           ( S31 S31 )
        Teuchos::SerialDenseMatrix<int,ScalarType> C(nevLocal_,twoBlocks),
                                                tmp1(nevLocal_,twoBlocks),
                                                tmp2(twoBlocks  ,twoBlocks);

        // first block of rows: ( S11 0 )
        for (int j=0; j<oneBlock; j++) {
          for (int i=0; i<oneBlock; i++) {
            // CX
            C(i,j) = lclS(i,j);
            // CP
            C(i,j+oneBlock) = ZERO;
          }
        }
        // second block of rows: (S21 S21)
        for (int j=0; j<oneBlock; j++) {
          for (int i=oneBlock; i<twoBlocks; i++) {
            // CX
            C(i,j)          = lclS(i,j);
            // CP
            C(i,j+oneBlock) = lclS(i,j);
          }
        }
        // third block of rows
        if (nevLocal_ == threeBlocks) {
          for (int j=0; j<oneBlock; j++) {
            for (int i=twoBlocks; i<threeBlocks; i++) {
              // CX
              C(i,j)          = lclS(i,j);
              // CP
              C(i,j+oneBlock) = lclS(i,j);
            }
          }
        }

        // compute tmp1 = lclMM*C
        int teuchosret;
        teuchosret = tmp1.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,lclMM,C,ZERO);
        TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,
                           "Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");

        // compute tmp2 = C^H*tmp1 == C^H*lclMM*C
        teuchosret = tmp2.multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,C,tmp1,ZERO);
        TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,
                           "Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");

        // compute R (cholesky) of tmp2
        int info;
        lapack.POTRF('U',twoBlocks,tmp2.values(),tmp2.stride(),&info);
        // our views ARE NOT currently in place; we must reestablish them before throwing an exception
        if (info != 0) {
          cXHP = Teuchos::null;
          cHP = Teuchos::null;
          cK_XHP = Teuchos::null;
          cK_HP = Teuchos::null;
          cM_XHP = Teuchos::null;
          cM_HP = Teuchos::null;
          setupViews();
        }
        TEST_FOR_EXCEPTION(info != 0, LOBPCGOrthoFailure, 
                           "Anasazi::LOBPCG::iterate(): Cholesky factorization failed during full orthogonalization.");
        // compute C = C inv(R)
        blas.TRSM(Teuchos::RIGHT_SIDE,Teuchos::UPPER_TRI,Teuchos::NO_TRANS,Teuchos::NON_UNIT_DIAG,
                  nevLocal_,twoBlocks,ONE,tmp2.values(),tmp2.stride(),C.values(),C.stride());
        // put C(:,0:oneBlock-1) into CX
        CX = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::Copy,C,nevLocal_,oneBlock,0,0) );
        // put C(:,oneBlock:twoBlocks-1) into CP
        CP = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::Copy,C,nevLocal_,oneBlock,0,oneBlock) );

        // check the results
        if (om_->isVerbosity( Debug ) ) {
          Teuchos::SerialDenseMatrix<int,ScalarType> tmp1(nevLocal_,oneBlock),
                                                     tmp2(oneBlock,oneBlock);
          MagnitudeType tmp;
          int teuchosret;
          std::stringstream os;
          os.precision(2);
          os.setf(std::ios::scientific, std::ios::floatfield);

          os << " Checking Full Ortho: iteration " << iter_ << std::endl;

          // check CX^T MM CX == I
          // compute tmp1 = MM*CX
          teuchosret = tmp1.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,lclMM,*CX,ZERO);
          TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,
              "Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");
          // compute tmp2 = CX^H*tmp1 == CX^H*MM*CX
          teuchosret = tmp2.multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,*CX,tmp1,ZERO);
          TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,
              "Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");
          // subtrace tmp2 - I == CX^H * MM * CX - I
          for (int i=0; i<oneBlock; i++) tmp2(i,i) -= ONE;
          tmp = tmp2.normFrobenius();          
          os << " >> Error in CX^H MM CX == I : " << tmp << std::endl;

          // check CP^T MM CP == I
          // compute tmp1 = MM*CP
          teuchosret = tmp1.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,lclMM,*CP,ZERO);
          TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,
              "Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");
          // compute tmp2 = CP^H*tmp1 == CP^H*MM*CP
          teuchosret = tmp2.multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,*CP,tmp1,ZERO);
          TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,
              "Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");
          // subtrace tmp2 - I == CP^H * MM * CP - I
          for (int i=0; i<oneBlock; i++) tmp2(i,i) -= ONE;
          tmp = tmp2.normFrobenius();          
          os << " >> Error in CP^H MM CP == I : " << tmp << std::endl;

          // check CX^T MM CP == 0
          // compute tmp1 = MM*CP
          teuchosret = tmp1.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,lclMM,*CP,ZERO);
          TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,"Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");
          // compute tmp2 = CX^H*tmp1 == CX^H*MM*CP
          teuchosret = tmp2.multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,*CX,tmp1,ZERO);
          TEST_FOR_EXCEPTION(teuchosret != 0,std::logic_error,"Anasazi::LOBPCG::iterate(): Logic error calling SerialDenseMatrix::multiply");
          // subtrace tmp2 == CX^H * MM * CP
          tmp = tmp2.normFrobenius();          
          os << " >> Error in CX^H MM CP == 0 : " << tmp << std::endl;

          os << std::endl;
          om_->print(Debug,os.str());
        }
      }
      else {
        //     [S11 ... ...]
        // S = [S21 ... ...]
        //     [S31 ... ...]
        //
        // CX = [S11]
        //      [S21]
        //      [S31]   ->  X = [X H P] CX
        //      
        // CP = [S21]   ->  P =   [H P] CP
        //      [S31]
        //
        CX = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::Copy,lclS,nevLocal_         ,oneBlock,0       ,0) );
        CP = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>(Teuchos::Copy,lclS,nevLocal_-oneBlock,oneBlock,oneBlock,0) );
      }

      //
      //----------------------------------------
      // Compute new X and new P
      //----------------------------------------
      // Note: Use R as a temporary work space and (if full ortho) tmpMV as well
      {
        Teuchos::TimeMonitor lcltimer( *timerLocalUpdate_ );

        // if full ortho, then CX and CP are dense
        // we multiply [X H P]*CX into tmpMV
        //             [X H P]*CP into R
        // then put V(:,firstblock) <- tmpMV
        //          V(:,thirdblock) <- R
        //
        // if no full ortho, then [H P]*CP doesn't reference first block (X) 
        // of V, so that we can modify it before computing P
        // so we multiply [X H P]*CX into R
        //                V(:,firstblock) <- R
        //       multiply [H P]*CP into R
        //                V(:,thirdblock) <- R
        //
        // mutatis mutandis for K[XP] and M[XP]
        //
        // use SetBlock to do the assignments into V_
        //
        // in either case, views are only allowed to be overlapping
        // if they are const, and it should be assume that SetBlock
        // creates a view of the associated part
        //
        // we have from above const-pointers to [KM]XHP, [KM]HP and (if hasP) [KM]P
        //
        if (fullOrtho_) {
          // X,P
          MVT::MvTimesMatAddMv(ONE,*cXHP,*CX,ZERO,*tmpmvec_);
          MVT::MvTimesMatAddMv(ONE,*cXHP,*CP,ZERO,*R_);
          cXHP = Teuchos::null;
          MVT::SetBlock(*tmpmvec_,indblock1,*V_);
          MVT::SetBlock(*R_      ,indblock3,*V_);
          // KX,KP
          MVT::MvTimesMatAddMv(ONE,*cK_XHP,*CX,ZERO,*tmpmvec_);
          MVT::MvTimesMatAddMv(ONE,*cK_XHP,*CP,ZERO,*R_);
          cK_XHP = Teuchos::null;
          MVT::SetBlock(*tmpmvec_,indblock1,*KV_);
          MVT::SetBlock(*R_      ,indblock3,*KV_);
          // MX,MP
          if (hasM_) {
            MVT::MvTimesMatAddMv(ONE,*cM_XHP,*CX,ZERO,*tmpmvec_);
            MVT::MvTimesMatAddMv(ONE,*cM_XHP,*CP,ZERO,*R_);
            cM_XHP = Teuchos::null;
            MVT::SetBlock(*tmpmvec_,indblock1,*MV_);
            MVT::SetBlock(*R_      ,indblock3,*MV_);
          }
          else {
            cM_XHP = Teuchos::null;
          }
        }
        else {
          // X,P
          MVT::MvTimesMatAddMv(ONE,*cXHP,*CX,ZERO,*R_);
          cXHP = Teuchos::null;
          MVT::SetBlock(*R_,indblock1,*V_);
          MVT::MvTimesMatAddMv(ONE,*cHP,*CP,ZERO,*R_);
          cHP = Teuchos::null;
          MVT::SetBlock(*R_,indblock3,*V_);
          // KX,KP
          MVT::MvTimesMatAddMv(ONE,*cK_XHP,*CX,ZERO,*R_);
          cK_XHP = Teuchos::null;
          MVT::SetBlock(*R_,indblock1,*KV_);
          MVT::MvTimesMatAddMv(ONE,*cK_HP,*CP,ZERO,*R_);
          cK_HP = Teuchos::null;
          MVT::SetBlock(*R_,indblock3,*KV_);
          // MX,MP
          if (hasM_) {
            MVT::MvTimesMatAddMv(ONE,*cM_XHP,*CX,ZERO,*R_);
            cM_XHP = Teuchos::null;
            MVT::SetBlock(*R_,indblock1,*MV_);
            MVT::MvTimesMatAddMv(ONE,*cM_HP,*CP,ZERO,*R_);
            cM_HP = Teuchos::null;
            MVT::SetBlock(*R_,indblock3,*MV_);
          }
          else {
            cM_XHP = Teuchos::null;
            cM_HP = Teuchos::null;
          }
        }
      } // end timing block
      // done with coefficient matrices
      CX = Teuchos::null;
      CP = Teuchos::null;

      //
      // we now have a P direction
      hasP_ = true;

      // debugging check: all of our const views should have been cleared by now
      // if not, we have a logic error above
      TEST_FOR_EXCEPTION(   cXHP != Teuchos::null || cK_XHP != Teuchos::null || cM_XHP != Teuchos::null
                          || cHP != Teuchos::null ||  cK_HP != Teuchos::null || cM_HP  != Teuchos::null
                          ||  cP != Teuchos::null ||   cK_P != Teuchos::null || cM_P   != Teuchos::null,
                          std::logic_error,
                          "Anasazi::BlockKrylovSchur::iterate(): const views were not all cleared! Something went wrong!" );

      //
      // recreate our const MV views of X,H,P and friends
      setupViews();

      //
      // Compute the new residuals, explicitly
      {
        Teuchos::TimeMonitor lcltimer( *timerCompRes_ );
        MVT::MvAddMv( ONE, *KX_, ZERO, *KX_, *R_ );
        Teuchos::SerialDenseMatrix<int,ScalarType> T( blockSize_, blockSize_ );
        for (int i = 0; i < blockSize_; i++) {
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
        chk.checkX = true;
        chk.checkKX = true;
        chk.checkMX = true;
        chk.checkP = true;
        chk.checkKP = true;
        chk.checkMP = true;
        chk.checkR = true;
        om_->print( Debug, accuracyCheck(chk, ": after local update") );
      }
      else if (om_->isVerbosity( OrthoDetails )) {
        CheckList chk;
        chk.checkX = true;
        chk.checkP = true;
        chk.checkR = true;
        om_->print( OrthoDetails, accuracyCheck(chk, ": after local update") );
      }
    } // end while (statusTest == false)
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // compute/return residual M-norms
  template <class ScalarType, class MV, class OP>
  std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> 
  LOBPCG<ScalarType,MV,OP>::getResNorms() {
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
  LOBPCG<ScalarType,MV,OP>::getRes2Norms() {
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
  // checkX : X orthonormal
  //          orthogonal to auxvecs
  // checkMX: check MX == M*X
  // checkKX: check KX == K*X
  // checkP : if fullortho P orthonormal and orthogonal to X
  //          orthogonal to auxvecs
  // checkMP: check MP == M*P
  // checkKP: check KP == K*P
  // checkH : if fullortho H orthonormal and orthogonal to X and P
  //          orthogonal to auxvecs
  // checkMH: check MH == M*H
  // checkR : check R orthogonal to X
  // checkQ : check that auxiliary vectors are actually orthonormal
  //
  // TODO: 
  //  add checkTheta 
  //
  template <class ScalarType, class MV, class OP>
  std::string LOBPCG<ScalarType,MV,OP>::accuracyCheck( const CheckList &chk, const std::string &where ) const 
  {
    using std::endl;

    std::stringstream os;
    os.precision(2);
    os.setf(std::ios::scientific, std::ios::floatfield);
    MagnitudeType tmp;

    os << " Debugging checks: iteration " << iter_ << where << endl;

    // X and friends
    if (chk.checkX && initialized_) {
      tmp = orthman_->orthonormError(*X_);
      os << " >> Error in X^H M X == I : " << tmp << endl;
      for (unsigned int i=0; i<auxVecs_.size(); i++) {
        tmp = orthman_->orthogError(*X_,*auxVecs_[i]);
        os << " >> Error in X^H M Q[" << i << "] == 0 : " << tmp << endl;
      }
    }
    if (chk.checkMX && hasM_ && initialized_) {
      tmp = Utils::errorEquality(*X_, *MX_, MOp_);
      os << " >> Error in MX == M*X    : " << tmp << endl;
    }
    if (chk.checkKX && initialized_) {
      tmp = Utils::errorEquality(*X_, *KX_, Op_);
      os << " >> Error in KX == K*X    : " << tmp << endl;
    }

    // P and friends
    if (chk.checkP && hasP_ && initialized_) {
      if (fullOrtho_) {
        tmp = orthman_->orthonormError(*P_);
        os << " >> Error in P^H M P == I : " << tmp << endl;
        tmp = orthman_->orthogError(*P_,*X_);
        os << " >> Error in P^H M X == 0 : " << tmp << endl;
      }
      for (unsigned int i=0; i<auxVecs_.size(); i++) {
        tmp = orthman_->orthogError(*P_,*auxVecs_[i]);
        os << " >> Error in P^H M Q[" << i << "] == 0 : " << tmp << endl;
      }
    }
    if (chk.checkMP && hasM_ && hasP_ && initialized_) {
      tmp = Utils::errorEquality(*P_, *MP_, MOp_);
      os << " >> Error in MP == M*P    : " << tmp << endl;
    }
    if (chk.checkKP && hasP_ && initialized_) {
      tmp = Utils::errorEquality(*P_, *KP_, Op_);
      os << " >> Error in KP == K*P    : " << tmp << endl;
    }

    // H and friends
    if (chk.checkH && initialized_) {
      if (fullOrtho_) {
        tmp = orthman_->orthonormError(*H_);
        os << " >> Error in H^H M H == I : " << tmp << endl;
        tmp = orthman_->orthogError(*H_,*X_);
        os << " >> Error in H^H M X == 0 : " << tmp << endl;
        if (hasP_) {
          tmp = orthman_->orthogError(*H_,*P_);
          os << " >> Error in H^H M P == 0 : " << tmp << endl;
        }
      }
      for (unsigned int i=0; i<auxVecs_.size(); i++) {
        tmp = orthman_->orthogError(*H_,*auxVecs_[i]);
        os << " >> Error in H^H M Q[" << i << "] == 0 : " << tmp << endl;
      }
    }
    if (chk.checkMH && hasM_ && initialized_) {
      tmp = Utils::errorEquality(*H_, *MH_, MOp_);
      os << " >> Error in MH == M*H    : " << tmp << endl;
    }

    // R: this is not M-orthogonality, but standard euclidean orthogonality
    if (chk.checkR && initialized_) {
      Teuchos::SerialDenseMatrix<int,ScalarType> xTx(blockSize_,blockSize_);
      MVT::MvTransMv(ONE,*X_,*R_,xTx);
      tmp = xTx.normFrobenius();
      MVT::MvTransMv(ONE,*R_,*R_,xTx);
      double normR = xTx.normFrobenius();
      os << " >> RelError in X^H R == 0: " << tmp/normR << endl;
    }

    // Q
    if (chk.checkQ) {
      for (unsigned int i=0; i<auxVecs_.size(); i++) {
        tmp = orthman_->orthonormError(*auxVecs_[i]);
        os << " >> Error in Q[" << i << "]^H M Q[" << i << "] == I : " << tmp << endl;
        for (unsigned int j=i+1; j<auxVecs_.size(); j++) {
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
  LOBPCG<ScalarType,MV,OP>::currentStatus(std::ostream &os) 
  {
    using std::endl;

    os.setf(std::ios::scientific, std::ios::floatfield);  
    os.precision(6);
    os <<endl;
    os <<"================================================================================" << endl;
    os << endl;
    os <<"                              LOBPCG Solver Status" << endl;
    os << endl;
    os <<"The solver is "<<(initialized_ ? "initialized." : "not initialized.") << endl;
    os <<"The number of iterations performed is " << iter_       << endl;
    os <<"The current block size is             " << blockSize_  << endl;
    os <<"The number of auxiliary vectors is    " << numAuxVecs_ << endl;
    os <<"The number of operations Op*x   is " << count_ApplyOp_   << endl;
    os <<"The number of operations M*x    is " << count_ApplyM_    << endl;
    os <<"The number of operations Prec*x is " << count_ApplyPrec_ << endl;

    os.setf(std::ios_base::right, std::ios_base::adjustfield);

    if (initialized_) {
      os << endl;
      os <<"CURRENT EIGENVALUE ESTIMATES             "<<endl;
      os << std::setw(20) << "Eigenvalue" 
         << std::setw(20) << "Residual(M)"
         << std::setw(20) << "Residual(2)"
         << endl;
      os <<"--------------------------------------------------------------------------------"<<endl;
      for (int i=0; i<blockSize_; i++) {
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

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // are we initialized or not?
  template <class ScalarType, class MV, class OP>
  bool LOBPCG<ScalarType,MV,OP>::isInitialized() const { 
    return initialized_; 
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////
  // is P valid or not?
  template <class ScalarType, class MV, class OP>
  bool LOBPCG<ScalarType,MV,OP>::hasP() {
    return hasP_;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // is full orthogonalization enabled or not?
  template <class ScalarType, class MV, class OP>
  bool LOBPCG<ScalarType,MV,OP>::getFullOrtho() const { 
    return(fullOrtho_); 
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current auxilliary vectors
  template <class ScalarType, class MV, class OP>
  Teuchos::Array<Teuchos::RCP<const MV> > LOBPCG<ScalarType,MV,OP>::getAuxVecs() const {
    return auxVecs_;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current block size
  template <class ScalarType, class MV, class OP>
  int LOBPCG<ScalarType,MV,OP>::getBlockSize() const {
    return(blockSize_); 
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current eigenproblem
  template <class ScalarType, class MV, class OP>
  const Eigenproblem<ScalarType,MV,OP>& LOBPCG<ScalarType,MV,OP>::getProblem() const { 
    return(*problem_); 
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the max subspace dimension
  template <class ScalarType, class MV, class OP>
  int LOBPCG<ScalarType,MV,OP>::getMaxSubspaceDim() const {
    return 3*blockSize_;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current subspace dimension
  template <class ScalarType, class MV, class OP>
  int LOBPCG<ScalarType,MV,OP>::getCurSubspaceDim() const {
    if (!initialized_) return 0;
    return nevLocal_;
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current ritz residual norms
  template <class ScalarType, class MV, class OP>
  std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> 
  LOBPCG<ScalarType,MV,OP>::getRitzRes2Norms() 
  {
    return this->getRes2Norms();
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current compression indices
  template <class ScalarType, class MV, class OP>
  std::vector<int> LOBPCG<ScalarType,MV,OP>::getRitzIndex() {
    std::vector<int> ret(nevLocal_,0);
    return ret;
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current ritz values
  template <class ScalarType, class MV, class OP>
  std::vector<Value<ScalarType> > LOBPCG<ScalarType,MV,OP>::getRitzValues() { 
    std::vector<Value<ScalarType> > ret(nevLocal_);
    for (int i=0; i<nevLocal_; i++) {
      ret[i].realpart = theta_[i];
      ret[i].imagpart = ZERO;
    }
    return ret;
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the current ritz vectors
  template <class ScalarType, class MV, class OP>
  Teuchos::RCP<const MV> LOBPCG<ScalarType,MV,OP>::getRitzVectors() {
    return X_;
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // reset the iteration counter
  template <class ScalarType, class MV, class OP>
  void LOBPCG<ScalarType,MV,OP>::resetNumIters() {
    iter_=0; 
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the number of iterations
  template <class ScalarType, class MV, class OP>
  int LOBPCG<ScalarType,MV,OP>::getNumIters() const { 
    return(iter_); 
  }

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // return the state
  template <class ScalarType, class MV, class OP>
  LOBPCGState<ScalarType,MV> LOBPCG<ScalarType,MV,OP>::getState() const {
    LOBPCGState<ScalarType,MV> state;
    state.V = V_;
    state.KV = KV_;
    state.X = X_;
    state.KX = KX_;
    state.P = P_;
    state.KP = KP_;
    state.H = H_;
    state.KH = KH_;
    state.R = R_;
    state.T = Teuchos::rcp(new std::vector<MagnitudeType>(theta_));
    if (hasM_) {
      state.MV = MV_;
      state.MX = MX_;
      state.MP = MP_;
      state.MH = MH_;
    }
    else {
      state.MX = Teuchos::null;
      state.MP = Teuchos::null;
      state.MH = Teuchos::null;
    }
    return state;
  }

} // end Anasazi namespace

#endif // ANASAZI_LOBPCG_HPP
