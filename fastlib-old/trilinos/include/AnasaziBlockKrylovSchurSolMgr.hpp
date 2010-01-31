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

#ifndef ANASAZI_BLOCK_KRYLOV_SCHUR_SOLMGR_HPP
#define ANASAZI_BLOCK_KRYLOV_SCHUR_SOLMGR_HPP

/*! \file AnasaziBlockKrylovSchurSolMgr.hpp
 *  \brief The Anasazi::BlockKrylovSchurSolMgr provides a solver manager for the BlockKrylovSchur eigensolver.
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"

#include "AnasaziEigenproblem.hpp"
#include "AnasaziSolverManager.hpp"

#include "AnasaziBlockKrylovSchur.hpp"
#include "AnasaziBasicSort.hpp"
#include "AnasaziSVQBOrthoManager.hpp"
#include "AnasaziBasicOrthoManager.hpp"
#include "AnasaziStatusTestMaxIters.hpp"
#include "AnasaziStatusTestResNorm.hpp"
#include "AnasaziStatusTestOrderedResNorm.hpp"
#include "AnasaziStatusTestCombo.hpp"
#include "AnasaziStatusTestOutput.hpp"
#include "AnasaziBasicOutputManager.hpp"
#include "AnasaziSolverUtils.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_TimeMonitor.hpp"

/** \example BlockKrylovSchur/BlockKrylovSchurEpetraEx.cpp
    This is an example of how to use the Anasazi::BlockKrylovSchurSolMgr solver manager.
*/

/*! \class Anasazi::BlockKrylovSchurSolMgr
 *
 *  \brief The Anasazi::BlockKrylovSchurSolMgr provides a powerful and fully-featured solver manager over the BlockKrylovSchur eigensolver.

 \ingroup anasazi_solver_framework

 \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, Heidi Thornquist
 */

namespace Anasazi {

template<class ScalarType, class MV, class OP>
class BlockKrylovSchurSolMgr : public SolverManager<ScalarType,MV,OP> {

  private:
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef Teuchos::ScalarTraits<MagnitudeType> MT;
    
  public:

  //! @name Constructors/Destructor
  //@{ 

  /*! \brief Basic constructor for BlockKrylovSchurSolMgr.
   *
   * This constructor accepts the Eigenproblem to be solved in addition
   * to a parameter list of options for the solver manager. These options include the following:
   *   - "Which" - a \c string specifying the desired eigenvalues: SM, LM, SR or LR. Default: "LM"
   *   - "Block Size" - a \c int specifying the block size to be used by the underlying block Krylov-Schur solver. Default: 1
   *   - "Num Blocks" - a \c int specifying the number of blocks allocated for the Krylov basis. Default: 3*nev
   *   - "Extra NEV Blocks" - a \c int specifying the number of extra blocks the solver should keep in addition to those
          required to compute the number of eigenvalues requested.  Default: 0
   *   - "Maximum Restarts" - a \c int specifying the maximum number of restarts the underlying solver is allowed to perform. Default: 20
   *   - "Orthogonalization" - a \c string specifying the desired orthogonalization:  DGKS and SVQB. Default: "SVQB"
   *   - "Verbosity" - a sum of MsgType specifying the verbosity. Default: Anasazi::Errors
   *   - "Convergence Tolerance" - a \c MagnitudeType specifying the level that residual norms must reach to decide convergence. Default: machine precision.
   *   - "Relative Convergence Tolerance" - a \c bool specifying whether residuals norms should be scaled by their eigenvalues for the purposing of deciding convergence. Default: true
   */
  BlockKrylovSchurSolMgr( const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
                             Teuchos::ParameterList &pl );

  //! Destructor.
  virtual ~BlockKrylovSchurSolMgr() {};
  //@}
  
  //! @name Accessor methods
  //@{ 

  const Eigenproblem<ScalarType,MV,OP>& getProblem() const {
    return *_problem;
  }

  /*! \brief Return the Ritz values from the most recent solve.
   */
  std::vector<Value<ScalarType> > getRitzValues() const {
    std::vector<Value<ScalarType> > ret( _ritzValues );
    return ret;
  }

  /*! \brief Return the timers for this object. 
   *
   * The timers are ordered as follows:
   *   - time spent in solve() routine
   *   - time spent restarting
   */
   Teuchos::Array<Teuchos::RCP<Teuchos::Time> > getTimers() const {
     return tuple(_timerSolve, _timerRestarting);
   }

  //@}

  //! @name Solver application methods
  //@{ 
    
  /*! \brief This method performs possibly repeated calls to the underlying eigensolver's iterate() routine
   * until the problem has been solved (as decided by the solver manager) or the solver manager decides to 
   * quit.
   *
   * This method calls BlockKrylovSchur::iterate(), which will return either because a specially constructed status test evaluates to ::Passed
   * or an exception is thrown.
   *
   * A return from BlockKrylovSchur::iterate() signifies one of the following scenarios:
   *    - the maximum number of restarts has been exceeded. In this scenario, the solver manager will place\n
   *      all converged eigenpairs into the eigenproblem and return ::Unconverged.
   *    - global convergence has been met. In this case, the most significant NEV eigenpairs in the solver and locked storage  \n
   *      have met the convergence criterion. (Here, NEV refers to the number of eigenpairs requested by the Eigenproblem.)    \n
   *      In this scenario, the solver manager will return ::Converged.
   *
   * \returns ::ReturnType specifying:
   *     - ::Converged: the eigenproblem was solved to the specification required by the solver manager.
   *     - ::Unconverged: the eigenproblem was not solved to the specification desired by the solver manager.
   */
  ReturnType solve();
  //@}

  private:
  Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > _problem;
  Teuchos::RCP<SortManager<ScalarType,MV,OP> > _sort;

  std::string _whch, _ortho; 
  MagnitudeType _ortho_kappa;

  MagnitudeType _convtol;
  int _maxRestarts;
  bool _relconvtol,_conjSplit;
  int _blockSize, _numBlocks, _stepSize, _nevBlocks, _xtra_nevBlocks;
  int _verbosity;
  bool _inSituRestart;

  std::vector<Value<ScalarType> > _ritzValues;

  Teuchos::RCP<Teuchos::Time> _timerSolve, _timerRestarting;

};


// Constructor
template<class ScalarType, class MV, class OP>
BlockKrylovSchurSolMgr<ScalarType,MV,OP>::BlockKrylovSchurSolMgr( 
        const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
        Teuchos::ParameterList &pl ) : 
  _problem(problem),
  _whch("LM"),
  _ortho("SVQB"),
  _ortho_kappa(-1.0),
  _convtol(0),
  _maxRestarts(20),
  _relconvtol(true),
  _conjSplit(false),
  _blockSize(0),
  _numBlocks(0),
  _stepSize(0),
  _nevBlocks(0),
  _xtra_nevBlocks(0),
  _verbosity(Anasazi::Errors),
  _inSituRestart(false),
  _timerSolve(Teuchos::TimeMonitor::getNewTimer("BKSSolMgr::solve()")),
  _timerRestarting(Teuchos::TimeMonitor::getNewTimer("BKSSolMgr restarting"))
{
  TEST_FOR_EXCEPTION(_problem == Teuchos::null,               std::invalid_argument, "Problem not given to solver manager.");
  TEST_FOR_EXCEPTION(!_problem->isProblemSet(),               std::invalid_argument, "Problem not set.");
  TEST_FOR_EXCEPTION(_problem->getInitVec() == Teuchos::null, std::invalid_argument, "Problem does not contain initial vectors to clone from.");

  const int nev = _problem->getNEV();

  // convergence tolerance
  _convtol = pl.get("Convergence Tolerance",MT::prec());
  _relconvtol = pl.get("Relative Convergence Tolerance",_relconvtol);
  
  // maximum number of restarts
  _maxRestarts = pl.get("Maximum Restarts",_maxRestarts);

  // block size: default is 1
  _blockSize = pl.get("Block Size",1);
  TEST_FOR_EXCEPTION(_blockSize <= 0, std::invalid_argument,
                     "Anasazi::BlockKrylovSchurSolMgr: \"Block Size\" must be strictly positive.");

  // set the number of blocks we need to save to compute the nev eigenvalues of interest.
  _xtra_nevBlocks = pl.get("Extra NEV Blocks",0);
  if (nev%_blockSize) {
    _nevBlocks = nev/_blockSize + _xtra_nevBlocks + 1;
  } else {
    _nevBlocks = nev/_blockSize + _xtra_nevBlocks;
  }

  _numBlocks = pl.get("Num Blocks",3*_nevBlocks);
  TEST_FOR_EXCEPTION(_numBlocks <= _nevBlocks, std::invalid_argument,
                     "Anasazi::BlockKrylovSchurSolMgr: \"Num Blocks\" must be strictly positive and large enough to compute the requested eigenvalues.");

  TEST_FOR_EXCEPTION(_numBlocks*_blockSize > MVT::GetVecLength(*_problem->getInitVec()),
                     std::invalid_argument,
                     "Anasazi::BlockKrylovSchurSolMgr: Potentially impossible orthogonality requests. Reduce basis size.");
  
  // step size: the default is _maxRestarts*_numBlocks, so that Ritz values are only computed every restart.
  if (_maxRestarts) {
    _stepSize = pl.get("Step Size", (_maxRestarts+1)*(_numBlocks+1));
  } else {
    _stepSize = pl.get("Step Size", _numBlocks+1);
  }
  TEST_FOR_EXCEPTION(_stepSize < 1, std::invalid_argument,
                     "Anasazi::BlockKrylovSchurSolMgr: \"Step Size\" must be strictly positive.");

  // get the sort manager
  if (pl.isParameter("Sort Manager")) {
    _sort = Teuchos::getParameter<Teuchos::RCP<Anasazi::SortManager<ScalarType,MV,OP> > >(pl,"Sort Manager");
  } else {
    // which values to solve for
    _whch = pl.get("Which",_whch);
    TEST_FOR_EXCEPTION(_whch != "SM" && _whch != "LM" && _whch != "SR" && _whch != "LR" && _whch != "SI" && _whch != "LI",
                       std::invalid_argument, "Invalid sorting string.");
    _sort = Teuchos::rcp( new BasicSort<ScalarType,MV,OP>(_whch) );
  }

  // which orthogonalization to use
  _ortho = pl.get("Orthogonalization",_ortho);
  if (_ortho != "DGKS" && _ortho != "SVQB") {
    _ortho = "SVQB";
  }

  // which orthogonalization constant to use
  _ortho_kappa = pl.get("Orthogonalization Constant",_ortho_kappa);

  // verbosity level
  if (pl.isParameter("Verbosity")) {
    if (Teuchos::isParameterType<int>(pl,"Verbosity")) {
      _verbosity = pl.get("Verbosity", _verbosity);
    } else {
      _verbosity = (int)Teuchos::getParameter<Anasazi::MsgType>(pl,"Verbosity");
    }
  }

  // restarting technique: V*Q or applyHouse(V,H,tau)
  if (pl.isParameter("In Situ Restarting")) {
    if (Teuchos::isParameterType<bool>(pl,"In Situ Restarting")) {
      _inSituRestart = pl.get("In Situ Restarting",_inSituRestart);
    } else {
      _inSituRestart = (bool)Teuchos::getParameter<int>(pl,"In Situ Restarting");
    }
  }
}


// solve()
template<class ScalarType, class MV, class OP>
ReturnType 
BlockKrylovSchurSolMgr<ScalarType,MV,OP>::solve() {

  const int nev = _problem->getNEV();
  ScalarType one = Teuchos::ScalarTraits<ScalarType>::one();
  ScalarType zero = Teuchos::ScalarTraits<ScalarType>::zero();

  Teuchos::BLAS<int,ScalarType> blas;
  Teuchos::LAPACK<int,ScalarType> lapack;
  typedef SolverUtils<ScalarType,MV,OP> msutils;

  //////////////////////////////////////////////////////////////////////////////////////
  // Output manager
  Teuchos::RCP<BasicOutputManager<ScalarType> > printer = Teuchos::rcp( new BasicOutputManager<ScalarType>(_verbosity) );

  //////////////////////////////////////////////////////////////////////////////////////
  // Status tests
  //
  // convergence
  Teuchos::RCP<StatusTestOrderedResNorm<ScalarType,MV,OP> > convtest 
    = Teuchos::rcp( new StatusTestOrderedResNorm<ScalarType,MV,OP>(_sort,_convtol,nev,StatusTestOrderedResNorm<ScalarType,MV,OP>::RITZRES_2NORM,_relconvtol) );

  // printing StatusTest
  Teuchos::RCP<StatusTestOutput<ScalarType,MV,OP> > outputtest
    = Teuchos::rcp( new StatusTestOutput<ScalarType,MV,OP>( printer,convtest,1,Passed ) );

  //////////////////////////////////////////////////////////////////////////////////////
  // Orthomanager
  //
  Teuchos::RCP<OrthoManager<ScalarType,MV> > ortho; 
  if (_ortho=="SVQB") {
    ortho = Teuchos::rcp( new SVQBOrthoManager<ScalarType,MV,OP>(_problem->getM()) );
  } else if (_ortho=="DGKS") {
    if (_ortho_kappa <= 0) {
      ortho = Teuchos::rcp( new BasicOrthoManager<ScalarType,MV,OP>(_problem->getM()) );
    }
    else {
      ortho = Teuchos::rcp( new BasicOrthoManager<ScalarType,MV,OP>(_problem->getM(),_ortho_kappa) );
    }
  } else {
    TEST_FOR_EXCEPTION(_ortho!="SVQB"&&_ortho!="DGKS",std::logic_error,"Anasazi::BlockKrylovSchurSolMgr::solve(): Invalid orthogonalization type.");
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Parameter list
  Teuchos::ParameterList plist;
  plist.set("Block Size",_blockSize);
  plist.set("Num Blocks",_numBlocks);
  plist.set("Step Size",_stepSize);
  plist.set("Print Number of Ritz Values",_nevBlocks*_blockSize);

  //////////////////////////////////////////////////////////////////////////////////////
  // BlockKrylovSchur solver
  Teuchos::RCP<BlockKrylovSchur<ScalarType,MV,OP> > bks_solver 
    = Teuchos::rcp( new BlockKrylovSchur<ScalarType,MV,OP>(_problem,_sort,printer,outputtest,ortho,plist) );
  // set any auxiliary vectors defined in the problem
  Teuchos::RCP< const MV > probauxvecs = _problem->getAuxVecs();
  if (probauxvecs != Teuchos::null) {
    bks_solver->setAuxVecs( Teuchos::tuple< Teuchos::RCP<const MV> >(probauxvecs) );
  }

  // Create workspace for the Krylov basis generated during a restart
  // Need at most (_nevBlocks*_blockSize+1) for the updated factorization and another block for the current factorization residual block (F).
  //  ---> (_nevBlocks*_blockSize+1) + _blockSize
  // If Hermitian, this becomes _nevBlocks*_blockSize + _blockSize
  // we only need this if there is the possibility of restarting, ex situ
  Teuchos::RCP<MV> workMV;
  if (_maxRestarts > 0) {
    if (_inSituRestart==true) {
      // still need one work vector for applyHouse()
      workMV = MVT::Clone( *_problem->getInitVec(), 1 );
    }
    else { // inSituRestart == false
      if (_problem->isHermitian()) {
        workMV = MVT::Clone( *_problem->getInitVec(), _nevBlocks*_blockSize + _blockSize );
      } else {
        workMV = MVT::Clone( *_problem->getInitVec(), _nevBlocks*_blockSize+1 + _blockSize );
      }
    }
  } else {
    workMV = Teuchos::null;
  }

  // go ahead and initialize the solution to nothing in case we throw an exception
  Eigensolution<ScalarType,MV> sol;
  sol.numVecs = 0;
  _problem->setSolution(sol);

  int numRestarts = 0;
  int cur_nevBlocks = 0;

  // enter solve() iterations
  {
    Teuchos::TimeMonitor slvtimer(*_timerSolve);
  
    // tell bks_solver to iterate
    while (1) {
      try {
        bks_solver->iterate();
    
        ////////////////////////////////////////////////////////////////////////////////////
        //
        // check convergence first
        //
        ////////////////////////////////////////////////////////////////////////////////////
        if (convtest->getStatus() == Passed ) {
          // we have convergence
          // convtest->whichVecs() tells us which vectors from solver state are the ones we want
          // convtest->howMany() will tell us how many
          break;
        }
        ////////////////////////////////////////////////////////////////////////////////////
        //
        // check for restarting, i.e. the subspace is full
        //
        ////////////////////////////////////////////////////////////////////////////////////
        // this is for the Hermitian case, or non-Hermitian conjugate split situation.
        // --> for the Hermitian case the current subspace dimension needs to match the maximum subspace dimension
        // --> for the non-Hermitian case:
        //     --> if a conjugate pair was detected in the previous restart then the current subspace dimension needs to match the
        //         maximum subspace dimension (the BKS solver keeps one extra vector if the problem is non-Hermitian).
        //     --> if a conjugate pair was not detected in the previous restart then the current subspace dimension will be one less
        //         than the maximum subspace dimension.
        else if ( (bks_solver->getCurSubspaceDim() == bks_solver->getMaxSubspaceDim()) ||
                  (!_problem->isHermitian() && !_conjSplit && (bks_solver->getCurSubspaceDim()+1 == bks_solver->getMaxSubspaceDim())) ) {
  
          Teuchos::TimeMonitor restimer(*_timerRestarting);
  
          if ( numRestarts >= _maxRestarts ) {
            break; // break from while(1){bks_solver->iterate()}
          }
          numRestarts++;
  
          printer->stream(Debug) << " Performing restart number " << numRestarts << " of " << _maxRestarts << std::endl << std::endl;
  
          // Update the Schur form of the projected eigenproblem, then sort it.
          if (!bks_solver->isSchurCurrent())
            bks_solver->computeSchurForm( true );

          // Get the most current Ritz values before we continue.
          _ritzValues = bks_solver->getRitzValues();

          // Get the state.
          BlockKrylovSchurState<ScalarType,MV> oldState = bks_solver->getState();

          // Get the current dimension of the factorization
          int curDim = oldState.curDim;

          // Determine if the storage for the nev eigenvalues of interest splits a complex conjugate pair.
          std::vector<int> ritzIndex = bks_solver->getRitzIndex();
          if (ritzIndex[_nevBlocks*_blockSize-1]==1) {
            _conjSplit = true;
            cur_nevBlocks = _nevBlocks*_blockSize+1;
          } else {
            _conjSplit = false;
            cur_nevBlocks = _nevBlocks*_blockSize;
          }

          // Update the Krylov-Schur decomposition

          // Get a view of the Schur vectors of interest.
          Teuchos::SerialDenseMatrix<int,ScalarType> Qnev(Teuchos::View, *(oldState.Q), curDim, cur_nevBlocks);

          // Get a view of the current Krylov basis.
          std::vector<int> curind( curDim );
          for (int i=0; i<curDim; i++) { curind[i] = i; }
          Teuchos::RCP<const MV> basistemp = MVT::CloneView( *(oldState.V), curind );

          // Compute the new Krylov basis: Vnew = V*Qnev
          // 
          // this will occur ex situ in workspace allocated for this purpose (tmpMV)
          // or in situ in the solver's memory space.
          //
          // we will also set a pointer for the location that the current factorization residual block (F),
          // currently located after the current basis in oldstate.V, will be moved to
          //
          Teuchos::RCP<MV> newF;
          if (_inSituRestart) {
            //
            // get non-const pointer to solver's basis so we can work in situ
            Teuchos::RCP<MV> solverbasis = Teuchos::rcp_const_cast<MV>(oldState.V);
            Teuchos::SerialDenseMatrix<int,ScalarType> copyQnev(Qnev);
            // 
            // perform Householder QR of copyQnev = Q [D;0], where D is unit diag. We will want D below.
            std::vector<ScalarType> tau(cur_nevBlocks), work(cur_nevBlocks);
            int info;
            lapack.GEQRF(curDim,cur_nevBlocks,copyQnev.values(),copyQnev.stride(),&tau[0],&work[0],work.size(),&info);
            TEST_FOR_EXCEPTION(info != 0,std::logic_error,
                               "Anasazi::BlockDavidsonSolMgr::solve(): error calling GEQRF during restarting.");
            // we need to get the diagonal of D
            std::vector<ScalarType> d(cur_nevBlocks);
            for (int j=0; j<copyQnev.numCols(); j++) {
              d[j] = copyQnev(j,j);
            }
            if (printer->isVerbosity(Debug)) {
              Teuchos::SerialDenseMatrix<int,ScalarType> R(Teuchos::Copy,copyQnev,cur_nevBlocks,cur_nevBlocks);
              for (int j=0; j<R.numCols(); j++) {
                R(j,j) = SCT::magnitude(R(j,j)) - 1.0;
                for (int i=j+1; i<R.numRows(); i++) {
                  R(i,j) = zero;
                }
              }
              printer->stream(Debug) << "||Triangular factor of Su - I||: " << R.normFrobenius() << std::endl;
            }
            // 
            // perform implicit V*Qnev
            // this actually performs V*[Qnev Qtrunc*M] = [newV truncV], for some unitary M
            // we are interested in only the first cur_nevBlocks vectors of the result
            curind.resize(curDim);
            for (int i=0; i<curDim; i++) curind[i] = i;
            Teuchos::RCP<MV> oldV = MVT::CloneView(*solverbasis,curind);
            msutils::applyHouse(cur_nevBlocks,*oldV,copyQnev,tau,workMV);
            // clear pointer
            oldV = Teuchos::null;
            // multiply newV*D
            // get pointer to new basis
            curind.resize(cur_nevBlocks);
            for (int i=0; i<cur_nevBlocks; i++) { curind[i] = i; }
            oldV = MVT::CloneView( *solverbasis, curind );
            MVT::MvScale(*oldV,d);
            oldV = Teuchos::null;
            // get pointer to new location for F
            curind.resize(_blockSize);
            for (int i=0; i<_blockSize; i++) { curind[i] = cur_nevBlocks + i; }
            newF = MVT::CloneView( *solverbasis, curind );
          }
          else {
            // get pointer to first part of work space
            curind.resize(cur_nevBlocks);
            for (int i=0; i<cur_nevBlocks; i++) { curind[i] = i; }
            Teuchos::RCP<MV> tmp_newV = MVT::CloneView(*workMV, curind );
            // perform V*Qnev
            MVT::MvTimesMatAddMv( one, *basistemp, Qnev, zero, *tmp_newV );
            tmp_newV = Teuchos::null;
            // get pointer to new location for F
            curind.resize(_blockSize);
            for (int i=0; i<_blockSize; i++) { curind[i] = cur_nevBlocks + i; }
            newF = MVT::CloneView( *workMV, curind );
          }

          // Move the current factorization residual block (F) to the last block of newV.
          curind.resize(_blockSize);
          for (int i=0; i<_blockSize; i++) { curind[i] = curDim + i; }
          Teuchos::RCP<const MV> oldF = MVT::CloneView( *(oldState.V), curind );
          for (int i=0; i<_blockSize; i++) { curind[i] = i; }
          MVT::SetBlock( *oldF, curind, *newF );
          newF = Teuchos::null;

          // Update the Krylov-Schur quasi-triangular matrix.
          //
          // Create storage for the new Schur matrix of the Krylov-Schur factorization
          // Copy over the current quasi-triangular factorization of oldState.H which is stored in oldState.S.
          Teuchos::SerialDenseMatrix<int,ScalarType> oldS(Teuchos::View, *(oldState.S), cur_nevBlocks+_blockSize, cur_nevBlocks);
          Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > newH = 
            Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,ScalarType>( oldS ) );
          //
          // Get a view of the B block of the current factorization
          Teuchos::SerialDenseMatrix<int,ScalarType> oldB(Teuchos::View, *(oldState.H), _blockSize, _blockSize, curDim, curDim-_blockSize);
          //
          // Get a view of the a block row of the Schur vectors.
          Teuchos::SerialDenseMatrix<int,ScalarType> subQ(Teuchos::View, *(oldState.Q), _blockSize, cur_nevBlocks, curDim-_blockSize);
          //
          // Get a view of the new B block of the updated Krylov-Schur factorization
          Teuchos::SerialDenseMatrix<int,ScalarType> newB(Teuchos::View, *newH,  _blockSize, cur_nevBlocks, cur_nevBlocks);
          //
          // Compute the new B block.
          blas.GEMM( Teuchos::NO_TRANS, Teuchos::NO_TRANS, _blockSize, cur_nevBlocks, _blockSize, one, 
                     oldB.values(), oldB.stride(), subQ.values(), subQ.stride(), zero, newB.values(), newB.stride() );


          //
          // Set the new state and initialize the solver.
          BlockKrylovSchurState<ScalarType,MV> newstate;
          if (_inSituRestart) {
            newstate.V = oldState.V;
          } else {
            newstate.V = workMV;
          }
          newstate.H = newH;
          newstate.curDim = cur_nevBlocks;
          bks_solver->initialize(newstate);
  
        } // end of restarting
        ////////////////////////////////////////////////////////////////////////////////////
        //
        // we returned from iterate(), but none of our status tests Passed.
        // something is wrong, and it is probably our fault.
        //
        ////////////////////////////////////////////////////////////////////////////////////
        else {
          TEST_FOR_EXCEPTION(true,std::logic_error,"Anasazi::BlockKrylovSchurSolMgr::solve(): Invalid return from bks_solver::iterate().");
        }
      }
      catch (std::exception e) {
        printer->stream(Errors) << "Error! Caught exception in BlockKrylovSchur::iterate() at iteration " << bks_solver->getNumIters() << std::endl 
                                << e.what() << std::endl;
        throw;
      }
    }

    //
    // free temporary space
    workMV = Teuchos::null;
  
    // Get the most current Ritz values before we return
    _ritzValues = bks_solver->getRitzValues();
    
    sol.numVecs = convtest->howMany();
    if (sol.numVecs > 0) {
      sol.index = bks_solver->getRitzIndex();
      sol.Evals = bks_solver->getRitzValues();
      // Check to see if conjugate pair is on the boundary.
      if (sol.index[sol.numVecs-1]==1) {
        sol.numVecs++;
        sol.Evals.resize(sol.numVecs);
        sol.index.resize(sol.numVecs);
        bks_solver->setNumRitzVectors(sol.numVecs);
      } else {
        sol.Evals.resize(sol.numVecs);
        sol.index.resize(sol.numVecs);
        bks_solver->setNumRitzVectors(sol.numVecs);
      }
      bks_solver->computeRitzVectors();
      sol.Evecs = MVT::CloneCopy( *(bks_solver->getRitzVectors()) );
      sol.Espace = sol.Evecs;
    } 
  }

  // print final summary
  bks_solver->currentStatus(printer->stream(FinalSummary));

  // print timing information
  Teuchos::TimeMonitor::summarize(printer->stream(TimingDetails));

  _problem->setSolution(sol);
  printer->stream(Debug) << "Returning " << sol.numVecs << " eigenpairs to eigenproblem." << std::endl;

  if (sol.numVecs < nev) {
    return Unconverged; // return from BlockKrylovSchurSolMgr::solve() 
  }
  return Converged; // return from BlockKrylovSchurSolMgr::solve() 
}


} // end Anasazi namespace

#endif /* ANASAZI_BLOCK_KRYLOV_SCHUR_SOLMGR_HPP */
