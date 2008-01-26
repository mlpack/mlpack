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

#ifndef ANASAZI_BLOCKDAVIDSON_SOLMGR_HPP
#define ANASAZI_BLOCKDAVIDSON_SOLMGR_HPP

/*! \file AnasaziBlockDavidsonSolMgr.hpp
 *  \brief The Anasazi::BlockDavidsonSolMgr provides a solver manager for the BlockDavidson eigensolver.
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"

#include "AnasaziEigenproblem.hpp"
#include "AnasaziSolverManager.hpp"
#include "AnasaziSolverUtils.hpp"

#include "AnasaziBlockDavidson.hpp"
#include "AnasaziBasicSort.hpp"
#include "AnasaziSVQBOrthoManager.hpp"
#include "AnasaziStatusTestMaxIters.hpp"
#include "AnasaziStatusTestResNorm.hpp"
#include "AnasaziStatusTestOrderedResNorm.hpp"
#include "AnasaziStatusTestCombo.hpp"
#include "AnasaziStatusTestOutput.hpp"
#include "AnasaziBasicOutputManager.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_LAPACK.hpp"


/** \example BlockDavidson/BlockDavidsonEpetraEx.cpp
    This is an example of how to use the Anasazi::BlockDavidsonSolMgr solver manager to solve a standard eigenvalue problem.
*/

/** \example BlockDavidson/BlockDavidsonEpetraExGen.cpp
    This is an example of how to use the Anasazi::BlockDavidsonSolMgr solver manager to solve a generalized eigenvalue problem.
*/

/*! \class Anasazi::BlockDavidsonSolMgr
 *
 *  \brief The Anasazi::BlockDavidsonSolMgr provides a powerful and fully-featured solver manager over the BlockDavidson eigensolver.
 *
 * This solver manager implements a hard-locking mechanism, whereby eigenpairs designated to be locked are moved from the eigensolver and placed in
 * auxiliary storage. The eigensolver is then restarted and continues to iterate, always orthogonal to the locked eigenvectors.

 \ingroup anasazi_solver_framework

 \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, Heidi Thornquist
 */

namespace Anasazi {

template<class ScalarType, class MV, class OP>
class BlockDavidsonSolMgr : public SolverManager<ScalarType,MV,OP> {

  private:
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef OperatorTraits<ScalarType,MV,OP> OPT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef Teuchos::ScalarTraits<MagnitudeType> MT;
    
  public:

  //! @name Constructors/Destructor
  //@{ 

  /*! \brief Basic constructor for BlockDavidsonSolMgr.
   *
   * This constructor accepts the Eigenproblem to be solved in addition
   * to a parameter list of options for the solver manager. These options include the following:
   *   - "Which" - a \c string specifying the desired eigenvalues: SM, LM, SR or LR. Default: "SR"
   *   - "Block Size" - a \c int specifying the block size to be used by the underlying block Davidson solver. Default: problem->getNEV()
   *   - "Num Blocks" - a \c int specifying the number of blocks allocated for the Krylov basis. Default: 2
   *   - "Maximum Restarts" - a \c int specifying the maximum number of restarts the underlying solver is allowed to perform. Default: 20
   *   - "Verbosity" - a sum of MsgType specifying the verbosity. Default: Anasazi::Errors
   *   - "Convergence Tolerance" - a \c MagnitudeType specifying the level that residual norms must reach to decide convergence. Default: machine precision.
   *   - "Relative Convergence Tolerance" - a \c bool specifying whether residuals norms should be scaled by their eigenvalues for the purposing of deciding convergence. Default: true
   *   - "Use Locking" - a \c bool specifying whether the algorithm should employ locking of converged eigenpairs. Default: false
   *   - "Max Locked" - a \c int specifying the maximum number of eigenpairs to be locked. Default: problem->getNEV()
   *   - "Locking Quorum" - a \c int specifying the number of eigenpairs that must meet the locking criteria before locking actually occurs. Default: 1
   *   - "Locking Tolerance" - a \c MagnitudeType specifying the level that residual norms must reach to decide locking. Default: 0.1*convergence tolerance
   *   - "Relative Locking Tolerance" - a \c bool specifying whether residuals norms should be scaled by their eigenvalues for the purposing of deciding locking. Default: true
   */
  BlockDavidsonSolMgr( const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
                             Teuchos::ParameterList &pl );

  //! Destructor.
  virtual ~BlockDavidsonSolMgr() {};
  //@}
  
  //! @name Accessor methods
  //@{ 

  const Eigenproblem<ScalarType,MV,OP>& getProblem() const {
    return *problem_;
  }

  //@}

  //! @name Solver application methods
  //@{ 
    
  /*! \brief This method performs possibly repeated calls to the underlying eigensolver's iterate() routine
   * until the problem has been solved (as decided by the solver manager) or the solver manager decides to 
   * quit.
   *
   * This method calls BlockDavidson::iterate(), which will return either because a specially constructed status test evaluates to ::Passed
   * or an exception is thrown.
   *
   * A return from BlockDavidson::iterate() signifies one of the following scenarios:
   *    - the maximum number of restarts has been exceeded. In this scenario, the solver manager will place\n
   *      all converged eigenpairs into the eigenproblem and return ::Unconverged.
   *    - the locking conditions have been met. In this scenario, some of the current eigenpairs will be removed\n
   *      from the eigensolver and placed into auxiliary storage. The eigensolver will be restarted with the remaining part of the Krylov subspace\n
   *      and some random information to replace the removed subspace.
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
  Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > problem_;

  std::string whch_; 

  MagnitudeType convtol_, locktol_;
  int maxRestarts_;
  bool useLocking_;
  bool relconvtol_, rellocktol_;
  int blockSize_, numBlocks_;
  int maxLocked_;
  int verbosity_;
  int lockQuorum_;
  bool inSituRestart_;
  int numRestartBlocks_;
};


// Constructor
template<class ScalarType, class MV, class OP>
BlockDavidsonSolMgr<ScalarType,MV,OP>::BlockDavidsonSolMgr( 
        const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
        Teuchos::ParameterList &pl ) : 
  problem_(problem),
  whch_("SR"),
  convtol_(0),
  locktol_(0),
  maxRestarts_(20),
  useLocking_(false),
  relconvtol_(true),
  rellocktol_(true),
  blockSize_(0),
  numBlocks_(0),
  maxLocked_(0),
  verbosity_(Anasazi::Errors),
  lockQuorum_(1),
  inSituRestart_(false),
  numRestartBlocks_(1)
{
  TEST_FOR_EXCEPTION(problem_ == Teuchos::null,               std::invalid_argument, "Problem not given to solver manager.");
  TEST_FOR_EXCEPTION(!problem_->isProblemSet(),               std::invalid_argument, "Problem not set.");
  TEST_FOR_EXCEPTION(!problem_->isHermitian(),                std::invalid_argument, "Problem not symmetric.");
  TEST_FOR_EXCEPTION(problem_->getInitVec() == Teuchos::null, std::invalid_argument, "Problem does not contain initial vectors to clone from.");

  // which values to solve for
  whch_ = pl.get("Which",whch_);
  TEST_FOR_EXCEPTION(whch_ != "SM" && whch_ != "LM" && whch_ != "SR" && whch_ != "LR",std::invalid_argument, "Invalid sorting string.");

  // convergence tolerance
  convtol_ = pl.get("Convergence Tolerance",MT::prec());
  relconvtol_ = pl.get("Relative Convergence Tolerance",relconvtol_);
  
  // locking tolerance
  useLocking_ = pl.get("Use Locking",useLocking_);
  rellocktol_ = pl.get("Relative Locking Tolerance",rellocktol_);
  locktol_ = pl.get("Locking Tolerance",convtol_/10.0);

  // maximum number of restarts
  maxRestarts_ = pl.get("Maximum Restarts",maxRestarts_);

  // block size: default is nev()
  blockSize_ = pl.get("Block Size",problem_->getNEV());
  TEST_FOR_EXCEPTION(blockSize_ <= 0, std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: \"Block Size\" must be strictly positive.");
  numBlocks_ = pl.get("Num Blocks",2);
  TEST_FOR_EXCEPTION(numBlocks_ <= 1, std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: \"Num Blocks\" must be >= 1.");

  // max locked: default is nev(), must satisfy maxLocked_ + blockSize_ >= nev
  if (useLocking_) {
    maxLocked_ = pl.get("Max Locked",problem_->getNEV());
  }
  else {
    maxLocked_ = 0;
  }
  if (maxLocked_ == 0) {
    useLocking_ = false;
  }
  TEST_FOR_EXCEPTION(maxLocked_ < 0, std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: \"Max Locked\" must be positive.");
  TEST_FOR_EXCEPTION(maxLocked_ + blockSize_ < problem_->getNEV(), 
                     std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: Not enough storage space for requested number of eigenpairs.");
  TEST_FOR_EXCEPTION(numBlocks_*blockSize_ + maxLocked_ > MVT::GetVecLength(*problem_->getInitVec()),
                     std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: Potentially impossible orthogonality requests. Reduce basis size or locking size.");

  if (useLocking_) {
    lockQuorum_ = pl.get("Locking Quorum",lockQuorum_);
    TEST_FOR_EXCEPTION(lockQuorum_ <= 0,
                       std::invalid_argument,
                       "Anasazi::BlockDavidsonSolMgr: \"Locking Quorum\" must be strictly positive.");
  }

  // verbosity level
  if (pl.isParameter("Verbosity")) {
    if (Teuchos::isParameterType<int>(pl,"Verbosity")) {
      verbosity_ = pl.get("Verbosity", verbosity_);
    } else {
      verbosity_ = (int)Teuchos::getParameter<Anasazi::MsgType>(pl,"Verbosity");
    }
  }

  // restart size
  numRestartBlocks_ = pl.get("Num Restart Blocks",numRestartBlocks_);
  TEST_FOR_EXCEPTION(numRestartBlocks_ <= 0, std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: \"Num Restart Blocks\" must be strictly positive.");
  TEST_FOR_EXCEPTION(numRestartBlocks_ >= numBlocks_, std::invalid_argument,
                     "Anasazi::BlockDavidsonSolMgr: \"Num Restart Blocks\" must be strictly less than \"Num Blocks\".");

  // restarting technique: V*Q or applyHouse(V,H,tau)
  if (pl.isParameter("In Situ Restarting")) {
    if (Teuchos::isParameterType<bool>(pl,"In Situ Restarting")) {
      inSituRestart_ = pl.get("In Situ Restarting",inSituRestart_);
    } else {
      inSituRestart_ = (bool)Teuchos::getParameter<int>(pl,"In Situ Restarting");
    }
  }
}


// solve()
template<class ScalarType, class MV, class OP>
ReturnType 
BlockDavidsonSolMgr<ScalarType,MV,OP>::solve() {

  typedef SolverUtils<ScalarType,MV,OP> msutils;

  const int nev = problem_->getNEV();

  //////////////////////////////////////////////////////////////////////////////////////
  // Sort manager
  Teuchos::RCP<BasicSort<ScalarType,MV,OP> > sorter = Teuchos::rcp( new BasicSort<ScalarType,MV,OP>(whch_) );

  //////////////////////////////////////////////////////////////////////////////////////
  // Output manager
  Teuchos::RCP<BasicOutputManager<ScalarType> > printer = Teuchos::rcp( new BasicOutputManager<ScalarType>(verbosity_) );

  //////////////////////////////////////////////////////////////////////////////////////
  // Status tests
  //
  // convergence
  Teuchos::RCP<StatusTestOrderedResNorm<ScalarType,MV,OP> > convtest 
      = Teuchos::rcp( new StatusTestOrderedResNorm<ScalarType,MV,OP>(sorter,convtol_,nev,StatusTestOrderedResNorm<ScalarType,MV,OP>::RES_ORTH,relconvtol_) );
  // locking
  Teuchos::RCP<StatusTestResNorm<ScalarType,MV,OP> > locktest;
  if (useLocking_) {
    locktest = Teuchos::rcp( new StatusTestResNorm<ScalarType,MV,OP>(locktol_,lockQuorum_,StatusTestResNorm<ScalarType,MV,OP>::RES_ORTH,rellocktol_) );
  }
  // combo class
  Teuchos::Array<Teuchos::RCP<StatusTest<ScalarType,MV,OP> > > alltests;
  // for an OR test, the order doesn't matter
  alltests.push_back(convtest);
  if (locktest != Teuchos::null)   alltests.push_back(locktest);
  // combo: convergence || locking 
  Teuchos::RCP<StatusTestCombo<ScalarType,MV,OP> > combotest
    = Teuchos::rcp( new StatusTestCombo<ScalarType,MV,OP>( StatusTestCombo<ScalarType,MV,OP>::OR, alltests) );
  // printing StatusTest
  Teuchos::RCP<StatusTestOutput<ScalarType,MV,OP> > outputtest
    = Teuchos::rcp( new StatusTestOutput<ScalarType,MV,OP>( printer,combotest,1,Passed ) );

  //////////////////////////////////////////////////////////////////////////////////////
  // Orthomanager
  Teuchos::RCP<SVQBOrthoManager<ScalarType,MV,OP> > ortho 
    = Teuchos::rcp( new SVQBOrthoManager<ScalarType,MV,OP>(problem_->getM()) );

  //////////////////////////////////////////////////////////////////////////////////////
  // Parameter list
  Teuchos::ParameterList plist;
  plist.set("Block Size",blockSize_);
  plist.set("Num Blocks",numBlocks_);

  //////////////////////////////////////////////////////////////////////////////////////
  // BlockDavidson solver
  Teuchos::RCP<BlockDavidson<ScalarType,MV,OP> > bd_solver 
    = Teuchos::rcp( new BlockDavidson<ScalarType,MV,OP>(problem_,sorter,printer,outputtest,ortho,plist) );
  // set any auxiliary vectors defined in the problem
  Teuchos::RCP< const MV > probauxvecs = problem_->getAuxVecs();
  if (probauxvecs != Teuchos::null) {
    bd_solver->setAuxVecs( Teuchos::tuple< Teuchos::RCP<const MV> >(probauxvecs) );
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Storage
  // for locked vectors
  int curNumLocked = 0;
  Teuchos::RCP<MV> lockvecs;
  // lockvecs is used to hold the locked eigenvectors, as well as for temporary storage when locking.
  // when locking, we will lock some number of vectors numnew, where numnew <= maxlocked - curlocked
  // we will produce numnew random vectors, which will go into the space with the new basis.
  // we will also need numnew storage for the image of these random vectors under A and M; 
  // columns [curlocked+1,curlocked+numnew] will be used for this storage
  if (maxLocked_ > 0) {
    lockvecs = MVT::Clone(*problem_->getInitVec(),maxLocked_);
  }
  std::vector<MagnitudeType> lockvals;
  //
  // Restarting occurs under two scenarios: when the basis is full and after locking.
  //
  // For the former, a new basis of size blockSize*numRestartBlocks is generated using the current basis
  // and the most significant primitive Ritz vectors (projected eigenvectors).
  //     [S,L] = eig(KK)
  //     S = [Sr St]   // some for "r"estarting, some are "t"runcated
  //     newV = V*Sr
  //     KK_new = newV'*K*newV = Sr'*V'*K*V*Sr = Sr'*KK*Sr
  //  Therefore, the only multivector operation needed is for the generation of newV.
  //
  //  * If the multiplication is explicit, it requires a workspace of blockSize*numRestartBlocks vectors. 
  //    This space must be specifically allocated for that task, as we don't have any space of that size.
  //    It (workMV) will be allocated at the beginning of solve()
  //  * Optionally, the multiplication can be performed implicitly, via a Householder QR factorization of 
  //    Sr. This can be done in situ, using the basis multivector contained in the solver. This requires
  //    that we cast away the const on the multivector returned from getState(). Workspace for this approach
  //    is a single vector. the solver's internal storage must be preserved (X,MX,KX,R), requiring us to 
  //    allocate this vector.
  //
  // For the latter (restarting after locking), the new basis is the same size as existing basis. If numnew
  // vectors are locked, they are deflated from the current basis and replaced with randomly generated 
  // vectors.
  //     [S,L] = eig(KK)
  //     S = [Sl Su]  // partitioned: "l"ocked and "u"nlocked
  //     newL = V*Sl = X(locked)
  //     defV = V*Su
  //     augV = rand(numnew)  // orthogonal to oldL,newL,defV,auxvecs
  //     newV = [defV augV]
  //     Kknew = newV'*K*newV = [Su'*KK*Su    defV'*K*augV]
  //                            [augV'*K*defV augV'*K*augV]
  //     locked = [oldL newL]
  // Clearly, this operation is more complicated than the previous.
  // Here is a list of the significant computations that need to be performed:
  // - newL will be put into space in lockvecs, but will be copied from getState().X at the end
  // - defV,augV will be stored in workspace the size of the current basis.
  //   - If inSituRestart==true, we compute defV in situ in bd_solver::V_ and
  //     put augV at the end of bd_solver::V_
  //   - If inSituRestart==false, we must have curDim vectors available for 
  //     defV and augV; we will allocate a multivector (workMV) at the beginning of solve()
  //     for this purpose.
  // - M*augV and K*augV are needed; they will be stored in lockvecs. As a result, newL will
  //   not be put into lockvecs until the end.
  //
  // Therefore, we must allocate workMV when ((maxRestarts_ > 0) || (useLocking_ == true)) && inSituRestart == false
  // It will be allocated to size (numBlocks-1)*blockSize
  //
  Teuchos::RCP<MV> workMV;
  if (inSituRestart_ == false) {
    // we need storage space to restart, either if we may lock or if may restart after a full basis
    if (useLocking_==true || maxRestarts_ > 0) {
      workMV = MVT::Clone(*problem_->getInitVec(),(numBlocks_-1)*blockSize_);
    }
    else {
      // we will never need to restart.
      workMV = Teuchos::null;
    }
  }
  else { // inSituRestart_ == true 
    // we will restart in situ, if we need to restart
    // three situation remain: 
    // - never restart                                       => no space needed
    // - only restart for locking (i.e., never restart full) => no space needed
    // - restart for full basis                              => need one vector
    if (maxRestarts_ > 0) {
      workMV = MVT::Clone(*problem_->getInitVec(),1);
    }
    else {
      workMV = Teuchos::null;
    }
  }

  // some consts and utils
  const ScalarType ONE = SCT::one();
  const ScalarType ZERO = SCT::zero();
  Teuchos::LAPACK<int,ScalarType> lapack;
  Teuchos::BLAS<int,ScalarType> blas;

  // go ahead and initialize the solution to nothing in case we throw an exception
  Eigensolution<ScalarType,MV> sol;
  sol.numVecs = 0;
  problem_->setSolution(sol);

  int numRestarts = 0;

  // tell bd_solver to iterate
  while (1) {
    try {
      bd_solver->iterate();

      ////////////////////////////////////////////////////////////////////////////////////
      //
      // check convergence first
      //
      ////////////////////////////////////////////////////////////////////////////////////
      if (convtest->getStatus() == Passed ) {
        // we have convergence
        // convtest->whichVecs() tells us which vectors from lockvecs and solver state are the ones we want
        // convtest->howMany() will tell us how many
        break;
      }
      ////////////////////////////////////////////////////////////////////////////////////
      //
      // check for restarting before locking: if we need to lock, it will happen after the restart
      //
      ////////////////////////////////////////////////////////////////////////////////////
      else if ( bd_solver->getCurSubspaceDim() == bd_solver->getMaxSubspaceDim() ) {

        if ( numRestarts >= maxRestarts_ ) {
          break; // break from while(1){bd_solver->iterate()}
        }
        numRestarts++;

        printer->stream(IterationDetails) << " Performing restart number " << numRestarts << " of " << maxRestarts_ << std::endl << std::endl;

        BlockDavidsonState<ScalarType,MV> state = bd_solver->getState();
        int curdim = state.curDim;
        int newdim = numRestartBlocks_*blockSize_;

        //
        // compute eigenvectors of the projected problem
        Teuchos::SerialDenseMatrix<int,ScalarType> S(curdim,curdim);
        std::vector<MagnitudeType> theta(curdim);
        int rank = curdim;
        int info = msutils::directSolver(curdim,*state.KK,Teuchos::null,S,theta,rank,10);
        TEST_FOR_EXCEPTION(info != 0     ,std::logic_error,
            "Anasazi::BlockDavidsonSolMgr::solve(): error calling SolverUtils::directSolver.");       // this should never happen
        TEST_FOR_EXCEPTION(rank != curdim,std::logic_error,
            "Anasazi::BlockDavidsonSolMgr::solve(): direct solve did not compute all eigenvectors."); // this should never happen

        // 
        // sort the eigenvalues (so that we can order the eigenvectors)
        {
          std::vector<int> order(curdim);
          sorter->sort(bd_solver.get(),curdim,theta,&order);
          //
          // apply the same ordering to the primitive ritz vectors
          msutils::permuteVectors(order,S);
        }
        //
        // select the significant primitive ritz vectors
        Teuchos::SerialDenseMatrix<int,ScalarType> Sr(Teuchos::View,S,curdim,newdim);
        //
        // generate newKK = Sr'*KKold*Sr
        Teuchos::SerialDenseMatrix<int,ScalarType> newKK(newdim,newdim);
        {
          Teuchos::SerialDenseMatrix<int,ScalarType> KKtmp(curdim,newdim), 
                                                     KKold(Teuchos::View,*state.KK,curdim,curdim);
          int teuchosRet;
          // KKtmp = KKold*Sr
          teuchosRet = KKtmp.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,KKold,Sr,ZERO);
          TEST_FOR_EXCEPTION(teuchosRet != 0,std::logic_error,
                             "Anasazi::BlockDavidsonSolMgr::solve(): Logic error calling SerialDenseMatrix::multiply.");
          // newKK = Sr'*KKtmp = Sr'*KKold*Sr
          teuchosRet = newKK.multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,Sr,KKtmp,ZERO);
          TEST_FOR_EXCEPTION(teuchosRet != 0,std::logic_error,
                             "Anasazi::BlockDavidsonSolMgr::solve(): Logic error calling SerialDenseMatrix::multiply.");
          // make it Hermitian in memory
          for (int j=0; j<newdim; ++j) {
            for (int i=j+1; i<newdim; ++i) {
              newKK(i,j) = SCT::conjugate(newKK(j,i));
            }
          }
        }

        // prepare new state
        BlockDavidsonState<ScalarType,MV> rstate;
        rstate.curDim = newdim;
        rstate.KK = Teuchos::rcp( &newKK, false );
        // 
        // we know that newX = newV*Sr(:,1:bS) = oldV*S(:1:bS) = oldX
        // the restarting preserves the Ritz vectors and residual
        // for the Ritz values, we want all of the values associated with newV. 
        // these have already been placed at the beginning of theta
        rstate.X  = state.X;
        rstate.KX = state.KX;
        rstate.MX = state.MX;
        rstate.R  = state.R;
        rstate.T  = Teuchos::rcp( new std::vector<MagnitudeType>(&theta[0],&theta[newdim]) );

        if (inSituRestart_ == true) {
          //
          // get non-const pointer to solver's basis so we can work in situ
          Teuchos::RCP<MV> solverbasis = Teuchos::rcp_const_cast<MV>(state.V);
          // 
          // perform Householder QR of Sr = Q [D;0], where D is unit diag.
          // WARNING: this will overwrite Sr; however, we do not need Sr anymore after this
          std::vector<ScalarType> tau(newdim), work(newdim);
          int info;
          lapack.GEQRF(curdim,newdim,Sr.values(),Sr.stride(),&tau[0],&work[0],work.size(),&info);
          TEST_FOR_EXCEPTION(info != 0,std::logic_error,
                             "Anasazi::BlockDavidsonSolMgr::solve(): error calling GEQRF during restarting.");
          if (printer->isVerbosity(Debug)) {
            Teuchos::SerialDenseMatrix<int,ScalarType> R(Teuchos::Copy,Sr,newdim,newdim);
            for (int j=0; j<newdim; j++) {
              R(j,j) = SCT::magnitude(R(j,j)) - 1.0;
              for (int i=j+1; i<newdim; i++) {
                R(i,j) = ZERO;
              }
            }
            printer->stream(Debug) << "||Triangular factor of Sr - I||: " << R.normFrobenius() << std::endl;
          }
          // 
          // perform implicit oldV*Sr
          // this actually performs oldV*[Sr Su*M] = [newV truncV], for some unitary M
          // we are actually interested in only the first newdim vectors of the result
          {
            std::vector<int> curind(curdim);
            for (int i=0; i<curdim; i++) curind[i] = i;
            Teuchos::RCP<MV> oldV = MVT::CloneView(*solverbasis,curind);
            msutils::applyHouse(newdim,*oldV,Sr,tau,workMV);
          }
          // 
          // put the new basis into the state for initialize()
          // the new basis is contained in the the first newdim columns of solverbasis
          // initialize() will recognize that pointer bd_solver.V_ == pointer rstate.V, and will neglect the copy.
          rstate.V = solverbasis;
        }
        else { // inSituRestart == false)
          // newV = oldV*Sr, explicitly. workspace is in workMV
          std::vector<int> curind(curdim), newind(newdim);
          for (int i=0; i<curdim; i++) curind[i] = i;
          for (int i=0; i<newdim; i++) newind[i] = i;
          Teuchos::RCP<const MV> oldV = MVT::CloneView(*state.V,curind);
          Teuchos::RCP<MV>       newV = MVT::CloneView(*workMV ,newind);

          MVT::MvTimesMatAddMv(ONE,*oldV,Sr,ZERO,*newV);
          // 
          // put the new basis into the state for initialize()
          rstate.V = newV;
        }

        //
        // send the new state to the solver
        bd_solver->initialize(rstate);
      } // end of restarting
      ////////////////////////////////////////////////////////////////////////////////////
      //
      // check locking if we didn't converge or restart
      //
      ////////////////////////////////////////////////////////////////////////////////////
      else if (locktest != Teuchos::null && locktest->getStatus() == Passed) {

        // 
        // get current state
        BlockDavidsonState<ScalarType,MV> state = bd_solver->getState();
        const int curdim = state.curDim;

        //
        // get number,indices of vectors to be locked
        TEST_FOR_EXCEPTION(locktest->howMany() <= 0,std::logic_error,
            "Anasazi::BlockDavidsonSolMgr::solve(): status test mistake.");
        TEST_FOR_EXCEPTION(locktest->howMany() != (int)locktest->whichVecs().size(),std::logic_error,
            "Anasazi::BlockDavidsonSolMgr::solve(): status test mistake.");
        //
        // don't lock more than maxLocked_; we didn't allocate enough space.
        std::vector<int> tmp_vector_int;
        if (curNumLocked + locktest->howMany() > maxLocked_) {
          // just use the first of them
          tmp_vector_int.insert(tmp_vector_int.begin(),locktest->whichVecs().begin(),locktest->whichVecs().begin()+maxLocked_-curNumLocked);
        }
        else {
          tmp_vector_int = locktest->whichVecs();
        }
        const std::vector<int> lockind(tmp_vector_int);
        const int numNewLocked = lockind.size();
        //
        // generate indices of vectors left unlocked
        // curind = [0,...,curdim-1] = UNION( lockind, unlockind )
        const int numUnlocked = curdim-numNewLocked;
        tmp_vector_int.resize(curdim);
        for (int i=0; i<curdim; i++) tmp_vector_int[i] = i;
        const std::vector<int> curind(tmp_vector_int);       // curind = [0 ... curdim-1]
        tmp_vector_int.resize(numUnlocked); 
        set_difference(curind.begin(),curind.end(),lockind.begin(),lockind.end(),tmp_vector_int.begin());
        const std::vector<int> unlockind(tmp_vector_int);    // unlockind = [0 ... curdim-1] - lockind
        tmp_vector_int.clear();

        //
        // debug printing
        if (printer->isVerbosity(Debug)) {
          printer->print(Debug,"Locking vectors: ");
          for (unsigned int i=0; i<lockind.size(); i++) {printer->stream(Debug) << " " << lockind[i];}
          printer->print(Debug,"\n");
        }

        //
        // we need primitive ritz vectors/values:
        // [S,L] = eig(oldKK)
        //
        // this will be partitioned as follows:
        //   locked: Sl = S(lockind)      // we won't actually need Sl
        // unlocked: Su = S(unlockind)
        //
        Teuchos::SerialDenseMatrix<int,ScalarType> S(curdim,curdim);
        std::vector<MagnitudeType> theta(curdim);
        {
          int rank = curdim;
          int info = msutils::directSolver(curdim,*state.KK,Teuchos::null,S,theta,rank,10);
          TEST_FOR_EXCEPTION(info != 0     ,std::logic_error,
              "Anasazi::BlockDavidsonSolMgr::solve(): error calling SolverUtils::directSolver.");       // this should never happen
          TEST_FOR_EXCEPTION(rank != curdim,std::logic_error,
              "Anasazi::BlockDavidsonSolMgr::solve(): direct solve did not compute all eigenvectors."); // this should never happen
          // 
          // sort the eigenvalues (so that we can order the eigenvectors)
          std::vector<int> order(curdim);
          sorter->sort(bd_solver.get(),curdim,theta,&order);
          //
          // apply the same ordering to the primitive ritz vectors
          msutils::permuteVectors(order,S);
        }
        //
        // select the unlocked ritz vectors
        // the indexing in unlockind is relative to the ordered primitive ritz vectors
        // (this is why we ordered theta,S above)
        Teuchos::SerialDenseMatrix<int,ScalarType> Su(curdim,numUnlocked);
        for (int i=0; i<numUnlocked; i++) {
          blas.COPY(curdim, S[unlockind[i]], 1, Su[i], 1);
        }


        //
        // newV has the following form:
        // newV = [defV augV]
        // - defV will be of size curdim - numNewLocked, and contain the generated basis: defV = oldV*Su
        // - augV will be of size numNewLocked, and contain random directions to make up for the lost space
        //
        // we will need a pointer to defV below to generate the off-diagonal block of newKK
        // go ahead and setup pointer to augV
        //
        Teuchos::RCP<MV> defV, augV;
        if (inSituRestart_ == true) {
          //
          // get non-const pointer to solver's basis so we can work in situ
          Teuchos::RCP<MV> solverbasis = Teuchos::rcp_const_cast<MV>(state.V);
          // 
          // perform Householder QR of Su = Q [D;0], where D is unit diag.
          // work on a copy of Su, since we need Su below to build newKK
          Teuchos::SerialDenseMatrix<int,ScalarType> copySu(Su);
          std::vector<ScalarType> tau(numUnlocked), work(numUnlocked);
          int info;
          lapack.GEQRF(curdim,numUnlocked,copySu.values(),copySu.stride(),&tau[0],&work[0],work.size(),&info);
          TEST_FOR_EXCEPTION(info != 0,std::logic_error,
                             "Anasazi::BlockDavidsonSolMgr::solve(): error calling GEQRF during restarting.");
          if (printer->isVerbosity(Debug)) {
            Teuchos::SerialDenseMatrix<int,ScalarType> R(Teuchos::Copy,copySu,numUnlocked,numUnlocked);
            for (int j=0; j<numUnlocked; j++) {
              R(j,j) = SCT::magnitude(R(j,j)) - 1.0;
              for (int i=j+1; i<numUnlocked; i++) {
                R(i,j) = ZERO;
              }
            }
            printer->stream(Debug) << "||Triangular factor of Su - I||: " << R.normFrobenius() << std::endl;
          }
          // 
          // perform implicit oldV*Su
          // this actually performs oldV*[Su Sl*M] = [defV lockV], for some unitary M
          // we are actually interested in only the first numUnlocked vectors of the result
          {
            Teuchos::RCP<MV> oldV = MVT::CloneView(*solverbasis,curind);
            msutils::applyHouse(numUnlocked,*oldV,copySu,tau,workMV);
          }
          std::vector<int> defind(numUnlocked), augind(numNewLocked);
          for (int i=0; i<numUnlocked ; i++) defind[i] = i;
          for (int i=0; i<numNewLocked; i++) augind[i] = numUnlocked+i;
          defV = MVT::CloneView(*solverbasis,defind);
          augV = MVT::CloneView(*solverbasis,augind);
        }
        else { // inSituRestart == false)
          // defV = oldV*Su, explicitly. workspace is in workMV
          std::vector<int> defind(numUnlocked), augind(numNewLocked);
          for (int i=0; i<numUnlocked ; i++) defind[i] = i;
          for (int i=0; i<numNewLocked; i++) augind[i] = numUnlocked+i;
          Teuchos::RCP<const MV> oldV = MVT::CloneView(*state.V,curind);
          defV = MVT::CloneView(*workMV,defind);
          augV = MVT::CloneView(*workMV,augind);
          
          MVT::MvTimesMatAddMv(ONE,*oldV,Su,ZERO,*defV);
        }

        //
        // lockvecs will be partitioned as follows:
        // lockvecs = [curlocked augTmp ...]
        // - augTmp will be used for the storage of M*augV and K*augV
        //   later, the locked vectors (stored in state.X and referenced via const MV view newLocked) 
        //   will be moved into lockvecs on top of augTmp when it is no longer needed as workspace.
        // - curlocked will be used in orthogonalization of augV
        //
        // newL is the new locked vectors; newL = oldV*Sl = RitzVectors(lockind)
        // we will not produce them, but instead retrieve them from RitzVectors
        //
        Teuchos::RCP<const MV> curlocked, newLocked;
        Teuchos::RCP<MV> augTmp;
        {
          // setup curlocked
          if (curNumLocked > 0) {
            std::vector<int> curlockind(curNumLocked);
            for (int i=0; i<curNumLocked; i++) curlockind[i] = i;
            curlocked = MVT::CloneView(*lockvecs,curlockind);
          }
          else {
            curlocked = Teuchos::null;
          }
          // setup augTmp
          std::vector<int> augtmpind(numNewLocked); 
          for (int i=0; i<numNewLocked; i++) augtmpind[i] = curNumLocked+i;
          augTmp = MVT::CloneView(*lockvecs,augtmpind);
          // setup newLocked
          newLocked = MVT::CloneView(*bd_solver->getRitzVectors(),lockind);
        }

        // 
        // generate augV and perform orthogonalization
        //
        MVT::MvRandom(*augV);
        // 
        // orthogonalize it against auxvecs, defV, and all locked vectors (new and current)
        // use augTmp as storage for M*augV, if hasM
        {
          Teuchos::Array<Teuchos::RCP<const MV> > against;
          Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int,ScalarType> > > dummy;
          if (probauxvecs != Teuchos::null) against.push_back(probauxvecs);
          if (curlocked != Teuchos::null)   against.push_back(curlocked);
          against.push_back(newLocked);
          against.push_back(defV);
          if (problem_->getM() != Teuchos::null) {
            OPT::Apply(*problem_->getM(),*augV,*augTmp);
          }
          ortho->projectAndNormalizeMat(*augV,augTmp,dummy,Teuchos::null,against);
        }

        //
        // form newKK
        //
        // newKK = newV'*K*newV = [Su'*KK*Su    defV'*K*augV]
        //                        [augV'*K*defV augV'*K*augV]
        //
        // first, generate the principal submatrix, the projection of K onto the unlocked portion of oldV
        //
        Teuchos::SerialDenseMatrix<int,ScalarType> newKK(curdim,curdim);
        {
          Teuchos::SerialDenseMatrix<int,ScalarType> KKtmp(curdim,numUnlocked), 
                                                     KKold(Teuchos::View,*state.KK,curdim,curdim),
                                                     KK11(Teuchos::View,newKK,numUnlocked,numUnlocked);
          int teuchosRet;
          // KKtmp = KKold*Su
          teuchosRet = KKtmp.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,ONE,KKold,Su,ZERO);
          TEST_FOR_EXCEPTION(teuchosRet != 0,std::logic_error,
                             "Anasazi::BlockDavidsonSolMgr::solve(): Logic error calling SerialDenseMatrix::multiply.");
          // KK11 = Su'*KKtmp = Su'*KKold*Su
          teuchosRet = KK11.multiply(Teuchos::CONJ_TRANS,Teuchos::NO_TRANS,ONE,Su,KKtmp,ZERO);
          TEST_FOR_EXCEPTION(teuchosRet != 0,std::logic_error,
                             "Anasazi::BlockDavidsonSolMgr::solve(): Logic error calling SerialDenseMatrix::multiply.");
        }
        //
        // project the stiffness matrix on augV
        {
          OPT::Apply(*problem_->getOperator(),*augV,*augTmp);
          Teuchos::SerialDenseMatrix<int,ScalarType> KK12(Teuchos::View,newKK,numUnlocked,numNewLocked,0,numUnlocked),
                                                     KK22(Teuchos::View,newKK,numNewLocked,numNewLocked,numUnlocked,numUnlocked);
          MVT::MvTransMv(ONE,*defV,*augTmp,KK12);
          MVT::MvTransMv(ONE,*augV,*augTmp,KK22);
        }
        // 
        // done with defV,augV
        defV = Teuchos::null;
        augV = Teuchos::null;
        //
        // make it hermitian in memory (fill in KK21)
        for (int j=0; j<curdim; ++j) {
          for (int i=j+1; i<curdim; ++i) {
            newKK(i,j) = SCT::conjugate(newKK(j,i));
          }
        }
        //
        // we are done using augTmp as storage
        // put newLocked into lockvecs, new values into lockvals
        augTmp = Teuchos::null;
        {
          std::vector<Value<ScalarType> > allvals = bd_solver->getRitzValues();
          for (int i=0; i<numNewLocked; i++) {
            lockvals.push_back(allvals[lockind[i]].realpart);
          }

          std::vector<int> indlock(numNewLocked);
          for (int i=0; i<numNewLocked; i++) indlock[i] = curNumLocked+i;
          MVT::SetBlock(*newLocked,indlock,*lockvecs);
          newLocked = Teuchos::null;

          curNumLocked += numNewLocked;
          std::vector<int> curlockind(curNumLocked);
          for (int i=0; i<curNumLocked; i++) curlockind[i] = i;
          curlocked = MVT::CloneView(*lockvecs,curlockind);
        }
        // add locked vecs as aux vecs, along with aux vecs from problem
        // add lockvals to convtest
        // disable locktest if curNumLocked == maxLocked
        {
          convtest->setAuxVals(lockvals);

          Teuchos::Array< Teuchos::RCP<const MV> > aux;
          if (probauxvecs != Teuchos::null) aux.push_back(probauxvecs);
          aux.push_back(curlocked);
          bd_solver->setAuxVecs(aux);

          if (curNumLocked == maxLocked_) {
            // disabled locking now by setting quorum to unreachable number
            locktest->setQuorum(blockSize_+1);
          }
        }

        //
        // prepare new state
        BlockDavidsonState<ScalarType,MV> rstate;
        rstate.curDim = curdim;
        if (inSituRestart_) {
          // data is already in the solver's memory
          rstate.V = state.V;
        }
        else {
          // data is in workspace and will be copied to solver memory
          rstate.V = workMV;
        }
        rstate.KK = Teuchos::rcp( &newKK, false );
        //
        // pass new state to the solver
        bd_solver->initialize(rstate);
      } // end of locking
      ////////////////////////////////////////////////////////////////////////////////////
      //
      // we returned from iterate(), but none of our status tests Passed.
      // something is wrong, and it is probably our fault.
      //
      ////////////////////////////////////////////////////////////////////////////////////
      else {
        TEST_FOR_EXCEPTION(true,std::logic_error,"Anasazi::BlockDavidsonSolMgr::solve(): Invalid return from bd_solver::iterate().");
      }
    }
    catch (std::exception e) {
      printer->stream(Errors) << "Error! Caught exception in BlockDavidson::iterate() at iteration " << bd_solver->getNumIters() << std::endl 
                              << e.what() << std::endl;
      throw;
    }
  }

  // clear temp space
  workMV = Teuchos::null;

  sol.numVecs = convtest->howMany();
  if (sol.numVecs > 0) {
    sol.Evecs = MVT::Clone(*problem_->getInitVec(),sol.numVecs);
    sol.Espace = sol.Evecs;
    sol.Evals.resize(sol.numVecs);
    std::vector<MagnitudeType> vals(sol.numVecs);

    // copy them into the solution
    std::vector<int> which = convtest->whichVecs();
    // indices between [0,blockSize) refer to vectors/values in the solver
    // indices between [blockSize,blocksize+curNumLocked) refer to locked vectors/values
    // everything has already been ordered by the solver; we just have to partition the two references
    std::vector<int> inlocked(0), insolver(0);
    for (unsigned int i=0; i<which.size(); i++) {
      if (which[i] < blockSize_) {
        insolver.push_back(which[i]);
      }
      else {
        // sanity check
        TEST_FOR_EXCEPTION(which[i] >= curNumLocked+blockSize_,std::logic_error,"Anasazi::BlockDavidsonSolMgr::solve(): indexing mistake.");
        inlocked.push_back(which[i] - blockSize_);
      }
    }

    TEST_FOR_EXCEPTION(insolver.size() + inlocked.size() != (unsigned int)sol.numVecs,std::logic_error,"Anasazi::BlockDavidsonSolMgr::solve(): indexing mistake.");

    // set the vecs,vals in the solution
    if (insolver.size() > 0) {
      // set vecs
      int lclnum = insolver.size();
      std::vector<int> tosol(lclnum);
      for (int i=0; i<lclnum; i++) tosol[i] = i;
      Teuchos::RCP<const MV> v = MVT::CloneView(*bd_solver->getRitzVectors(),insolver);
      MVT::SetBlock(*v,tosol,*sol.Evecs);
      // set vals
      std::vector<Value<ScalarType> > fromsolver = bd_solver->getRitzValues();
      for (unsigned int i=0; i<insolver.size(); i++) {
        vals[i] = fromsolver[insolver[i]].realpart;
      }
    }

    // get the vecs,vals from locked storage
    if (inlocked.size() > 0) {
      int solnum = insolver.size();
      // set vecs
      int lclnum = inlocked.size();
      std::vector<int> tosol(lclnum);
      for (int i=0; i<lclnum; i++) tosol[i] = solnum + i;
      Teuchos::RCP<const MV> v = MVT::CloneView(*lockvecs,inlocked);
      MVT::SetBlock(*v,tosol,*sol.Evecs);
      // set vals
      for (unsigned int i=0; i<inlocked.size(); i++) {
        vals[i+solnum] = lockvals[inlocked[i]];
      }
    }

    // sort the eigenvalues and permute the eigenvectors appropriately
    {
      std::vector<int> order(sol.numVecs);
      sorter->sort(bd_solver.get(), sol.numVecs, vals, &order );
      // store the values in the Eigensolution
      for (int i=0; i<sol.numVecs; i++) {
        sol.Evals[i].realpart = vals[i];
        sol.Evals[i].imagpart = MT::zero();
      }
      // now permute the eigenvectors according to order
      msutils::permuteVectors(sol.numVecs,order,*sol.Evecs);
    }

    // setup sol.index, remembering that all eigenvalues are real so that index = {0,...,0}
    sol.index.resize(sol.numVecs,0);
  }

  // print final summary
  bd_solver->currentStatus(printer->stream(FinalSummary));

  // print timing information
  Teuchos::TimeMonitor::summarize(printer->stream(TimingDetails));

  problem_->setSolution(sol);
  printer->stream(Debug) << "Returning " << sol.numVecs << " eigenpairs to eigenproblem." << std::endl;

  if (sol.numVecs < nev) {
    return Unconverged; // return from BlockDavidsonSolMgr::solve() 
  }
  return Converged; // return from BlockDavidsonSolMgr::solve() 
}


} // end Anasazi namespace

#endif /* ANASAZI_BLOCKDAVIDSON_SOLMGR_HPP */
