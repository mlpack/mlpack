
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

#ifndef ANASAZI_SIMPLE_LOBPCG_SOLMGR_HPP
#define ANASAZI_SIMPLE_LOBPCG_SOLMGR_HPP

/*! \file AnasaziSimpleLOBPCGSolMgr.hpp
  \brief The Anasazi::SimpleLOBPCGSolMgr provides a simple solver manager over the LOBPCG 
  eigensolver.
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"

#include "AnasaziEigenproblem.hpp"
#include "AnasaziSolverManager.hpp"

#include "AnasaziLOBPCG.hpp"
#include "AnasaziBasicSort.hpp"
#include "AnasaziSVQBOrthoManager.hpp"
#include "AnasaziStatusTestMaxIters.hpp"
#include "AnasaziStatusTestResNorm.hpp"
#include "AnasaziStatusTestCombo.hpp"
#include "AnasaziStatusTestOutput.hpp"
#include "AnasaziBasicOutputManager.hpp"
#include "AnasaziSolverUtils.hpp"

#include "Teuchos_TimeMonitor.hpp"

/** \example LOBPCG/LOBPCGEpetraExSimple.cpp
    This is an example of how to use the Anasazi::SimpleLOBPCGSolMgr solver manager.
*/

/*! \class Anasazi::SimpleLOBPCGSolMgr
  \brief The Anasazi::SimpleLOBPCGSolMgr provides a simple solver
  manager over the LOBPCG eigensolver. 

  Anasazi::SimpleLOBPCGSolMgr allows the user to specify convergence
  tolerance, verbosity level and block size. When block size is less than the
  number of requested eigenvalues specified in the eigenproblem, checkpointing
  is activated.

  The purpose of this solver manager was to provide an example of a simple
  solver manager, useful for demonstration as well as a jumping-off point for
  solvermanager development. Also, the solver manager is useful for testing
  some of the features of the Anasazi::LOBPCG eigensolver, principally the use
  of auxiliary vectors.

  This solver manager does not verify before quitting that the nev eigenvectors
  that have converged are also the smallest nev eigenvectors that are known.

  \ingroup anasazi_solver_framework

  \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, Heidi Thornquist
*/

namespace Anasazi {

template<class ScalarType, class MV, class OP>
class SimpleLOBPCGSolMgr : public SolverManager<ScalarType,MV,OP> {

  private:
    typedef MultiVecTraits<ScalarType,MV> MVT;
    typedef Teuchos::ScalarTraits<ScalarType> SCT;
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef Teuchos::ScalarTraits<MagnitudeType> MT;
    
  public:

  //!@name Constructors/Destructor 
  //@{ 

  /*! \brief Basic constructor for SimpleLOBPCGSolMgr.
   *
   * This constructor accepts the Eigenproblem to be solved in addition
   * to a parameter list of options for the solver manager. These options include the following:
   *   - "Which" - a \c string specifying the desired eigenvalues: SM, LM, SR or LR. Default: SR
   *   - "Block Size" - a \c int specifying the block size to be used by the underlying LOBPCG solver. Default: problem->getNEV()
   *   - "Maximum Iterations" - a \c int specifying the maximum number of iterations the underlying solver is allowed to perform. Default: 100
   *   - "Verbosity" - a sum of MsgType specifying the verbosity. Default: Anasazi::Errors
   *   - "Convergence Tolerance" - a \c MagnitudeType specifying the level that residual norms must reach to decide convergence. Default: machine precision
   */
  SimpleLOBPCGSolMgr( const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
                             Teuchos::ParameterList &pl );

  //! Destructor.
  virtual ~SimpleLOBPCGSolMgr() {};
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
   * \returns ::ReturnType specifying:
   *    - ::Converged: the eigenproblem was solved to the specification required by the solver manager.
   *    - ::Unconverged: the eigenproblem was not solved to the specification desired by the solver manager
  */
  ReturnType solve();
  //@}

  private:
  Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > problem_;
  std::string whch_; 
  MagnitudeType tol_;
  int verb_;
  int blockSize_;
  int maxIters_;
};


////////////////////////////////////////////////////////////////////////////////////////
template<class ScalarType, class MV, class OP>
SimpleLOBPCGSolMgr<ScalarType,MV,OP>::SimpleLOBPCGSolMgr( 
        const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
        Teuchos::ParameterList &pl ) : 
  problem_(problem),
  whch_("LM"),
  tol_(1e-6),
  verb_(Anasazi::Errors),
  blockSize_(0),
  maxIters_(100)
{
  TEST_FOR_EXCEPTION(problem_ == Teuchos::null,              std::invalid_argument, "Problem not given to solver manager.");
  TEST_FOR_EXCEPTION(!problem_->isProblemSet(),              std::invalid_argument, "Problem not set.");
  TEST_FOR_EXCEPTION(!problem_->isHermitian(),               std::invalid_argument, "Problem not symmetric.");
  TEST_FOR_EXCEPTION(problem_->getInitVec() == Teuchos::null,std::invalid_argument, "Problem does not contain initial vectors to clone from.");

  whch_ = pl.get("Which","SR");
  TEST_FOR_EXCEPTION(whch_ != "SM" && whch_ != "LM" && whch_ != "SR" && whch_ != "LR",
                     AnasaziError,
                     "SimpleLOBPCGSolMgr: \"Which\" parameter must be SM, LM, SR or LR.");

  tol_ = pl.get("Convergence Tolerance",tol_);
  TEST_FOR_EXCEPTION(tol_ <= 0,
                     AnasaziError,
                     "SimpleLOBPCGSolMgr: \"Tolerance\" parameter must be strictly postiive.");

  // verbosity level
  if (pl.isParameter("Verbosity")) {
    if (Teuchos::isParameterType<int>(pl,"Verbosity")) {
      verb_ = pl.get("Verbosity", verb_);
    } else {
      verb_ = (int)Teuchos::getParameter<Anasazi::MsgType>(pl,"Verbosity");
    }
  }


  blockSize_= pl.get("Block Size",problem_->getNEV());
  TEST_FOR_EXCEPTION(blockSize_ <= 0,
                     AnasaziError,
                     "SimpleLOBPCGSolMgr: \"Block Size\" parameter must be strictly positive.");

  maxIters_ = pl.get("Maximum Iterations",maxIters_);
}



////////////////////////////////////////////////////////////////////////////////////////
template<class ScalarType, class MV, class OP>
ReturnType 
SimpleLOBPCGSolMgr<ScalarType,MV,OP>::solve() {

  // sort manager
  Teuchos::RCP<BasicSort<ScalarType,MV,OP> > sorter = Teuchos::rcp( new BasicSort<ScalarType,MV,OP>(whch_) );
  // output manager
  Teuchos::RCP<BasicOutputManager<ScalarType> > printer = Teuchos::rcp( new BasicOutputManager<ScalarType>(verb_) );
  // status tests
  Teuchos::RCP<StatusTestMaxIters<ScalarType,MV,OP> > max;
  if (maxIters_ > 0) {
    max = Teuchos::rcp( new StatusTestMaxIters<ScalarType,MV,OP>(maxIters_) );
  }
  else {
    max = Teuchos::null;
  }
  Teuchos::RCP<StatusTestResNorm<ScalarType,MV,OP> > norm 
      = Teuchos::rcp( new StatusTestResNorm<ScalarType,MV,OP>(tol_) );
  Teuchos::Array< Teuchos::RCP<StatusTest<ScalarType,MV,OP> > > alltests;
  alltests.push_back(norm);
  if (max != Teuchos::null) alltests.push_back(max);
  Teuchos::RCP<StatusTestCombo<ScalarType,MV,OP> > combo 
      = Teuchos::rcp( new StatusTestCombo<ScalarType,MV,OP>(
              StatusTestCombo<ScalarType,MV,OP>::OR, alltests
        ));
  // printing StatusTest
  Teuchos::RCP<StatusTestOutput<ScalarType,MV,OP> > outputtest
    = Teuchos::rcp( new StatusTestOutput<ScalarType,MV,OP>( printer,combo,1,Passed ) );
  // orthomanager
  Teuchos::RCP<SVQBOrthoManager<ScalarType,MV,OP> > ortho 
    = Teuchos::rcp( new SVQBOrthoManager<ScalarType,MV,OP>(problem_->getM()) );
  // parameter list
  Teuchos::ParameterList plist;
  plist.set("Block Size",blockSize_);
  plist.set("Full Ortho",true);

  // create an LOBPCG solver
  Teuchos::RCP<LOBPCG<ScalarType,MV,OP> > lobpcg_solver 
    = Teuchos::rcp( new LOBPCG<ScalarType,MV,OP>(problem_,sorter,printer,outputtest,ortho,plist) );
  // add the auxillary vecs from the eigenproblem to the solver
  if (problem_->getAuxVecs() != Teuchos::null) {
    lobpcg_solver->setAuxVecs( Teuchos::tuple<Teuchos::RCP<const MV> >(problem_->getAuxVecs()) );
  }

  int numfound = 0;
  int nev = problem_->getNEV();
  Teuchos::Array< Teuchos::RCP<MV> > foundvecs;
  Teuchos::Array< Teuchos::RCP< std::vector<MagnitudeType> > > foundvals;
  while (numfound < nev) {
    // reduce the strain on norm test, if we are almost done
    if (nev - numfound < blockSize_) {
      norm->setQuorum(nev-numfound);
    }

    // tell the solver to iterate
    try {
      lobpcg_solver->iterate();
    }
    catch (std::exception e) {
      // we are a simple solver manager. we don't catch exceptions. set solution empty, then rethrow.
      printer->stream(Anasazi::Errors) << "Exception: " << e.what() << std::endl;
      Eigensolution<ScalarType,MV> sol;
      sol.numVecs = 0;
      problem_->setSolution(sol);
      throw;
    }

    // check the status tests
    if (norm->getStatus() == Passed) {

      int num = norm->howMany();
      // if num < blockSize_, it is because we are on the last iteration: num+numfound>=nev
      TEST_FOR_EXCEPTION(num < blockSize_ && num+numfound < nev,
                         std::logic_error,
                         "Anasazi::SimpleLOBPCGSolMgr::solve(): logic error.");
      std::vector<int> ind = norm->whichVecs();
      // just grab the ones that we need
      if (num + numfound > nev) {
        num = nev - numfound;
        ind.resize(num);
      }

      // copy the converged eigenvectors
      Teuchos::RCP<MV> newvecs = MVT::CloneCopy(*lobpcg_solver->getRitzVectors(),ind);
      // store them
      foundvecs.push_back(newvecs);
      // add them as auxiliary vectors
      Teuchos::Array<Teuchos::RCP<const MV> > auxvecs = lobpcg_solver->getAuxVecs();
      auxvecs.push_back(newvecs);
      // setAuxVecs() will reset the solver to uninitialized, without messing with numIters()
      lobpcg_solver->setAuxVecs(auxvecs);

      // copy the converged eigenvalues
      Teuchos::RCP<std::vector<MagnitudeType> > newvals = Teuchos::rcp( new std::vector<MagnitudeType>(num) );
      std::vector<Value<ScalarType> > all = lobpcg_solver->getRitzValues();
      for (int i=0; i<num; i++) {
        (*newvals)[i] = all[ind[i]].realpart;
      }
      foundvals.push_back(newvals);

      numfound += num;
    }
    else if (max != Teuchos::null && max->getStatus() == Passed) {

      int num = norm->howMany();
      std::vector<int> ind = norm->whichVecs();
      
      if (num > 0) {
        // copy the converged eigenvectors
        Teuchos::RCP<MV> newvecs = MVT::CloneCopy(*lobpcg_solver->getRitzVectors(),ind);
        // orthornormalize to be safe
        ortho->normalizeMat(*newvecs,Teuchos::null,Teuchos::null);
        // store them
        foundvecs.push_back(newvecs);
        // don't bother adding them as auxiliary vectors; we have reached maxiters and are going to quit
        
        // copy the converged eigenvalues
        Teuchos::RCP<std::vector<MagnitudeType> > newvals = Teuchos::rcp( new std::vector<MagnitudeType>(num) );
        std::vector<Value<ScalarType> > all = lobpcg_solver->getRitzValues();
        for (int i=0; i<num; i++) {
          (*newvals)[i] = all[ind[i]].realpart;
        }
        foundvals.push_back(newvals);
  
        numfound += num;
      }
      break;  // while(numfound < nev)
    }
    else {
      TEST_FOR_EXCEPTION(true,std::logic_error,"Anasazi::SimpleLOBPCGSolMgr::solve(): solver returned without satisfy status test.");
    }
  } // end of while(numfound < nev)

  TEST_FOR_EXCEPTION(foundvecs.size() != foundvals.size(),std::logic_error,"Anasazi::SimpleLOBPCGSolMgr::solve(): inconsistent array sizes");

  // create contiguous storage for all eigenvectors, eigenvalues
  Eigensolution<ScalarType,MV> sol;
  sol.numVecs = numfound;
  if (numfound > 0) {
    // allocate space for eigenvectors
    sol.Evecs = MVT::Clone(*problem_->getInitVec(),numfound);
  }
  else {
    sol.Evecs = Teuchos::null;
  }
  sol.Espace = sol.Evecs;
  // allocate space for eigenvalues
  std::vector<MagnitudeType> vals(numfound);
  sol.Evals.resize(numfound);
  // all real eigenvalues: set index vectors [0,...,numfound-1]
  sol.index.resize(numfound,0);
  // store eigenvectors, eigenvalues
  int curttl = 0;
  for (unsigned int i=0; i<foundvals.size(); i++) {
    TEST_FOR_EXCEPTION((signed int)(foundvals[i]->size()) != MVT::GetNumberVecs(*foundvecs[i]), std::logic_error, "Anasazi::SimpleLOBPCGSolMgr::solve(): inconsistent sizes");
    unsigned int lclnum = foundvals[i]->size();
    std::vector<int> lclind(lclnum);
    for (unsigned int j=0; j<lclnum; j++) lclind[j] = curttl+j;
    // put the eigenvectors
    MVT::SetBlock(*foundvecs[i],lclind,*sol.Evecs);
    // put the eigenvalues
    copy( foundvals[i]->begin(), foundvals[i]->end(), vals.begin()+curttl );

    curttl += lclnum;
  }
  TEST_FOR_EXCEPTION( curttl != sol.numVecs, std::logic_error, "Anasazi::SimpleLOBPCGSolMgr::solve(): inconsistent sizes");

  // sort the eigenvalues and permute the eigenvectors appropriately
  if (numfound > 0) {
    std::vector<int> order(sol.numVecs);
    sorter->sort( lobpcg_solver.get(), sol.numVecs, vals, &order );
    // store the values in the Eigensolution
    for (int i=0; i<sol.numVecs; i++) {
      sol.Evals[i].realpart = vals[i];
      sol.Evals[i].imagpart = MT::zero();
    }
    // now permute the eigenvectors according to order
    SolverUtils<ScalarType,MV,OP> msutils;
    msutils.permuteVectors(sol.numVecs,order,*sol.Evecs);
  }

  // print final summary
  lobpcg_solver->currentStatus(printer->stream(FinalSummary));

  // print timing information
  Teuchos::TimeMonitor::summarize(printer->stream(TimingDetails));

  // send the solution to the eigenproblem
  problem_->setSolution(sol);
  printer->stream(Debug) << "Returning " << sol.numVecs << " eigenpairs to eigenproblem." << std::endl;

  // return from SolMgr::solve()
  if (sol.numVecs < nev) return Unconverged;
  return Converged;
}




} // end Anasazi namespace

#endif /* ANASAZI_SIMPLE_LOBPCG_SOLMGR_HPP */
