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
//

#ifndef ANASAZI_STATUS_TEST_ORDEREDRESNORM_HPP
#define ANASAZI_STATUS_TEST_ORDEREDRESNORM_HPP

/*!
  \file AnasaziStatusTestOrderedResNorm.hpp
  \brief A status test for testing the norm of the eigenvectors residuals along with a 
         set of auxiliary eigenvalues.
*/


#include "AnasaziStatusTest.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_LAPACK.hpp"

  /*! 
    \class Anasazi::StatusTestOrderedResNorm 
    
    \brief A status test for testing the norm of the eigenvectors residuals
    along with a set of auxiliary eigenvalues. 
    
    The test evaluates to ::Passed when then the most significant of the
    eigenvalues all have a residual below a certain threshhold. The purpose of
    the test is to not only test convergence for some number of eigenvalues,
    but to test convergence for the correct ones.
    
    In addition to specifying the tolerance, the user may specify:
    <ul>
      <li> the norm to be used: 2-norm or OrthoManager::norm() or getRitzRes2Norms()
      <li> the scale: absolute or relative to magnitude of Ritz value 
      <li> the quorum: the number of vectors required for the test to 
           evaluate as ::Passed.
    </ul>

    Finally, the user must specify the Anasazi::SortManager used for deciding
    significance. 
  */

namespace Anasazi {


template <class ScalarType, class MV, class OP>
class StatusTestOrderedResNorm : public StatusTest<ScalarType,MV,OP> {

 private:
  typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
  typedef Teuchos::ScalarTraits<MagnitudeType> MT;

 public:

  //! @name Enums
  //@{

  /*! \enum ResType 
      \brief Enumerated type used to specify which residual norm used by this status test.
   */
  enum ResType {
    RES_ORTH,
    RES_2NORM,
    RITZRES_2NORM
  };

  //@}

  //! @name Constructors/destructors
  //@{ 

  //! Constructor
  StatusTestOrderedResNorm(Teuchos::RCP<SortManager<ScalarType,MV,OP> > sorter, typename Teuchos::ScalarTraits<ScalarType>::magnitudeType tol, int quorum = -1, ResType whichNorm = RES_ORTH, bool scaled = true);

  //! Destructor
  virtual ~StatusTestOrderedResNorm() {};
  //@}

  //! @name Status methods
  //@{ 
  /*! Check status as defined by test.
    \return TestStatus indicating whether the test passed or failed.
  */
  TestStatus checkStatus( Eigensolver<ScalarType,MV,OP>* solver );

  //! Return the result of the most recent checkStatus call.
  TestStatus getStatus() const { return state_; }
  //@}

  //! @name Accessor methods
  //@{ 

  /*! \brief Set tolerance.
   *  This also resets the test status to ::Undefined.
   */
  void setTolerance(typename Teuchos::ScalarTraits<ScalarType>::magnitudeType tol) {
    state_ = Undefined;
    tol_ = tol;
  }

  //! Get tolerance.
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType getTolerance() {return tol_;}

  /*! \brief Set the residual norm to be used by the status test.
   *
   *  This also resets the test status to ::Undefined.
   */
  void setWhichNorm(ResType whichNorm) {
    state_ = Undefined;
    whichNorm_ = whichNorm;
  }

  //! Return the residual norm used by the status test.
  ResType getWhichNorm() {return whichNorm_;}

  /*! \brief Instruct test to scale norms by eigenvalue estimates (relative scale).
   *  This also resets the test status to ::Undefined.
   */
  void setScale(bool relscale) {
    state_ = Undefined;
    scaled_ = relscale;
  }

  //! Returns true if the test scales the norms by the eigenvalue estimates (relative scale).
  bool getScale() {return scaled_;}

  //! Get the indices for the vectors that passed the test.
  std::vector<int> whichVecs() {
    return ind_;
  }

  //! Get the number of vectors that passed the test.
  int howMany() {
    return ind_.size();
  }

  //@}

  //! @name Reset methods
  //@{ 
  //! Informs the status test that it should reset its internal configuration to the uninitialized state.
  /*! This is necessary for the case when the status test is being reused by another solver or for another
    eigenvalue problem. The status test may have information that pertains to a particular problem or solver 
    state. The internal information will be reset back to the uninitialized state. The user specified information 
    that the convergence test uses will remain.
  */
  void reset() { 
    state_ = Undefined;
  }

  //! Clears the results of the last status test.
  /*! This should be distinguished from the reset() method, as it only clears the cached result from the last 
   * status test, so that a call to getStatus() will return ::Undefined. This is necessary for the SEQOR and SEQAND
   * tests in the StatusTestCombo class, which may short circuit and not evaluate all of the StatusTests contained
   * in them.
  */
  void clearStatus() {
    state_ = Undefined;
  }

  /*! \brief Set the auxiliary eigenvalues.
   *
   *  This routine sets only the real part of the auxiliary eigenvalues; the imaginary part is set to zero. This routine also resets the state to ::Undefined.
   */
  void setAuxVals(const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &vals) {
    rvals_ = vals;
    ivals_.resize(rvals_.size(),MT::zero());
    state_ = Undefined;
  }

  /*! \brief Set the auxiliary eigenvalues.
   *
   *  This routine sets both the real and imaginary parts of the auxiliary eigenvalues. This routine also resets the state to ::Undefined.
   */
  void setAuxVals(const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &rvals, const std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> &ivals) {
    rvals_ = rvals;
    ivals_ = ivals;
    state_ = Undefined;
  }

  //@}

  //! @name Print methods
  //@{ 
  
  //! Output formatted description of stopping test to output stream.
  std::ostream& print(std::ostream& os, int indent = 0) const;
 
  //@}
  private:
    TestStatus state_;
    MagnitudeType tol_;
    std::vector<int> ind_;
    int quorum_;
    bool scaled_;
    ResType whichNorm_;
    std::vector<MagnitudeType> rvals_, ivals_;
    Teuchos::RCP<SortManager<ScalarType,MV,OP> > sorter_;
};


template <class ScalarType, class MV, class OP>
StatusTestOrderedResNorm<ScalarType,MV,OP>::StatusTestOrderedResNorm(Teuchos::RCP<SortManager<ScalarType,MV,OP> > sorter, typename Teuchos::ScalarTraits<ScalarType>::magnitudeType tol, int quorum, ResType whichNorm, bool scaled)
  : state_(Undefined), quorum_(quorum), scaled_(scaled), whichNorm_(whichNorm), sorter_(sorter) 
{
  TEST_FOR_EXCEPTION(sorter_ == Teuchos::null, StatusTestError, "StatusTestOrderedResNorm::constructor() was passed null pointer for SortManager.");
  setTolerance(tol); 
}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestOrderedResNorm<ScalarType,MV,OP>::checkStatus( Eigensolver<ScalarType,MV,OP>* solver ) {


  // get the eigenvector/ritz residuals norms (using the appropriate norm)
  // get the eigenvalues/ritzvalues as well
  std::vector<MagnitudeType> res; 
  std::vector<Value<ScalarType> > vals = solver->getRitzValues();
  switch (whichNorm_) {
    case RES_2NORM:
      res = solver->getRes2Norms();
      vals.resize(res.size());
      break;
    case RES_ORTH:
      res = solver->getResNorms();
      vals.resize(res.size());
      break;
    case RITZRES_2NORM:
      res = solver->getRitzRes2Norms();
      break;
  }

  int numaux = rvals_.size();
  int bs = res.size();
  int num = bs + numaux;

  if (num == 0) {
    ind_.resize(0);
    return Failed;
  }

  // extract the real and imaginary parts from the 
  std::vector<MagnitudeType> allrvals(bs), allivals(bs);
  for (int i=0; i<bs; i++) {
    allrvals[i] = vals[i].realpart;
    allivals[i] = vals[i].imagpart;
  }

  // put the auxiliary values in the vectors as well
  allrvals.insert(allrvals.end(),rvals_.begin(),rvals_.end());
  allivals.insert(allivals.end(),ivals_.begin(),ivals_.end());

  // if appropriate, scale the norms by the magnitude of the eigenvalue estimate
  Teuchos::LAPACK<int,MagnitudeType> lapack;
  if (scaled_) {
    for (unsigned int i=0; i<res.size(); i++) {
      MagnitudeType tmp = lapack.LAPY2(allrvals[i],allivals[i]);
      if ( tmp != MT::zero() ) {
        res[i] /= tmp;
      }
    }
  }
  // add -1 residuals for the auxiliary values (because -1 < tol_)
  res.insert(res.end(),numaux,-MT::one());

  // we don't actually need the sorted eigenvalues; just the permutation vector
  std::vector<int> perm(num,-1);
  sorter_->sort(solver,num,allrvals,allivals,&perm);

  // apply the sorting to the residuals and original indices
  std::vector<MagnitudeType> oldres = res;
  for (int i=0; i<num; i++) {
    res[i] = oldres[perm[i]];
  }

  // indices: [0,bs) are from solver, [bs,bs+numaux) are from auxiliary values
  ind_.resize(num);

  // test the norms: we want res [0,quorum) to be <= tol
  int have = 0;
  int need = (quorum_ == -1) ? num : quorum_;
  int tocheck = need > num ? num : need;
  for (int i=0; i<tocheck; i++) {
    TEST_FOR_EXCEPTION( MT::isnaninf(res[i]), StatusTestError, "StatusTestOrderedResNorm::checkStatus(): residual norm is nan or inf" );
    if (res[i] < tol_) {
      ind_[have] = perm[i];
      have++;
    }
  }
  ind_.resize(have);
  state_ = (have >= need) ? Passed : Failed;
  return state_;
}


template <class ScalarType, class MV, class OP>
std::ostream& StatusTestOrderedResNorm<ScalarType,MV,OP>::print(std::ostream& os, int indent) const {
  std::string ind(indent,' ');
  os << ind << "- StatusTestOrderedResNorm: ";
  switch (state_) {
  case Passed:
    os << "Passed" << std::endl;
    break;
  case Failed:
    os << "Failed" << std::endl;
    break;
  case Undefined:
    os << "Undefined" << std::endl;
    break;
  }
  os << ind << "  (Tolerance,WhichNorm,Scaled,Quorum): " 
            << "(" << tol_;
  switch (whichNorm_) {
  case RES_ORTH:
    os << ",RES_ORTH";
    break;
  case RES_2NORM:
    os << ",RES_2NORM";
    break;
  case RITZRES_2NORM:
    os << ",RITZRES_2NORM";
    break;
  }
  os        << "," << (scaled_   ? "true" : "false")
            << "," << quorum_ 
            << ")" << std::endl;
  os << ind << "  Auxiliary values: ";
  if (rvals_.size() > 0) {
    for (unsigned int i=0; i<rvals_.size(); i++) {
      os << "(" << rvals_[i] << ", " << ivals_[i] << ")  ";
    }
    os << std::endl;
  }
  else {
    os << "[empty]" << std::endl;
  }

  if (state_ != Undefined) {
    os << ind << "  Which vectors: ";
    if (ind_.size() > 0) {
      for (unsigned int i=0; i<ind_.size(); i++) os << ind_[i] << " ";
      os << std::endl;
    }
    else {
      os << "[empty]" << std::endl;
    }
  }
  return os;
}


} // end of Anasazi namespace

#endif /* ANASAZI_STATUS_TEST_ORDEREDRESNORM_HPP */
