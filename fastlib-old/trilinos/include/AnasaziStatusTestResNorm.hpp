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

#ifndef ANASAZI_STATUS_TEST_RESNORM_HPP
#define ANASAZI_STATUS_TEST_RESNORM_HPP

/*!
  \file AnasaziStatusTestResNorm.hpp
  \brief A status test for testing the norm of the eigenvectors residuals.
*/


#include "AnasaziStatusTest.hpp"
#include "Teuchos_ScalarTraits.hpp"

  /*! 
    \class Anasazi::StatusTestResNorm
    \brief A status test for testing the norm of the eigenvectors residuals.
    
    Anasazi::StatusTestResNorm was designed to be used as a test for
    convergence. The tester compares the norms of the residual vectors against
    a user specified tolerance. 
    
    In addition to specifying the tolerance, the user may specify:
    <ul>
      <li> the norm to be used: 2-norm or OrthoManager::norm() or getRitzRes2Norms()
      <li> the scale: absolute or relative to magnitude of Ritz value 
      <li> the quorum: the number of vectors required for the test to 
           evaluate as ::Passed.
    </ul>
  */

namespace Anasazi {


template <class ScalarType, class MV, class OP>
class StatusTestResNorm : public StatusTest<ScalarType,MV,OP> {

  typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;

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
  StatusTestResNorm(typename Teuchos::ScalarTraits<ScalarType>::magnitudeType tol, int quorum = -1, ResType whichNorm = RES_ORTH, bool scaled = true);

  //! Destructor
  virtual ~StatusTestResNorm() {};
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

  /*! \brief Set quorum.
   *
   *  Setting quorum to -1 signifies that all residuals from the solver must meet the tolerance.
   *  This also resets the test status to ::Undefined.
   */
  void setQuorum(int quorum) {
    state_ = Undefined;
    quorum_ = quorum;
  }

  /*! \brief Get quorum.
   */
  int getQuorum() {
    return quorum_;
  }

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
};


template <class ScalarType, class MV, class OP>
StatusTestResNorm<ScalarType,MV,OP>::StatusTestResNorm(typename Teuchos::ScalarTraits<ScalarType>::magnitudeType tol, int quorum, ResType whichNorm, bool scaled)
  : state_(Undefined), tol_(tol), quorum_(quorum), scaled_(scaled), whichNorm_(whichNorm) {}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestResNorm<ScalarType,MV,OP>::checkStatus( Eigensolver<ScalarType,MV,OP>* solver ) {
  typedef Teuchos::ScalarTraits<MagnitudeType> MT;
  
  std::vector<MagnitudeType> res;

  // get the eigenvector/ritz residuals norms (using the appropriate norm)
  // get the eigenvalues/ritzvalues and ritz index as well
  std::vector<Value<ScalarType> > vals = solver->getRitzValues();
  switch (whichNorm_) {
    case RES_2NORM:
      res = solver->getRes2Norms();
      // we want only the ritz values corresponding to our eigenvector residuals
      vals.resize(res.size());
      break;
    case RES_ORTH:
      res = solver->getResNorms();
      // we want only the ritz values corresponding to our eigenvector residuals
      vals.resize(res.size());
      break;
    case RITZRES_2NORM:
      res = solver->getRitzRes2Norms();
      break;
  }

  // if appropriate, scale the norms by the magnitude of the eigenvalue estimate
  if (scaled_) {
    Teuchos::LAPACK<int,MagnitudeType> lapack;

    for (unsigned int i=0; i<res.size(); i++) {
      MagnitudeType tmp = lapack.LAPY2(vals[i].realpart,vals[i].imagpart);
      // scale by the newly computed magnitude of the ritz values
      if ( tmp != MT::zero() ) {
        res[i] /= tmp;
      }
    }
  }

  // test the norms
  int have = 0;
  ind_.resize(res.size());
  for (unsigned int i=0; i<res.size(); i++) {
    TEST_FOR_EXCEPTION( MT::isnaninf(res[i]), StatusTestError, "StatusTestResNorm::checkStatus(): residual norm is nan or inf" );
    if (res[i] < tol_) {
      ind_[have] = i;
      have++;
    }
  }
  ind_.resize(have);
  int need = (quorum_ == -1) ? res.size() : quorum_;
  state_ = (have >= need) ? Passed : Failed;
  return state_;
}


template <class ScalarType, class MV, class OP>
std::ostream& StatusTestResNorm<ScalarType,MV,OP>::print(std::ostream& os, int indent) const {
  std::string ind(indent,' ');
  os << ind << "- StatusTestResNorm: ";
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

#endif /* ANASAZI_STATUS_TEST_RESNORM_HPP */
