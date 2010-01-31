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

#ifndef ANASAZI_STATUS_TEST_COMBO_HPP
#define ANASAZI_STATUS_TEST_COMBO_HPP

/*!
  \file AnasaziStatusTestCombo.hpp
  \brief Status test for forming logical combinations of other status tests.
*/


#include "AnasaziTypes.hpp"
#include "AnasaziStatusTest.hpp"
#include "Teuchos_Array.hpp"

  /*! 
    \class Anasazi::StatusTestCombo
    \brief Status test for forming logical combinations of other status tests.
    
    Test types include OR, AND, SEQOR and SEQAND.  The OR and AND tests
    evaluate all of the tests, in the order they were passed to the
    StatusTestCombo.  The SEQOR and SEQAND run only the tests necessary to
    determine the final outcome, short-circuiting on the first test that
    conclusively decides the outcome. More formally, SEQAND runs the tests in
    the order they were given to the StatusTestCombo class and stops after the
    first test that evaluates ::Failed. SEQOR run the tests in the order they
    were given to the StatusTestCombo class and stops after the first test that
    evaluates ::Passed.
  */

namespace Anasazi {


template <class ScalarType, class MV, class OP>
class StatusTestCombo : public StatusTest<ScalarType,MV,OP> {

 private:
   typedef Teuchos::Array< Teuchos::RCP< StatusTest<ScalarType,MV,OP> > > STPArray;

 public:

 //!  \brief Enumerated type to list the types of StatusTestCombo combo types.
 enum ComboType
   {
     OR,           /*!< Logical OR which evaluates all tests */
     AND,          /*!< Logical AND which evaluates all tests */
     SEQOR,        /*!< Short-circuited logical OR */
     SEQAND        /*!< Short-circuited logical AND */
   };


#ifndef DOXYGEN_SHOULD_SKIP_THIS

  typedef Teuchos::Array< Teuchos::RCP< StatusTest<ScalarType,MV,OP> > > t_arr;
  typedef std::vector< Teuchos::RCP< StatusTest<ScalarType,MV,OP> > > st_vector;
  typedef typename st_vector::iterator                 iterator;
  typedef typename st_vector::const_iterator     const_iterator;

#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @name Constructors/destructors
  //@{ 

  //! Constructor
  //! \brief Default constructor has no tests and initializes to ComboType OR.
  StatusTestCombo() : state_(Undefined) {}

  //! Constructor
  //! \brief Constructor specifying the ComboType and the tests.
  StatusTestCombo(ComboType type, Teuchos::Array< Teuchos::RCP< StatusTest<ScalarType,MV,OP> > > tests) :
    state_(Undefined), 
    type_(type)
  {
    setTests(tests);
  };

  //! Destructor
  virtual ~StatusTestCombo() {};
  //@}

  //! @name Status methods
  //@{ 
  /*! Check status as defined by test.
    
    \return TestStatus indicating whether the test passed or failed.
  */
  TestStatus checkStatus( Eigensolver<ScalarType,MV,OP>* solver );

  //! Return the result of the most recent checkStatus call.
  TestStatus getStatus() const {
    return state_;
  }
  //@}

  //! @name Accessor methods
  //@{ 

  /*! \brief Set the maximum number of iterations.
   *  This also resets the test status to ::Undefined.
   */
  void setComboType(ComboType type) {
    type_ = type;
    state_ = Undefined;
  }

  //! Get the maximum number of iterations.
  ComboType getComboType() const {return type_;}

  /*! \brief Set the tests
   *  This also resets the test status to ::Undefined.
   */
  void setTests(Teuchos::Array<Teuchos::RCP<StatusTest<ScalarType,MV,OP> > > tests) {
    tests_ = tests;
    state_ = Undefined;
  }

  //! Get the tests
  Teuchos::Array<Teuchos::RCP<StatusTest<ScalarType,MV,OP> > > getTests() const {return tests_;}

  /*! \brief Add a test to the combination.
   *
   *  This also resets the test status to ::Undefined.
   */
  void addTest(Teuchos::RCP<StatusTest<ScalarType,MV,OP> > test) {
    tests_.push_back(test);
    state_ = Undefined;
  }

  /*! \brief Removes a test from the combination, if it exists in the tester.
   *
   * This also resets the test status to ::Undefined, if a test was removed.
   */
  void removeTest(const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &test);

  //@}

  //! @name Reset methods
  //@{ 
  //! \brief Informs the status test that it should reset its internal configuration to the uninitialized state.
  /*! The StatusTestCombo class has no internal state, but children classes might, so this method will call
     reset() on all child status tests. It also resets the test status to ::Undefined.
  */
  void reset();

  //! \brief Clears the results of the last status test.
  /*! This should be distinguished from the reset() method, as it only clears the cached result from the last 
   * status test, so that a call to getStatus() will return ::Undefined. This is necessary for the SEQOR and SEQAND
   * tests in the StatusTestCombo class, which may short circuit and not evaluate all of the StatusTests contained
   * in them.
  */
  void clearStatus();

  //@}

  //! @name Print methods
  //@{ 
  
  //! Output formatted description of stopping test to output stream.
  std::ostream& print(std::ostream& os, int indent = 0) const;
 
  //@}
  private:

  TestStatus evalOR(Eigensolver<ScalarType,MV,OP>* solver);
  TestStatus evalAND(Eigensolver<ScalarType,MV,OP>* solver);
  TestStatus evalSEQOR(Eigensolver<ScalarType,MV,OP>* solver);
  TestStatus evalSEQAND(Eigensolver<ScalarType,MV,OP>* solver);

  TestStatus state_;
  ComboType type_;
  STPArray tests_;

};


template <class ScalarType, class MV, class OP>
void StatusTestCombo<ScalarType,MV,OP>::removeTest(const Teuchos::RCP<StatusTest<ScalarType,MV,OP> > &test) 
{
  typename STPArray::iterator iter1;
  iter1 = find(tests_.begin(),tests_.end(),test);
  if (iter1 != tests_.end()) {
    tests_.erase(iter1);
    state_ = Undefined;
  }
}


template <class ScalarType, class MV, class OP>
TestStatus StatusTestCombo<ScalarType,MV,OP>::checkStatus( Eigensolver<ScalarType,MV,OP>* solver ) {
  clearStatus();
  switch (type_) {
    case OR:
      state_ = evalOR(solver);
      break;
    case AND:
      state_ = evalAND(solver);
      break;
    case SEQOR:
      state_ = evalSEQOR(solver);
      break;
    case SEQAND:
      state_ = evalSEQAND(solver);
      break;
  }
  return state_;
}


template <class ScalarType, class MV, class OP>
void StatusTestCombo<ScalarType,MV,OP>::reset() {
  state_ = Undefined;
  for (iterator i=tests_.begin(); i != tests_.end(); i++) {
    (*i)->reset();
  }
}

template <class ScalarType, class MV, class OP>
void StatusTestCombo<ScalarType,MV,OP>::clearStatus() {
  state_ = Undefined;
  for (iterator i=tests_.begin(); i != tests_.end(); i++) {
    (*i)->clearStatus();
  }
}

template <class ScalarType, class MV, class OP>
std::ostream& StatusTestCombo<ScalarType,MV,OP>::print(std::ostream& os, int indent) const {
  std::string ind(indent,' ');
  os << ind << "- StatusTestCombo: ";
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
  // print children, with extra indention
  for (const_iterator i=tests_.begin(); i != tests_.end(); i++) {
    (*i)->print(os,indent+2);
  }
  return os;
}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestCombo<ScalarType,MV,OP>::evalOR( Eigensolver<ScalarType,MV,OP>* solver ) {
  state_ = Failed;
  for (iterator i=tests_.begin(); i != tests_.end(); i++) {
    TestStatus r = (*i)->checkStatus(solver);
    if (r == Passed) {
      state_ = Passed;
    }
    else {
      TEST_FOR_EXCEPTION(r != Failed,StatusTestError,
                         "Anasazi::StatusTestCombo::evalOR(): child test gave invalid return");
    }
  }
  return state_;
}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestCombo<ScalarType,MV,OP>::evalSEQOR( Eigensolver<ScalarType,MV,OP>* solver ) {
  state_ = Failed;
  for (iterator i=tests_.begin(); i != tests_.end(); i++) {
    TestStatus r = (*i)->checkStatus(solver);
    if (r == Passed) {
      state_ = Passed;
      break;
    }
    else {
      TEST_FOR_EXCEPTION(r != Failed,StatusTestError,
                         "Anasazi::StatusTestCombo::evalSEQOR(): child test gave invalid return");
    }
  }
  return state_;
}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestCombo<ScalarType,MV,OP>::evalAND( Eigensolver<ScalarType,MV,OP>* solver ) {
  state_ = Passed;
  for (iterator i=tests_.begin(); i != tests_.end(); i++) {
    TestStatus r = (*i)->checkStatus(solver);
    if (r == Failed) {
      state_ = Failed;
    }
    else {
      TEST_FOR_EXCEPTION(r != Passed,StatusTestError,
                         "Anasazi::StatusTestCombo::evalAND(): child test gave invalid return");
    }
  }
  return state_;
}

template <class ScalarType, class MV, class OP>
TestStatus StatusTestCombo<ScalarType,MV,OP>::evalSEQAND( Eigensolver<ScalarType,MV,OP>* solver ) {
  state_ = Passed;
  for (iterator i=tests_.begin(); i != tests_.end(); i++) {
    TestStatus r = (*i)->checkStatus(solver);
    if (r == Failed) {
      state_ = Failed;
      break;
    }
    else {
      TEST_FOR_EXCEPTION(r != Passed,StatusTestError,
                         "Anasazi::StatusTestCombo::evalAND(): child test gave invalid return");
    }
  }
  return state_;
}



} // end of Anasazi namespace

#endif /* ANASAZI_STATUS_TEST_COMBO_HPP */
