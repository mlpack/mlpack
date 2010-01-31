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

#ifndef ANASAZI_STATUS_TEST_OUTPUT_HPP
#define ANASAZI_STATUS_TEST_OUTPUT_HPP

/*!
  \file AnasaziStatusTestOutput.hpp
  \brief Special StatusTest for printing status tests.
*/


#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziEigensolver.hpp"

#include "AnasaziStatusTest.hpp"



namespace Anasazi {

  /*! 
    \class StatusTestOutput
    \brief A special StatusTest for printing other status tests. 
    
    StatusTestOutput is a wrapper around another StatusTest that calls 
    StatusTest::print() on the underlying object on calls to StatusTestOutput::checkStatus().
    The frequency and occasion of the printing can be dictated according to some parameters passed to 
    StatusTestOutput::StatusTestOutput().
  */
template <class ScalarType, class MV, class OP>
class StatusTestOutput : public StatusTest<ScalarType,MV,OP> {

 public:
  //! @name Constructors/destructors
  //@{ 

  /*! \brief Constructor
   *
   * The StatusTestOutput requires an OutputManager for printing the underlying StatusTest on
   * calls to checkStatus(), as well as an underlying StatusTest.
   *
   * The last two parameters, described below, in addition to the verbosity level of the OutputManager, control when printing is 
   * called. When both the \c mod criterion and the \c printStates criterion are satisfied, the status test will be printed to the 
   * OutputManager with ::MsgType of ::StatusTestDetails.
   *
   * @param[in] mod A positive number describes how often the output should be printed. On every call to checkStatus(), an internal counter
   *                is incremented. Printing may only occur when this counter is congruent to zero modulo \c mod. Default: 1 (attempt to print on every call to checkStatus())
   * @param[in] printStates A combination of ::TestStatus values for which the output may be printed. Default: ::Passed (attempt to print whenever checkStatus() will return ::Passed)
   *
   */
  StatusTestOutput(const Teuchos::RCP<OutputManager<ScalarType> > &printer, 
                   Teuchos::RCP<StatusTest<ScalarType,MV,OP> > test,
                   int mod = 1,
                   int printStates = Passed)
    : printer_(printer), test_(test), state_(Undefined), stateTest_(printStates), modTest_(mod), numCalls_(0) {}

  //! Destructor
  virtual ~StatusTestOutput() {};
  //@}

  //! @name Status methods
  //@{ 
  /*! Check and return status of underlying StatusTest.

    This method calls checkStatus() on the StatusTest object passed in the constructor. If appropriate, the
    method will follow this call with a call to print() on the underlying object, using the OutputManager passed via the constructor
    with verbosity level ::StatusTestDetails.

    The internal counter will be incremented during this call, but only after
    performing the tests to decide whether or not to print the underlying
    StatusTest. This way, the very first call to checkStatus() following
    initialization or reset() will enable the underlying StatusTest to be
    printed, regardless of the mod parameter, as the current number of calls
    will be zero.

    If the specified Teuchos::RCP for the child class is Teuchos::null, then calling checkStatus() will result in a StatusTestError exception being thrown.
    
    \return ::TestStatus indicating whether the underlying test passed or failed.
  */
  TestStatus checkStatus( Eigensolver<ScalarType,MV,OP>* solver ) {
    TEST_FOR_EXCEPTION(test_ == Teuchos::null,StatusTestError,"StatusTestOutput::checkStatus(): child pointer is null.");
    state_ = test_->checkStatus(solver);

    if (numCalls_++ % modTest_ == 0) {
      if ( (state_ & stateTest_) == state_) {
        if ( printer_->isVerbosity(StatusTestDetails) ) {
          print( printer_->stream(StatusTestDetails) );
        }
        else if ( printer_->isVerbosity(Debug) ) {
          print( printer_->stream(Debug) );
        }
      }
    }

    return state_;
  }

  //! Return the result of the most recent checkStatus call, or undefined if it has not been run.
  TestStatus getStatus() const {
    return state_;
  }
  //@}


  //! @name Accessor methods
  //@{ 

  /*! \brief Set child test.
   *
   *  \note This also resets the test status to ::Undefined.
   */
  void setChild(Teuchos::RCP<StatusTest<ScalarType,MV,OP> > test) {
    test_ = test;
    state_ = Undefined;
  }

  //! \brief Get child test.
  Teuchos::RCP<StatusTest<ScalarType,MV,OP> > getChild() const {
    return test_;
  }

  //@}


  //! @name Reset methods
  //@{ 
  /*! \brief Informs the status test that it should reset its internal configuration to the uninitialized state.
   *
   *  This resets the cached state to an ::Undefined state and calls reset() on the underlying test. It also 
   *  resets the counter for the number of calls to checkStatus().
   */
  void reset() { 
    state_ = Undefined;
    test_->reset();
    numCalls_ = 0;
  }

  //! Clears the results of the last status test.
  //! This resets the cached state to an ::Undefined state and calls clearStatus() on the underlying test.
  void clearStatus() {
    state_ = Undefined;
    test_->clearStatus();
  }

  //@}

  //! @name Print methods
  //@{ 
  
  //! Output formatted description of stopping test to output stream.
  std::ostream& print(std::ostream& os, int indent = 0) const {
    std::string ind(indent,' ');
    os << ind << "- StatusTestOutput: ";
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
    os << ind << "  (Num calls,Mod test,State test): " << "(" << numCalls_ << ", " << modTest_ << ",";
    if (stateTest_ == 0) {
      os << " none )" << std::endl;
    }
    else {
      if ( (stateTest_ & Passed) == Passed ) os << " Passed";
      if ( (stateTest_ & Failed) == Failed ) os << " Failed";
      if ( (stateTest_ & Undefined) == Undefined ) os << " Undefined";
      os << " )" << std::endl;
    }
    // print child, with extra indention
    test_->print(os,indent+3);
    return os;
  }
 
  //@}

  private:
    Teuchos::RCP<OutputManager<ScalarType> > printer_;
    Teuchos::RCP<StatusTest<ScalarType,MV,OP> > test_;
    TestStatus state_;
    int stateTest_;
    int modTest_;
    int numCalls_;
};

} // end of Anasazi namespace

#endif /* ANASAZI_STATUS_TEST_OUTPUT_HPP */
