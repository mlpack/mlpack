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

#ifndef ANASAZI_BASIC_OUTPUT_MANAGER_HPP
#define ANASAZI_BASIC_OUTPUT_MANAGER_HPP

/*!     \file AnasaziBasicOutputManager.hpp
        \brief Basic output manager for sending information of select verbosity levels to the appropriate output stream
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziOutputManager.hpp"
#include "Teuchos_oblackholestream.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

/*!  \class Anasazi::BasicOutputManager

  \brief Anasazi's basic output manager for sending information of select verbosity levels
  to the appropriate output stream.

  \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

namespace Anasazi {

  using std::ostream;

  template <class ScalarType>
  class BasicOutputManager : public OutputManager<ScalarType> {

    public:

      //! @name Constructors/Destructor
      //@{ 

      //! Default constructor
      BasicOutputManager( int vb = Anasazi::Errors, Teuchos::RCP<ostream> os = Teuchos::rcp(&std::cout,false) );

      //! Destructor.
      virtual ~BasicOutputManager() {};
      //@}

      //! @name Set/Get methods
      //@{ 

      //! Set the output stream for this manager.
      void setOStream( Teuchos::RCP<ostream> os );

      //! Get the output stream for this manager.
      Teuchos::RCP<ostream> getOStream();

      //@}

      //! @name Output methods
      //@{ 

      //! Find out whether we need to print out information for this message type.
      /*! This method is used by the solver to determine whether computations are
        necessary for this message type.
        */
      bool isVerbosity( MsgType type ) const;

      //! Send some output to this output stream.
      void print( MsgType type, const std::string output );

      //! Return a stream for outputting to.
      ostream &stream( MsgType type );

      //@}

    private:

      //! @name Undefined methods
      //@{ 

      //! Copy constructor.
      BasicOutputManager( const OutputManager<ScalarType>& OM );

      //! Assignment operator.
      BasicOutputManager<ScalarType>& operator=( const OutputManager<ScalarType>& OM );

      //@}

      Teuchos::RCP<ostream> myOS_;
      Teuchos::oblackholestream myBHS_;
      bool iPrint_;
  };

  template<class ScalarType>
  BasicOutputManager<ScalarType>::BasicOutputManager(int vb, Teuchos::RCP<ostream> os)
  : OutputManager<ScalarType>(vb), myOS_(os) {
    int MyPID;
#ifdef HAVE_MPI
    // Initialize MPI
    int mpiStarted = 0;
    MPI_Initialized(&mpiStarted);
    if (mpiStarted) MPI_Comm_rank(MPI_COMM_WORLD, &MyPID);
    else MyPID=0;
#else 
    MyPID = 0;
#endif
    iPrint_ = (MyPID == 0);
  } 

  template<class ScalarType>
  void BasicOutputManager<ScalarType>::setOStream( Teuchos::RCP<ostream> os ) { 
    myOS_ = os; 
  }

  template<class ScalarType>
  Teuchos::RCP<ostream> BasicOutputManager<ScalarType>::getOStream() { 
    return myOS_; 
  }

  template<class ScalarType>
  bool BasicOutputManager<ScalarType>::isVerbosity( MsgType type ) const {
    if ( (type & this->vb_) == type ) {
      return true;
    }
    return false;
  }

  template<class ScalarType>
  void BasicOutputManager<ScalarType>::print( MsgType type, const std::string output ) {
    if ( (type & this->vb_) == type && iPrint_ ) {
      *myOS_ << output;
    }
  }

  template<class ScalarType>
  ostream & BasicOutputManager<ScalarType>::stream( MsgType type ) {
    if ( (type & this->vb_) == type && iPrint_ ) {
      return *myOS_;
    }
    return myBHS_;
  }

} // end Anasazi namespace

#endif

// end of file AnasaziOutputManager.hpp
