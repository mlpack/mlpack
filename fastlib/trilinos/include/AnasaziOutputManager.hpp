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

#ifndef ANASAZI_OUTPUT_MANAGER_HPP
#define ANASAZI_OUTPUT_MANAGER_HPP

/*!     \file AnasaziOutputManager.hpp
        \brief Abstract class definition for Anasazi Output Managers.
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"

/*!  \class Anasazi::OutputManager

  \brief Output managers remove the need for the eigensolver to know any information
  about the required output.  Calling isVerbosity( MsgType type ) informs the solver if
  it is supposed to output the information corresponding to the message type.

  \author Chris Baker, Ulrich Hetmaniuk, Rich Lehoucq, and Heidi Thornquist
*/

namespace Anasazi {

template <class ScalarType>
class OutputManager {

  public:

    //!@name Constructors/Destructor 
  //@{ 

  //! Default constructor
  OutputManager( int vb = Anasazi::Errors ) : vb_(vb) {};

  //! Destructor.
  virtual ~OutputManager() {};
  //@}
  
  //! @name Set/Get methods
  //@{ 

  //! Set the message output types for this manager.
  virtual void setVerbosity( int vb ) { vb_ = vb; }

  //! Get the message output types for this manager.
  virtual int getVerbosity( ) const { return vb_; }

  //@}

  //! @name Output methods
  //@{ 

  //! Find out whether we need to print out information for this message type.
  /*! This method is used by the solver to determine whether computations are
      necessary for this message type.
  */
  virtual bool isVerbosity( MsgType type ) const = 0;

  //! Send output to the output manager.
  virtual void print( MsgType type, const std::string output ) = 0;

  //! Create a stream for outputting to.
  virtual std::ostream &stream( MsgType type ) = 0;

  //@}

  private:

  //! @name Undefined methods
  //@{ 

  //! Copy constructor.
  OutputManager( const OutputManager<ScalarType>& OM );

  //! Assignment operator.
  OutputManager<ScalarType>& operator=( const OutputManager<ScalarType>& OM );

  //@}

  protected:
  int vb_;
};

} // end Anasazi namespace

#endif

// end of file AnasaziOutputManager.hpp
