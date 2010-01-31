// @HEADER
// ***********************************************************************
// 
//                    Teuchos: Common Tools Package
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

// Kris
// 07.08.03 -- Move into Teuchos package/namespace

#ifndef _TEUCHOS_OBJECT_HPP_
#define _TEUCHOS_OBJECT_HPP_

/*! \file Teuchos_Object.hpp
    \brief The base Teuchos object.
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_DataAccess.hpp"

/*! \class Teuchos::Object
    \brief The base Teuchos class.

    The Object class provides capabilities common to all Teuchos objects,
    such as a label that identifies an object instance, constant definitions,
    enum types.
*/

namespace Teuchos
{

class Object
{
  public:
  //! @name Constructors/Destructor.
  //@{ 
  //! Default Constructor.
  /*! Object is the primary base class in Teuchos.  All Teuchos class
      are derived from it, directly or indirectly.  This class is seldom
      used explictly.
  */
  Object(int tracebackModeIn = -1);

  //! Labeling Constructor.
  /*! Creates an Object with the given label.
  */
  Object(const char* label, int tracebackModeIn = -1);

  //! Copy Constructor.
  /*! Makes an exact copy of an existing Object instance.
  */
  Object(const Object& obj);

  //! Destructor.
  /*! Completely deletes an Object object.  
  */
  virtual ~Object();

  //@}
  
  //! @name Set methods.
  //@{ 

  //! Define object label using a character std::string.
  /*! Defines the label used to describe \c this object.
  */
  virtual void setLabel(const char* label);

  //! Set the value of the Object error traceback report mode.
  /*! Sets the integer error traceback behavior.
      TracebackMode controls whether or not traceback information is printed when run time
      integer errors are detected:

      <= 0 - No information report

       = 1 - Fatal (negative) values are reported

      >= 2 - All values (except zero) reported.

      \note Default is set to -1 when object is constructed.
  */
  static void setTracebackMode(int tracebackModeValue);

  //@}

  //! @name Accessor methods.
  //@{ 

  //! Access the object label.
  /*! Returns the std::string used to define \e this object.
  */
  virtual char* label() const;  

  //! Get the value of the Object error traceback report mode.
  static int getTracebackMode();

  //@}

  //! @name I/O method.
  //@{ 

  //! Print method for placing the object in an output stream
  virtual void print(std::ostream& os) const;
  //@}

  //! @name Error reporting method.
  //@{ 

  //!  Method for reporting errors with Teuchos objects.
  virtual int reportError(const std::string message, int errorCode) const 
  {
  // NOTE:  We are extracting a C-style std::string from Message because 
  //        the SGI compiler does not have a real std::string class with 
  //        the << operator.  Some day we should get rid of ".c_str()"
	if ( (tracebackMode==1) && (errorCode < 0) )
	{  // Report fatal error
	   std::cerr << std::endl << "Error in Teuchos Object with label: " << label_ << std::endl 
		 << "Teuchos Error:  " << message.c_str() << "  Error Code:  " << errorCode << std::endl;
	   return(errorCode);
        }
	if ( (tracebackMode==2) && (errorCode != 0 ) ) 
	{
	   std::cerr << std::endl << "Error in Teuchos Object with label: " << label_ << std::endl 
		 << "Teuchos Error:  " << message.c_str() << "  Error Code:  " << errorCode << std::endl;
	   return(errorCode);
	}
	return(errorCode);
  }

  //@}

  static int tracebackMode;  

 protected:

 private:

  char* label_;

}; // class Object

/*! \relates Object
    Output stream operator for handling the printing of Object.
*/
inline std::ostream& operator<<(std::ostream& os, const Teuchos::Object& Obj)
{
  os << Obj.label() << std::endl;
  Obj.print(os);
 
  return os;
}

} // namespace Teuchos

// #include "Teuchos_Object.cpp"


#endif /* _TEUCHOS_OBJECT_HPP_ */
