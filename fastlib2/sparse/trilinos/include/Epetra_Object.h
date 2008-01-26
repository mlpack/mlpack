
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_OBJECT_H
#define EPETRA_OBJECT_H

#include "Epetra_CombineMode.h"
#include "Epetra_DataAccess.h"
#include "Epetra_ConfigDefs.h"

//! Epetra_Object:  The base Epetra class.
/*! The Epetra_Object class provides capabilities common to all Epetra objects,
    such as a label that identifies an object instance, constant definitions,
    enum types.
  
*/
class Epetra_Object {
    
  public:
    //! @name Constructors/destructor
  //@{ 
  //! Epetra_Object Constructor.
  /*! Epetra_Object is the primary base class in Epetra.  All Epetra class
      are derived from it, directly or indirectly.  This class is seldom
      used explictly.
  */
  Epetra_Object(int TracebackModeIn = -1, bool set_label = true);

  //! Epetra_Object Constructor.
  /*! Creates a Epetra_Object with the given label.
  */
  Epetra_Object(const char * const Label, int TracebackModeIn = -1);

  //! Epetra_Object Copy Constructor.
  /*! Makes an exact copy of an existing Epetra_Object instance.
  */
  Epetra_Object(const Epetra_Object& Object);

  //! Epetra_Object Destructor.
  /*! Completely deletes a Epetra_Object object.  
  */
  virtual ~Epetra_Object();
  //@}
  
  //! @name Attribute set/get methods
  //@{ 

  //! Epetra_Object Label definition using char *.
  /*! Defines the label used to describe the \e this object.  
  */
  virtual void SetLabel(const char * const Label);

  //! Epetra_Object Label access funtion.
  /*! Returns the string used to define this object.  
  */
  virtual const char * Label() const;

  //! Set the value of the Epetra_Object error traceback report mode.
  /*! Sets the integer error traceback behavior.  
      TracebackMode controls whether or not traceback information is printed when run time 
      integer errors are detected:

      <= 0 - No information report

       = 1 - Fatal (negative) values are reported

      >= 2 - All values (except zero) reported.

      Default is set to 1.
  */
  static void SetTracebackMode(int TracebackModeValue);

  //! Get the value of the Epetra_Object error report mode.
  static int GetTracebackMode();

  //! Get the output stream for error reporting
  static std::ostream& GetTracebackStream();

  //@}

  //! @name Miscellaneous
  //@{ 

  //! Print object to an output stream
  //! Print method
  virtual void Print(ostream & os) const;

  //! Error reporting method
  virtual int ReportError(const string Message, int ErrorCode) const;
  //@}

  
// TracebackMode controls how much traceback information is printed when run time 
// integer errors are detected:
// = 0 - No information report
// = 1 - Fatal (negative) values are reported
// = 2 - All values (except zero) reported.

// Default is set to 1.  Can be set to different value using SetTracebackMode() method in
// Epetra_Object class
  static int TracebackMode;


 protected:
  string toString(const int& x) const {
     char s[100];
     sprintf(s, "%d", x);
     return string(s);
}

  string toString(const double& x) const {
     char s[100];
     sprintf(s, "%g", x);
     return string(s);
}
  

 private:
  Epetra_Object& operator=(const Epetra_Object& src) {
    SetLabel(src.Label());
    return *this;
  }

  char * Label_;

};

inline ostream& operator<<(ostream& os, const Epetra_Object& obj)
{
  if (Epetra_FormatStdout) {
/*    const Epetra_fmtflags  olda = os.setf(ios::right,ios::adjustfield);
    const Epetra_fmtflags  oldf = os.setf(ios::scientific,ios::floatfield);
    const int              oldp = os.precision(12); */

    os << obj.Label() << endl;
    obj.Print(os);

/*    os.setf(olda,ios::adjustfield);
    os.setf(oldf,ios::floatfield);
    os.precision(oldp); */
  }
  else {

    os << obj.Label();
    obj.Print(os);
  }
  
  return os;
}

/** \brief Macro for testing for and throwing and int exception for objects
 * derived from Epetra_Object.
 *
 * This macro adds the file name and line number to teh 
 */
#define EPETRA_TEST_FOR_EXCEPTION(throw_exception_test,errCode,msg) \
{ \
    const bool throw_exception = (throw_exception_test); \
    if(throw_exception) { \
        std::ostringstream omsg; \
	    omsg \
        << __FILE__ << ":" << __LINE__ << ":" \
        << " Throw test that evaluated to true: "#throw_exception_test << ":" \
        << "Error message : " << msg; \
	    throw ReportError(omsg.str(),errCode); \
    } \
}

#endif /* EPETRA_OBJECT_H */
