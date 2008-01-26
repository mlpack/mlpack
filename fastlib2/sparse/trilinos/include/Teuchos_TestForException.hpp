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

#ifndef TEUCHOS_TEST_FOR_EXCEPTION_H
#define TEUCHOS_TEST_FOR_EXCEPTION_H

/*! \file Teuchos_TestForException.hpp
\brief Macro for throwing an exception with breakpointing to ease debugging.
*/

#include "Teuchos_TypeNameTraits.hpp"

/*! \defgroup TestForException_grp Utility code for throwing exceptions and setting breakpoints. 
\ingroup teuchos_language_support_grp
*/
//@{

/** \brief Increment the throw number. */
void TestForException_incrThrowNumber();

/** \brief Increment the throw number. */
int TestForException_getThrowNumber();

/** \brief The only purpose for this function is to set a breakpoint. */
void TestForException_break( const std::string &msg );

/** \brief Macro for throwing an exception with breakpointing to ease debugging
 *
 * @param  throw_exception_test
 *               [in] Test for when to throw the exception.  This can and
 *               should be an expression that may mean something to the user.
 *               The text verbatim of this expression is included in the
 *               formed error string.
 * @param  Exception
 *               [in] This should be the name of an exception class.  The
 *               only requirement for this class is that it have a constructor
 *               that accepts an std::string object (as all of the standard
 *               exception classes do).
 * @param  msg   [in] This is any expression that can be included in an
 *               output stream operation.  This is useful when buinding
 *               error messages on the fly.  Note that the code in this
 *               argument only gets evaluated if <tt>throw_exception_test</tt>
 *               evaluates to <tt>true</tt> when an exception is throw.
 *
 * The way that this macro is intended to be used is to 
 * call it in the source code like a function.  For example,
 * suppose that in a piece of code in the file <tt>my_source_file.cpp</tt>
 * that the exception <tt>std::out_of_range</tt> is thrown if <tt>n > 100</tt>.
 * To use the macro, the source code would contain (at line 225
 * for instance):
 \verbatim

 TEST_FOR_EXCEPTION( n > 100, std::out_of_range
    , "Error, n = " << n << is bad" );
 \endverbatim
 * When the program runs and with <tt>n = 125 > 100</tt> for instance,
 * the <tt>std::out_of_range</tt> exception would be thrown with the
 * error message:
 \verbatim

 /home/bob/project/src/my_source_file.cpp:225: n > 100: Error, n = 125 is bad
 \endverbatim
 *
 * In order to debug this, simply open your debugger (gdb for instance),
 * set a break point at <tt>my_soure_file.cpp:225</tt> and then set the condition
 * to break for <tt>n > 100</tt> (e.g. in gdb the command
 * is <tt>cond break_point_number n > 100</tt> and then run the
 * program.  The program should stop a the point in the source file
 * right where the exception will be thrown at but before the exception
 * is thrown.  Try not to use expression for <tt>throw_exception_test</tt> that
 * includes virtual function calls, etc. as most debuggers will not be able to check
 * these types of conditions in order to stop at a breakpoint.  For example,
 * instead of:
 \verbatim

 TEST_FOR_EXCEPTION( obj1->val() > obj2->val(), std::logic_error, "Oh no!" );
 \endverbatim
 * try:
 \verbatim

 double obj1_val = obj1->val(), obj2_val = obj2->val();
 TEST_FOR_EXCEPTION( obj1_val > obj2_val, std::logic_error, "Oh no!" );
 \endverbatim
 * If the developer goes to the line in the source file that is contained
 * in the error message of the exception thrown, he/she will see the
 * underlying condition.
 *
 * As an alternative, you can set a breakpoint for any exception thrown
 * by setting a breakpoint in the function <tt>ThrowException_break()</tt>.
 */
#define TEST_FOR_EXCEPTION(throw_exception_test,Exception,msg) \
{ \
    const bool throw_exception = (throw_exception_test); \
    if(throw_exception) { \
      TestForException_incrThrowNumber(); \
      std::ostringstream omsg; \
	    omsg \
        << __FILE__ << ":" << __LINE__ << ":\n\n" \
        << "Throw number = " << TestForException_getThrowNumber() << "\n\n" \
        << "Throw test that evaluated to true: "#throw_exception_test << "\n\n" \
        << msg; \
      const std::string &omsgstr = omsg.str(); \
      TestForException_break(omsgstr); \
      throw Exception(omsgstr); \
    } \
}

/** \brief Macro for throwing an exception with breakpointing to ease debugging
 *
 * This macro is equivalent to the <tt>TEST_FOR_EXCEPTION()</tt> macro except
 * the file name, line number, and test condition are not printed.
 */
#define TEST_FOR_EXCEPTION_PURE_MSG(throw_exception_test,Exception,msg) \
{ \
    const bool throw_exception = (throw_exception_test); \
    if(throw_exception) { \
      TestForException_incrThrowNumber(); \
      std::ostringstream omsg; \
	    omsg << msg; \
      omsg << "\n\nThrow number = " << TestForException_getThrowNumber() << "\n\n"; \
      const std::string &omsgstr = omsg.str(); \
      TestForException_break(omsgstr); \
      throw Exception(omsgstr); \
    } \
}

/** \brief This macro is designed to be a short version of
 * <tt>TEST_FOR_EXCEPTION()</tt> that is easier to call.
 *
 * @param  throw_exception_test
 *               [in] Test for when to throw the exception.  This can and
 *               should be an expression that may mean something to the user.
 *               The text verbatim of this expression is included in the
 *               formed error string.
 *
 * \note The exception thrown is <tt>std::logic_error</tt>.
 */
#define TEST_FOR_EXCEPT(throw_exception_test) \
  TEST_FOR_EXCEPTION(throw_exception_test,std::logic_error,"Error!")

/** \brief This macro is the same as <tt>TEST_FOR_EXCEPTION()</tt> except that the
 * exception will be caught, the message printed, and then rethrown.
 *
 * @param  throw_exception_test
 *               [in] See <tt>TEST_FOR_EXCEPTION()</tt>.
 * @param  Exception
 *               [in] See <tt>TEST_FOR_EXCEPTION()</tt>.
 * @param  msg   [in] See <tt>TEST_FOR_EXCEPTION()</tt>.
 * @param  out_ptr
 *               [in] If <tt>out_ptr!=NULL</tt> then <tt>*out_ptr</tt> will receive
 *               a printout of a line of output that gives the exception type and
 *               the error message that is generated.
 */
#define TEST_FOR_EXCEPTION_PRINT(throw_exception_test,Exception,msg,out_ptr) \
try { \
  TEST_FOR_EXCEPTION(throw_exception_test,Exception,msg); \
} \
catch(const std::exception &except) { \
  std::ostream *l_out_ptr = (out_ptr); \
  if(l_out_ptr) { \
    *l_out_ptr \
      << "\nThorwing an std::exception of type \'"<<Teuchos::typeName(except)<<"\' with the error message: " \
      << except.what(); \
  } \
  throw; \
}

/** \brief This macro is the same as <tt>TEST_FOR_EXCEPT()</tt> except that the
 * exception will be caught, the message printed, and then rethrown.
 *
 * @param  throw_exception_test
 *               [in] See <tt>TEST_FOR_EXCEPT()</tt>.
 * @param  out_ptr
 *               [in] If <tt>out_ptr!=NULL</tt> then <tt>*out_ptr</tt> will receive
 *               a printout of a line of output that gives the exception type and
 *               the error message that is generated.
 */
#define TEST_FOR_EXCEPT_PRINT(throw_exception_test,out_ptr) \
  TEST_FOR_EXCEPTION_PRINT(throw_exception_test,std::logic_error,"Error!",out_ptr)


/** \brief This macro intercepts an exception, prints a standardized message including
 * the current filename and line number, and then throws the exception up the stack
 * @param exc [in] the exception that has been caught
 */
#define TEUCHOS_TRACE(exc)\
{ \
  std::ostringstream omsg; \
	omsg << exc.what() << std::endl \
       << "caught in " << __FILE__ << ":" << __LINE__ << std::endl ; \
  throw std::runtime_error(omsg.str()); \
}


//@}

#endif // TEUCHOS_TEST_FOR_EXCEPTION_H
