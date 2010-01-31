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

#ifndef TEUCHOS_ARRAY_H
#define TEUCHOS_ARRAY_H

/*! \file Teuchos_Array.hpp
    \brief Templated array class derived from the STL std::vector
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_Utils.hpp"
#include "Teuchos_TypeNameTraits.hpp"

namespace Teuchos
{

  /** \brief .
   * \relates Array
   */
  class InvalidArrayStringRepresentation : public std::logic_error
  {public:InvalidArrayStringRepresentation(const std::string& what_arg) : std::logic_error(what_arg) {}};

  /**
   * \brief Array is a templated array class derived from the STL std::vector, but with
   * index boundschecking and an extended interface.
   */
  template<class T>
  class Array : public std::vector<T>
  {
  public:
    //! Empty constructor
    Array();

    //! Allocate an array with n elements 
    Array(int n);

    //! Allocate n elements, and fill with value \c t
    Array(int n, const T& t);

    //! Add a new entry at the end of the array. Resize to allow space for the new entry.
    inline Array<T>& append(const T& entry) {this->push_back(entry); return *this;}

    //! Remove the i-th element from the array, with optional boundschecking.
    void remove(int i);

    /*! \brief Return number of elements in the array. 
     *	Equivalent to size(), but included for backwards compatibility.
     */
    int length() const {return this->size();}

    //! Read/Write access to a the i-th element, with optional boundschecking.
    inline T& operator[](int i);

    //! Read-only access to a the i-th element, with optional boundschecking.
    inline const T& operator[](int i) const;

    //! Write Array as a std::string
    std::string toString() const ;

    //! Return true if Array has been compiled with boundschecking on 
    static bool hasBoundsChecking();

  private:

    /** check for a bounds violation if HAVE_ARRAY_BOUNDSCHECK has been
     * defined as 1. */
    void indexCheckCrash(int i) const;
  };

  /** \relates Array 
      \brief Write an Array to a stream
  */
  template<class T> std::ostream& operator<<(std::ostream& os, 
                                             const Array<T>& array);

  /** \relates Array */
  template<class T> int hashCode(const Array<T>& array);

  /** \relates Array */
  template<class T> std::string toString(const Array<T>& array);

  template<class T> inline Array<T>::Array()
    : std::vector<T>()
  {}

  template<class T> inline Array<T>::Array(int n)
    : std::vector<T>(n)
  {}

  template<class T> inline Array<T>::Array(int n, const T& t)
    : std::vector<T>(n, t)
  {}

  template<class T>
  void Array<T>::remove(int i) {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    indexCheckCrash(i);
#endif
    // Erase the i-th element of this array.
    this->erase( this->begin() + i );
  }

  template<class T> inline
  T& Array<T>::operator[](int i) {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    indexCheckCrash(i);
#endif
    return std::vector<T>::operator[](i);
  }

  template<class T> inline
  const T& Array<T>::operator[](int i) const {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
    indexCheckCrash(i);
#endif
    return std::vector<T>::operator[](i);
  }

  template<class T> inline
  bool Array<T>::hasBoundsChecking()
  {
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK  
    return true;
#else
    return false;
#endif
  }

  template<class T> inline
  void Array<T>::indexCheckCrash(int i) const
  {
    TEST_FOR_EXCEPTION(
      !( 0 <= i && i < length() ), std::range_error,
      "Array<"<<TypeNameTraits<T>::name()<<">::indexCheckCrash: "
      "index " << i << " out of range [0, "<< length() << ")"
      );
  }

  // print in form (), (1), or (1,2)
  template<class T> inline std::ostream& operator<<(std::ostream& os, const Array<T>& array)
  {
    return os << Teuchos::toString(array);
  }

  template<class T> inline int hashCode(const Array<T>& array)
  {
    int rtn = hashCode(array.length());
    for (int i=0; i<array.length(); i++)
      {
        rtn += hashCode(array[i]);
      }
    return rtn;
  }

  template<class T> inline std::string Array<T>::toString() const
  {
    std::ostringstream ss;
    ss << "{";

    for (int i=0; i<length(); i++)
      {
        ss << operator[](i);
        if (i<length()-1) ss << ", ";
      }
    ss << "}";

    return ss.str();
  }

  /** \relates Array. */
  template<class T> inline
  std::string toString(const Array<T>& array)
  {
    return array.toString();
  }

  /** \brief Converts from std::string representation (as created by
   * <tt>toString()</tt>) back into the array object.
   *
   * \param  arrayStr
   *           [in] The std::string representation of the array (see below).
   *
   * <b>Exceptions:</b> If the std::string representation is not valid, then an
   * std::exception of type <tt>InvalidArrayStringRepresentation</tt> with be
   * thrown with a decent error message attached.
   *
   * The formating of the std::string <tt>arrayStr</tt> must look like:
   
   \verbatim

      {  val[0], val[1], val[2], val[3], ..., val[n-1] }

   \endverbatim

   * Currently <tt>operator>>()</tt> is used to convert the entries from their
   * std::string representation to objects of type <tt>T</tt>.  White space is
   * unimportant and the parser keys off of ',', '{' and '}' so even newlines
   * are allowed.  In the future, a traits class might be defined that will
   * allow for finer-grained control of how the conversion from strings to
   * values is performed in cases where <tt>operator>>()</tt> does not exist
   * for certain types.
   *
   * <b>Warning!</b> Currently this function only supports reading in flat
   * array objects for basic types like <tt>bool</tt>, <tt>int</tt>, and
   * <tt>double</tt> and does not yet support nested arrays (i.e. no
   * <tt>Array<Array<int> ></tt>) or other such fancy nested types.  Support
   * for nested arrays and other user defined types <tt>T</tt> can be added in
   * the future with no impact on user code.  Only the parser for the array
   * needs to be improved.  More specifically, the current implementation will
   * not work for any types <tt>T</tt> who's std::string representation contains
   * the characters <tt>','</tt> or <tt>'}'</tt>.  This implementation can be
   * modified to allow any such types by watching for the nesting of common
   * enclosing structures like <tt>[...]</tt>, <tt>{...}</tt> or
   * <tt>(...)</tt> within each entry of the std::string representation.  However,
   * this should all just work fine on most machines for the types
   * <tt>int</tt>, <tt>bool</tt>, <tt>float</tt>, <tt>double</tt> etc.
   *
   * <b>Warning!</b> Trying to read in an array in std::string format of doubles in
   * scientific notation such as <tt>{1e+2,3.53+6,...}</tt> into an array
   * object such as <tt>Array<int></tt> will not yield the correct results.
   * If one wants to allow a neutral std::string representation to be read in as an
   * <tt>Array<double></tt> object or an <tt>Array<int></tt> object, then
   * general formating such as <tt>{100,3530000,...}</tt> should be used.
   * This templated function is unable to deal std::complex type conversion issues.
   * 
   * \relates Array.
   */
  template<class T>
  Array<T> fromStringToArray(const std::string& arrayStr)
  {
    const std::string str = Utils::trimWhiteSpace(arrayStr);
    std::istringstream iss(str);
    TEST_FOR_EXCEPTION(
      ( str[0]!='{' || str[str.length()-1] != '}' )
      ,InvalidArrayStringRepresentation
      ,"Error, the std::string:\n"
      "----------\n"
      <<str<<
      "\n----------\n"
      "is not a valid array represntation!"
      );
    char c;
    c = iss.get(); // Read initial '{'
    TEST_FOR_EXCEPT(c!='{'); // Should not throw!
    // Now we are ready to begin reading the entries of the array!
    Array<T> a;
    while( !iss.eof() ) {
      // Get the basic entry std::string
      std::string entryStr;
      std::getline(iss,entryStr,','); // Get next entry up to ,!
      // ToDo: Above, we might have to be careful to look for the opening and
      // closing of parentheses in order not to pick up an internal ',' in the
      // middle of an entry (for a std::complex number for instance).  The above
      // implementation assumes that there will be no commas in the middle of
      // the std::string representation of an entry.  This is certainly true for
      // the types bool, int, float, and double.
      //
      // Trim whitespace from beginning and end
      entryStr = Utils::trimWhiteSpace(entryStr);
      // Remove the final '}' if this is the last entry and we did not
      // actually terminate the above getline(...) on ','
      bool found_end = false;
      if(entryStr[entryStr.length()-1]=='}') {
        entryStr = entryStr.substr(0,entryStr.length()-1);
        found_end = true;
        if( entryStr.length()==0 && a.size()==0 )
          return a; // This is the empty array "{}" (with any spaces in it!)
      }
      TEST_FOR_EXCEPTION(
        0 == entryStr.length()
        ,InvalidArrayStringRepresentation
        ,"Error, the std::string:\n"
        "----------\n"
        <<str<<
        "\n----------\n"
        "is not a valid array represntation!"
        );
      // Finally we can convert the entry and add it to the array!
      std::istringstream entryiss(entryStr);
      T entry;
      entryiss >> entry; // Assumes type has operator>>(...) defined!
      // ToDo: We may need to define a traits class to allow us to specialized
      // how conversion from a std::string to a object is done!
      a.push_back(entry);
      // At the end of the loop body here, if we have reached the last '}'
      // then the input stream iss should be empty and iss.eof() should be
      // true, so the loop should terminate.  We put an std::exception test here
      // just in case something has gone wrong.
      TEST_FOR_EXCEPTION(
        found_end && !iss.eof()
        ,InvalidArrayStringRepresentation
        ,"Error, the std::string:\n"
        "----------\n"
        <<str<<
        "\n----------\n"
        "is not a valid array represntation!"
        );
    }
    return a;
  }
                      
  /** \relates Array 
      \brief Create an array with one entry 
  */
  template<class T> inline
  Array<T> tuple(const T& a)
  {
    Array<T> rtn(1, a);
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with two entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b)
  {
    Array<T> rtn(2);
    rtn[0] = a;
    rtn[1] = b;
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with three entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c)
  {
    Array<T> rtn(3);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with four entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d)
  {
    Array<T> rtn(4);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with five entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d, const T& e)
  {
    Array<T> rtn(5);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    rtn[4] = e;
    return rtn;
  }


  /** \relates Array 
      \brief Create an array with six entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d, const T& e,
                 const T& f)
  {
    Array<T> rtn(6);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    rtn[4] = e;
    rtn[5] = f;
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with seven entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d, const T& e,
                 const T& f, const T& g)
  {
    Array<T> rtn(7);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    rtn[4] = e;
    rtn[5] = f;
    rtn[6] = g;
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with eight entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d, const T& e,
                 const T& f, const T& g, const T& h)
  {
    Array<T> rtn(8);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    rtn[4] = e;
    rtn[5] = f;
    rtn[6] = g;
    rtn[7] = h;
    return rtn;
  }

  /** \relates Array 
      \brief Create an array with nine entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d, const T& e,
                 const T& f, const T& g, const T& h, const T& i)
  {
    Array<T> rtn(9);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    rtn[4] = e;
    rtn[5] = f;
    rtn[6] = g;
    rtn[7] = h;
    rtn[8] = i;
    return rtn;
  }


  /** \relates Array 
      \brief Create an array with ten entries 
  */
  template<class T> inline
  Array<T> tuple(const T& a, const T& b, const T& c, const T& d, const T& e,
                 const T& f, const T& g, const T& h, const T& i, const T& j)
  {
    Array<T> rtn(10);
    rtn[0] = a;
    rtn[1] = b;
    rtn[2] = c;
    rtn[3] = d;
    rtn[4] = e;
    rtn[5] = f;
    rtn[6] = g;
    rtn[7] = h;
    rtn[8] = i;
    rtn[9] = j;
    return rtn;
  }
}

#endif // TEUCHOS_ARRAY_H
