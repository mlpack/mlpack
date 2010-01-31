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

#ifndef TEUCHOS_TABLEENTRY_H
#define TEUCHOS_TABLEENTRY_H

/** \file Teuchos_TableEntry.hpp
    \brief Base class for representing compound entries in a printed
    * table of data. 
    * "Compound" means that each entry may be some aggregation 
    * of more than one item,
    * for example a timer together with a number of calls, or a 
    * value together with its estimated measurement error.
    */

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include <iostream>

namespace Teuchos
{
  /**
   * \brief An entry, perhaps compound, to be written into a table. 
   * 
   * KL 30 Apr 2006 -- initial design. Can you say overengineering??
   * The complexity is to support a nice interface for pair entries
   * such as time/numCalls.
   */
  class TableEntry
  {
  public:
    /** \brief Empty ctor. */
    TableEntry() {}

    /** \brief virtual dtor */
    virtual ~TableEntry() {}
    
    /**  \brief Return a std::string representation of this entry */
    virtual std::string toString() const = 0 ;

    /** \brief Return a std::string representation of this entry,
     * truncated if necessary to fit within the given column width.
     *
     * \param maxWidth [in] the maximum width of the std::string form. Larger
     * strings must be truncated in a subclass-dependent way.
     * \return the std::string, truncated if necessary
     */
    virtual std::string toChoppedString(int maxWidth) const ;

  protected:
  };


  /** 
   * \brief A table entry that is a simple double-precision number
   */
  class DoubleEntry : public TableEntry
  {
  public:
    /** \brief Construct with a value
     * and a precision */
    DoubleEntry(const double& value, int precision);
    
    /** \brief Write the specified entry to a std::string */
    virtual std::string toString() const ;

  private:
    double data_;
    int precision_;
  };

  
  /** 
   * \brief A table entry that is a simple integer
   */
  class IntEntry : public TableEntry
  {
  public:
    /** \brief Construct with a value */
    IntEntry(int value);
    
    /** \brief Write the specified entry to a std::string */
    virtual std::string toString() const ;

  private:
    int data_;
  };

  
  /** 
   * \brief A table entry that is a simple std::string
   */
  class StringEntry : public TableEntry
  {
  public:
    /** \brief Construct with a value */
    StringEntry(std::string value);
    
    /** \brief Write the specified entry to a std::string */
    virtual std::string toString() const ;

  private:
    std::string data_;
  };

  /** 
   * \brief An entry containing two subentries, with the second
   * to be written in parentheses after the first. For example,
   * \code
   * 1.23(456)
   * \endcode 
   * The two subentries can be any type of data, each represented 
   * with a TableEntry derived type. 
   */
  class CompoundEntryWithParentheses : public TableEntry
  {
  public:
    /** \brief */
    CompoundEntryWithParentheses(const RCP<TableEntry>& first,
                                 const RCP<TableEntry>& second,
                                 bool spaceBeforeParens=true);
    
    /** \brief Write the specified entry to a std::string */
    virtual std::string toString() const ;
    
  private:
    RCP<TableEntry> first_;
    RCP<TableEntry> second_;
    bool spaceBeforeParens_;
  };

  

 

}
#endif
