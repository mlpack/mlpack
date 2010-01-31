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

#ifndef TEUCHOS_TABLEFORMAT_H
#define TEUCHOS_TABLEFORMAT_H

/*! \file Teuchos_TableFormat.hpp
  \brief Provides utilities for formatting tabular output
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_TableColumn.hpp"
#include <iostream>

namespace Teuchos
{

/** \brief Encapsulation of formatting specifications for writing data
 * in a clean tabular form.
 *
 * Note: it is left to the programmer to avoid invalid settings such as
 * negative column spaces, zero page widths, and other such potentially
 * bad things.
 *
 * KL 30 Apr 2006 -- initial design.
 */
class TableFormat
{
public:
  /** \brief Construct with a header and default format settings */
  TableFormat()
    : pageWidth_(80), precision_(4), columnSpacing_(4),
      maxNameSize_(40), columnWidths_(), lineInterval_(10)
    {}
    
  /** \brief Get the maximum number of characters per line.
   * Default is 80. */
  int pageWidth() const {return pageWidth_;}
    
  /** \brief Get the precision for writing doubles.
   * Default is 4. */
  int precision() const {return precision_;}

  /** \brief Get the number of characters to be left as blank 
   * spaces in each column. Default is 4. */
  int columnSpacing() const {return columnSpacing_;}

  /** \brief Set the number of characters on a line.
   * This quantity can be updated within the const 
   * method writeWholeTables() */
  void setPageWidth(int pw) const {pageWidth_ = pw;}

  /** \brief Set the precision for writing doubles */
  void setPrecision(int p) {precision_ = p;}

  /** \brief Set the number of characters to be left as blank spaces in each column */
  void setColumnSpacing(int columnSpacing) {columnSpacing_ = columnSpacing;}

  /** \brief Set the interval at which a horizontal line will be written between
   * rows. 
   *
   * \break lineInterval [in] the number of rows between each horizontal line
   */
  void setRowsBetweenLines(int lineInterval) {lineInterval_=lineInterval;}

  /** \brief Return a horizontal line in dashes "----" 
   * the width of the page. 
   *
   * Originally called <tt>hbar</tt>, but changed to avoid 
   * possible confusion for physicists expecting <tt>hbar()</tt> to return 
   * \f$1.05457168e-34\f$ :-).  */
  std::string thinline() const ;

  /** \brief Return a thick horizontal line in equal signs "====" the 
   * width of the page */
  std::string thickline() const ;

  /** \brief Return a std::string full of blanks up to the requested size */
  std::string blanks(int size) const ;

  /** \brief Computes the column width required to write all values
   * to the required precision.
   *
   * \param name [in] the title of the column
   * \param column [in] the column data
   *
   * Postcondition: colString.size()==values.size()
   */
  int computeRequiredColumnWidth(const std::string& name,
    const TableColumn& column) const ;

  /** \brief Set the column widths to be used for subsequent rows */
  void setColumnWidths(const Array<int>& colWidths) 
    {columnWidths_ = colWidths;}

  /** \brief Write the row of entries.
   *
   * \param out [in/out] the output stream to which the row will be written
   * \param entries [in] the data to be written into this row. Each array
   * element is the entry for a column on this row.
   */ 
  void writeRow(
    std::ostream& out,
    const Array<RCP<TableEntry> >& entries
    ) const;

  /** \brief Write the row of entries.
   *
   * \param out [in/out] the output stream to which the row will be written
   * \param columns [in] the columns of data from which this row is to be sliced
   * \param rowIndex [in] the index into the columns used to obtain the values for
   * this row
   */ 
  void writeRow(
    std::ostream& out,
    int rowIndex,
    const Array<TableColumn>& columns
    ) const;

  /** \brief . */
  void writeWholeTable(
    std::ostream& out,
    const std::string& tableTitle,
    const Array<std::string>& columnNames,
    const Array<TableColumn>& columns
    ) const ;

protected:

  int defaultColumnWidth() const {return 20;}

private:

  mutable int pageWidth_;
  int precision_;
  int columnSpacing_;
  int maxNameSize_;
  Array<int> columnWidths_;
  int lineInterval_;
};


} // namespace Teuchos


#endif
