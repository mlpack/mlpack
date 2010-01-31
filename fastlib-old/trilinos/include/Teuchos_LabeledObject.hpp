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
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
// @HEADER

#ifndef TEUCHOS_LABELED_OBJECT_HPP
#define TEUCHOS_LABELED_OBJECT_HPP

#include "Teuchos_ConfigDefs.hpp"


namespace Teuchos {


/** \brief Base class for objects that contain a std::string label.
 *
 * The object label std::string <tt>objectLabel</tt> set in
 * <tt>setObjectLabel()</tt> should be a simple one-line label given to an
 * object to differentiate it from all other objects.  A subclass
 * implementation can define a default label in some cases but typically this
 * label is designed for end users to set to give the object a name that is
 * meaningful to the user.  The label should not contain any information about
 * the actual type of the object.  Adding type information is appropriate in
 * the <tt>Describable</tt> interface, which inherits from this interface.
 *
 * This base class provides a default implementation for the functions
 * <tt>setObjectLabel()</tt> and <tt>getObjectLabel()</tt> as well as private
 * data to hold the label.  Subclasses can override these functions but
 * general, there should be no need to do so.
 *
 * \ingroup teuchos_outputting_grp
 */
class LabeledObject {
public:
  /** \brief Construct with an empty label. */
  LabeledObject();
  /** \brief . */
  virtual ~LabeledObject();
  /** \brief Set the object label (see LabeledObject). */
  virtual void setObjectLabel( const std::string &objectLabel );
  /** \brief Get the object label (see LabeledObject). */
  virtual std::string getObjectLabel() const;
private:
  std::string objectLabel_;
};


} // namespace Teuchos


#endif // TEUCHOS_LABELED_OBJECT_HPP
