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

#ifndef TEUCHOS_CONST_NONCONST_OBJECT_CONTAINER_HPP
#define TEUCHOS_CONST_NONCONST_OBJECT_CONTAINER_HPP

#include "Teuchos_RCP.hpp"

namespace Teuchos {

/** \brief Simple class for containing an object and protecting const with
 * a runtime check which throws an std::exception.
 *
 * This class is simple enough and developers are encouraged to look at the
 * simple inline definition of this class.
 *
 * The default copy constructor and assignment operator functions are allowed
 * and result in shallow copied (i.e. just the RCP objects are copied).
 * However, the protection of const will be maintained in the copied/assigned
 * objects as well.
 */
template<class ObjType>
class ConstNonconstObjectContainer {
public:
  /** \brief. Constructs to uninitialized */
  ConstNonconstObjectContainer()
    :constObj_(null),isConst_(true) {}
  /** \brief. Calls <tt>initialize()</tt> with a non-const object. */
  ConstNonconstObjectContainer( const RCP<ObjType> &obj )
    { initialize(obj); }
  /** \brief. Calls <tt>initialize()</tt> with a const object. */
  ConstNonconstObjectContainer( const RCP<const ObjType> &obj )
    { initialize(obj); }
  /** \brief. Initialize using a non-const object.
   * Allows both const and non-const access to the contained object. */
  void initialize( const RCP<ObjType> &obj )
    { TEST_FOR_EXCEPT(!obj.get()); constObj_=obj; isConst_=false; }
  /** \brief. Initialize using a const object.
   * Allows only const access enforced with a runtime check. */
  void initialize( const RCP<const ObjType> &obj )
    { TEST_FOR_EXCEPT(!obj.get()); constObj_=obj; isConst_=true; }
  /** \brief. Uninitialize. */
  void uninitialize()
    { constObj_=null; isConst_=true; }
  /** \brief Returns true if const-only access to the object is allowed. */
  bool isConst() const
    { return isConst_; }
  /** \brief Get an RCP to the non-const contained object.
   *
   * <b>Preconditions:</b>
   * <ul>
   * <li> [<tt>getConstObj().get()!=NULL</tt>] <tt>isConst()==false</tt>
   * </ul>
   *
   * <b>Postconditions:</b>
   * <ul>
   * <li>[<tt>getConstObj().get()==NULL</tt>] <tt>return.get()==NULL</tt>
   * <li>[<tt>getConstObj().get()!=NULL</tt>] <tt>return.get()!=NULL</tt>
   * </ul>
   */
  RCP<ObjType> getNonconstObj()
    {
      TEST_FOR_EXCEPTION(
        constObj_.get() && isConst_, std::logic_error
        ,"Error, the object of reference type \""<<TypeNameTraits<ObjType>::name()<<"\" was given "
        "as a const-only object and non-const access is not allowed."
        );
      return rcp_const_cast<ObjType>(constObj_);
    }
  /** \brief Get an RCP to the const contained object.
   *
   * If <tt>return.get()==NULL</tt>, then this means that no object was given
   * to <tt>*this</tt> data container object.
   */
  RCP<const ObjType> getConstObj() const
    { return constObj_; }
  /** \brief Perform shorthand for <tt>getConstObj(). */
  RCP<const ObjType> operator()() const
    { return getConstObj(); }
  
private:
  RCP<const ObjType>   constObj_;
  bool                         isConst_;
};

} // namespace Teuchos


#endif // TEUCHOS_CONST_NONCONST_OBJECT_CONTAINER_HPP
