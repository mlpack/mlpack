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

#ifndef TEUCHOS_ABSTRACT_FACTORY_STD_HPP
#define TEUCHOS_ABSTRACT_FACTORY_STD_HPP

#include "Teuchos_AbstractFactory.hpp"

namespace Teuchos {

/** \brief Default post-modification policy class for
 * <tt>AbstractFactorStd</tt> which does nothing!
 */
template<class T_impl>
class PostModNothing {
public:
	/** \brief . */
	void initialize(T_impl* p) const {} // required!
};

/** \brief Default allocation policy class for
 * <tt>AbstractFactoryStd</tt> which returns <tt>new T_impl()</tt>.
 */
template<class T_impl>
class AllocatorNew {
public:
  /** \brief . */
  typedef Teuchos::RCP<T_impl>  ptr_t;                         // required!
  /** \brief . */
  const ptr_t allocate() const { return Teuchos::rcp(new T_impl()); }  // required!
};

/** \brief Simple, templated concrete subclass of universal "Abstract
 * Factory" interface for the creation of objects.
 *
 * This concrete subclass represents a general
 * <tt>AbstractFactory</tt> subclass that can be modified through
 * policy template parameters.  This class is templated on the
 * interface type <tt>T_itfc</tt> that is exposed by the
 * <tt>AbstractFactory<T_itfc></tt> base interface and by a (concrete)
 * implementation type <tt>T_impl</tt>.  The most typical use of this
 * subclass is to use a concrete, default-constructable subclass for
 * <tt>T_impl</tt> and then to simply instantiate a concrete abstract
 * factory for that class using:
 
 \verbatim

 Teuchos::AbstractFactoryStd<T_itfc,T_impl>  abstractFactory;
 \endverbatim
 *
 * For this default usage, The only requirements for the derived classes type
 * <tt>T_impl</tt> is that it allow the default constructor
 * <tt>T_impl::T_impl()</tt> (i.e. dont make <tt>T_impl::T_impl()</tt>
 * private) and that it allow <tt>T_impl::new()</tt> and
 * <tt>T_impl::delete</tt> (i.e.  don't make them private functions, see
 * Meyers, More Effective C++, Item 27).
 
 * However, this subclass is also templated on two other policy types that
 * allow a modification on how objects are created and destroyed.  The first
 * templated policy type, <tt>T_PostMod</tt>, defines how an object is
 * modified after it is initially created.  The second templated policy type,
 * <tt>T_Allocator</tt>, defines exactly how an object is created (and
 * therefore also how it is destroyed).  These type two policy classes are
 * described in more detail below.
 *
 * The type <tt>T_PostMod</tt> is responsible for performing any post
 * modifications on a dynamically allocated object before returning it from
 * <tt>create()</tt>.  The requirements for the type <tt>T_PostMod</tt> are
 * that it has a default constructor, a copy constructor and a method
 * <tt>T_PostMod::initialize(T_itfc2*) const</tt> that will perform any
 * required post modifications (initializations).  The type <tt>T_itfc2</tt>
 * argument for this function must be a base class of <tt>T_impl</tt> of
 * course.  The default type for <tt>T_PostMod</tt> is
 * <tt>PostModNothing</tt><tt><T_impl></tt> which does nothing.
 *
 * The type <tt>T_Allocator</tt> allows for specialized memory allocation and
 * cleanup.  This type must allow the default constructor and copy constructor
 * and have a method <tt>Teuchos::RCP<T_impl> T_Allocator::allocate()
 * const</tt> which creates a smart reference-counted pointer to the allocated
 * object.  Also, in returning a <tt>RCP<></tt> object, the client can
 * set a deallocatioin policy object that can specialize the deallocation of
 * the object (see <tt>RCP</tt>).  In defining a specialized
 * <tt>T_Allocator</tt> class, the client can all initialize the object using
 * more than just the default constructor.  Therefore, if the client provides
 * a specialized <tt>T_Allocator</tt> class, there are no restrictions on the
 * class <tt>T_impl</tt> (i.e. does not have to have a default constructor or
 * allow <tt>new</tt> or <tt>delete</tt>).  The default class for
 * <tt>T_Allocator</tt> is <tt>AllocatorNew</tt><tt><T_impl></tt> who's
 * <tt>allocate()</tt> function just returns <tt>rcp(new T_impl())</tt>.
 * 
 * Since the <tt>T_Allocator</tt> class can specialize both the memory
 * management and can initialize the object using more that the default
 * constructor, the class <tt>T_PostMod</tt> may seem unecessary.  However, it
 * is more likely that the client will want to define an initialization for a
 * set of classes through an abstract interface and can not for a particular
 * concrete subclass.  Also the initialization for an object can be orthogonal
 * to how it is created and destroyed, thus the two classes <tt>T_PostMod</tt>
 * and <tt>T_Allocator</tt> are both needed for a more general implementation.
 */
template<class T_itfc, class T_impl
				 ,class T_PostMod = PostModNothing<T_impl>
				 ,class T_Allocator = AllocatorNew<T_impl>
        >
class AbstractFactoryStd : public AbstractFactory<T_itfc> {
public:

	typedef typename Teuchos::AbstractFactory<T_itfc>::obj_ptr_t   obj_ptr_t;  // RAB: 20030916: G++ 3.2 complains without this

  /** \brief . */
  AbstractFactoryStd( const T_PostMod& post_mod = T_PostMod(), const T_Allocator& alloc = T_Allocator() );

  /** @name Overriden from AbstractFactory */
  //@{
  /** \brief . */
  obj_ptr_t create() const;
  //@}

private:
  T_PostMod    post_mod_;
  T_Allocator  alloc_;

};


/** \brief Nonmember constructor for an standar abstract factory object.
 *
 * \relates AbstractFactoryStd
 */
template<class T_itfc, class T_impl>
RCP<const AbstractFactory<T_itfc> >
abstractFactoryStd()
{
	return rcp(
		new AbstractFactoryStd<T_itfc,T_impl,PostModNothing<T_impl>,AllocatorNew<T_impl> >()
		);
}


/** \brief Nonmember constructor for an standar abstract factory object.
 *
 * \relates AbstractFactoryStd
 */
template<class T_itfc, class T_impl, class T_Allocator>
RCP<const AbstractFactory<T_itfc> >
abstractFactoryStd( const T_Allocator& alloc = T_Allocator() )
{
	return rcp(
		new AbstractFactoryStd<T_itfc,T_impl,PostModNothing<T_impl>,T_Allocator>(
      PostModNothing<T_impl>(), alloc
      )
		);
}


// ///////////////////////////////////////////////////////
// Template member definitions

template<class T_itfc, class T_impl, class T_PostMod, class T_Allocator>
inline
AbstractFactoryStd<T_itfc,T_impl,T_PostMod,T_Allocator>::AbstractFactoryStd(
	const T_PostMod& post_mod, const T_Allocator& alloc
	)
	:post_mod_(post_mod)
	,alloc_(alloc)
{}

template<class T_itfc, class T_impl, class T_PostMod, class T_Allocator>
inline
typename AbstractFactoryStd<T_itfc,T_impl,T_PostMod,T_Allocator>::obj_ptr_t
AbstractFactoryStd<T_itfc,T_impl,T_PostMod,T_Allocator>::create() const
{
	typename T_Allocator::ptr_t
		ptr = alloc_.allocate();
	post_mod_.initialize(ptr.get());
	return ptr;
}

} // end Teuchos

#endif // TEUCHOS_ABSTRACT_FACTORY_STD_HPP
