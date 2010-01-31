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

#ifndef TEUCHOS_RCP_DECL_HPP
#define TEUCHOS_RCP_DECL_HPP

/*! \file Teuchos_RCPDecl.hpp
    \brief Reference-counted pointer class and non-member templated function implementations.
*/

#include "Teuchos_any.hpp"

#ifdef REFCOUNTPTR_INLINE_FUNCS
#  define REFCOUNTPTR_INLINE inline
#else
#  define REFCOUNTPTR_INLINE
#endif

#ifdef TEUCHOS_DEBUG
#  define TEUCHOS_REFCOUNTPTR_ASSERT_NONNULL
#endif

namespace Teuchos {

namespace PrivateUtilityPack {
  class RCP_node;
}

/** \brief Used to initialize a <tt>RCP</tt> object to NULL using an
 * implicit conversion!
 *
 * \relates RCP
 */
enum ENull { null };

/** \brief Used to specify a pre or post destruction of extra data
 *
 * \relates RCP
 */
enum EPrePostDestruction { PRE_DESTROY, POST_DESTROY };

/** \brief Smart reference counting pointer class for automatic garbage
  collection.
  
For a carefully written discussion about what this class is and basic
details on how to use it see the
<A HREF="http://software.sandia.gov/Trilinos/RCPBeginnersGuideSAND.pdf">beginners guide</A>.

<b>Quickstart for <tt>RCP</tt></b>
 
Here we present a short, but fairly comprehensive, quick-start for the
use of <tt>RCP<></tt>.  The use cases described here
should cover the overwhelming majority of the use instances of
<tt>RCP<></tt> in a typical program.

The following class hierarchy will be used in the C++ examples given
below.

\code

class A { public: virtual ~A(){} virtual void f(){} };
class B1 : virtual public A {};
class B2 : virtual public A {};
class C : virtual public B1, virtual public B2 {};

class D {};
class E : public D {};

\endcode

All of the following code examples used in this quickstart are assumed to be
in the namespace <tt>Teuchos</tt> or have appropriate <tt>using
Teuchos::...</tt> declarations.  This removes the need to explicitly use
<tt>Teuchos::</tt> to qualify classes, functions and other declarations from
the <tt>Teuchos</tt> namespace.  Note that some of the runtime checks are
denoted as "debug runtime checked" which means that checking will only be
performed in a debug build (that is one where the macro
TEUCHOS_REFCOUNTPTR_ASSERT_NONNULL, or TEUCHOS_DEBUG is defined at compile time).

<ol>

<li> <b>Creation of <tt>RCP<></tt> objects</b>

<ol>

<li> <b>Creating a <tt>RCP<></tt> object using <tt>new</tt></b>

\code
RCP<C> c_ptr = rcp(new C);
\endcode

<li> <b>Creating a <tt>RCP<></tt> object to an array allocated using <tt>new[n]</tt></b> : <tt>Teuchos::DeallocArrayDelete</tt>

\code

RCP<C> c_ptr = rcp(new C[n],DeallocArrayDelete<C>(),true);
\endcode

<li> <b>Creating a <tt>RCP<></tt> object equipped with a specialized deallocator function</b> : <tt>Teuchos::DeallocFunctorDelete</tt>

\code

void someDeallocFunction(C* c_ptr);

RCP<C> c_ptr = rcp(new deallocFunctorDelete<C>(someDeallocFunction),true);
\endcode

<li> <b>Initializing a <tt>RCP<></tt> object to NULL</b>

\code
RCP<C> c_ptr;
\endcode
or
\code
RCP<C> c_ptr = null;
\endcode

<li> <b>Initializing a <tt>RCP<></tt> object to an object
       \underline{not} allocated with <tt>new</tt></b>

\code
C              c;
RCP<C> c_ptr = rcp(&c,false);
\endcode

<li> <b>Copy constructor (implicit casting)</b>

\code
RCP<C>       c_ptr  = rcp(new C); // No cast
RCP<A>       a_ptr  = c_ptr;      // Cast to base class
RCP<const A> ca_ptr = a_ptr;      // Cast from non-const to const
\endcode

<li> <b>Representing constantness and non-constantness</b>

<ol>

<li> <b>Non-constant pointer to non-constant object</b>
\code
RCP<C> c_ptr;
\endcode

<li> <b>Constant pointer to non-constant object</b>
\code
const RCP<C> c_ptr;
\endcode

<li> <b>Non-Constant pointer to constant object</b>
\code
RCP<const C> c_ptr;
\endcode

<li> <b>Constant pointer to constant object</b>
\code
const RCP<const C> c_ptr;
\endcode

</ol>

</ol>

<li> <b>Reinitialization of <tt>RCP<></tt> objects (using assignment operator)</b>

<ol>

<li> <b>Resetting from a raw pointer</b>

\code
RCP<A> a_ptr;
a_ptr = rcp(new C());
\endcode

<li> <b>Resetting to null</b>

\code
RCP<A> a_ptr = rcp(new C());
a_ptr = null; // The C object will be deleted here
\endcode

<li> <b>Assigning from a <tt>RCP<></tt> object</b>

\code
RCP<A> a_ptr1;
RCP<A> a_ptr2 = rcp(new C());
a_ptr1 = a_ptr2; // Now a_ptr1 and a_ptr2 point to same C object
\endcode

</ol>

<li> <b>Accessing the reference-counted object</b>

<ol>

<li> <b>Access to object reference (debug runtime checked)</b> : <tt>Teuchos::RCP::operator*()</tt> 

\code
C &c_ref = *c_ptr;
\endcode

<li> <b>Access to object pointer (unchecked, may return <tt>NULL</tt>)</b> : <tt>Teuchos::RCP::get()</tt>

\code
C *c_rptr = c_ptr.get();
\endcode

<li> <b>Access to object pointer (debug runtime checked, will not return <tt>NULL</tt>)</b> : <tt>Teuchos::RCP::operator*()</tt>

\code
C *c_rptr = &*c_ptr;
\endcode

<li> <b>Access of object's member (debug runtime checked)</b> : <tt>Teuchos::RCP::operator->()</tt>

\code
c_ptr->f();
\endcode

<li> <b>Testing for non-null</b> : <tt>Teuchos::RCP::get()</tt>, <tt>Teuchos::operator==()</tt>, <tt>Teuchos::operator!=()</tt>

\code
if( a_ptr.get() ) std::cout << "a_ptr is not null!\n";
\endcode

or

\code
if( a_ptr != null ) std::cout << "a_ptr is not null!\n";
\endcode

or

<li> <b>Testing for null</b>

\code
if( !a_ptr.get() ) std::cout << "a_ptr is null!\n";
\endcode

or

\code
if( a_ptr == null ) std::cout << "a_ptr is null!\n";
\endcode

or

\code
if( is_null(a_ptr) ) std::cout << "a_ptr is null!\n";
\endcode

</ol>

<li> <b>Casting</b>

<ol>

<li> <b>Implicit casting (see copy constructor above)</b>

<ol>

<li> <b>Using copy constructor (see above)</b>

<li> <b>Using conversion function</b>

\code
RCP<C>       c_ptr  = rcp(new C);                       // No cast
RCP<A>       a_ptr  = rcp_implicit_cast<A>(c_ptr);      // To base
RCP<const A> ca_ptr = rcp_implicit_cast<const A>(a_ptr);// To const
\endcode

</ol>

<li> <b>Casting away <tt>const</tt></b> : <tt>rcp_const_cast()</tt>

\code
RCP<const A>  ca_ptr = rcp(new C);
RCP<A>        a_ptr  = rcp_const_cast<A>(ca_ptr); // cast away const!
\endcode

<li> <b>Static cast (no runtime check)</b> : <tt>rcp_static_cast()</tt>

\code
RCP<D>     d_ptr = rcp(new E);
RCP<E>     e_ptr = rcp_static_cast<E>(d_ptr); // Unchecked, unsafe?
\endcode

<li> <b>Dynamic cast (runtime checked, failed cast allowed)</b> : <tt>rcp_dynamic_cast()</tt>

\code
RCP<A>     a_ptr  = rcp(new C);
RCP<B1>    b1_ptr = rcp_dynamic_cast<B1>(a_ptr);  // Checked, safe!
RCP<B2>    b2_ptr = rcp_dynamic_cast<B2>(b1_ptr); // Checked, safe!
RCP<C>     c_ptr  = rcp_dynamic_cast<C>(b2_ptr);  // Checked, safe!
\endcode

<li> <b>Dynamic cast (runtime checked, failed cast not allowed)</b> : <tt>rcp_dynamic_cast()</tt>

\code
RCP<A>     a_ptr1  = rcp(new C);
RCP<A>     a_ptr2  = rcp(new A);
RCP<B1>    b1_ptr1 = rcp_dynamic_cast<B1>(a_ptr1,true);  // Success!
RCP<B1>    b1_ptr2 = rcp_dynamic_cast<B1>(a_ptr2,true);  // Throw std::bad_cast!
\endcode

</ol>


<li> <b>Customized deallocators</b>

<ol>

<li> <b>Creating a <tt>RCP<></tt> object with a custom deallocator</b> : <tt>Teuchos::DeallocArrayDelete</tt>

\code
RCP<C> c_ptr = rcp(new C[N],DeallocArrayDelete<C>(),true);
\endcode

<li> <b>Access customized deallocator (runtime checked, throws on failure)</b> : <tt>Teuchos::get_dealloc()</tt>

\code
const DeallocArrayDelete<C>
  &dealloc = get_dealloc<DeallocArrayDelete<C> >(c_ptr);
\endcode

<li> <b>Access optional customized deallocator</b> : <tt>Teuchos::get_optional_dealloc()</tt>

\code
const DeallocArrayDelete<C>
  *dealloc = get_optional_dealloc<DeallocArrayDelete<C> >(c_ptr);
if(dealloc) std::cout << "This deallocator exits!\n";
\endcode

</ol>

<li> <b>Managing extra data</b>

<ol>

<li> <b>Adding extra data (post destruction of extra data)</b> : <tt>Teuchos::set_extra_data()</tt>

\code
set_extra_data(rcp(new B1),"A:B1",&a_ptr);
\endcode

<li> <b>Adding extra data (pre destruction of extra data)</b> : <tt>Teuchos::get_extra_data()</tt>

\code
set_extra_data(rcp(new B1),"A:B1",&a_ptr,PRE_DESTORY);
\endcode

<li> <b>Retrieving extra data</b> : <tt>Teuchos::get_extra_data()</tt>

\code
get_extra_data<RCP<B1> >(a_ptr,"A:B1")->f();
\endcode

<li> <b>Resetting extra data</b> : <tt>Teuchos::get_extra_data()</tt>

\code
get_extra_data<RCP<B1> >(a_ptr,"A:B1") = rcp(new C);
\endcode

<li> <b>Retrieving optional extra data</b> : <tt>Teuchos::get_optional_extra_data()</tt>

\code
const RCP<B1>
  *b1 = get_optional_extra_data<RCP<B1> >(a_ptr,"A:B1");
if(b1) (*b1)->f();
\endcode

</ol>

</ol>

\ingroup teuchos_mem_mng_grp

 */
template<class T>
class RCP {
public:
  /** \brief . */
  typedef T  element_type;
  /** \brief Initialize <tt>RCP<T></tt> to NULL.
   *
   * This allows clients to write code like:
   \code
   RCP<int> p = null;
   \endcode
   or
   \code
   RCP<int> p;
   \endcode
   * and construct to <tt>NULL</tt>
   */
  RCP( ENull null_arg = null );
  /** \brief Construct from a raw pointer.
   *
   * Note that this constructor is declared explicit so there is no implicit
   * conversion from a raw pointer to an RCP allowed.  If
   * <tt>has_ownership==false</tt>, then no attempt to delete the object will
   * occur.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == p</tt>
   * <li> <tt>this->count() == 1</tt>
   * <li> <tt>this->has_ownership() == has_ownership</tt>
   * </ul>
   */
  explicit RCP( T* p, bool has_ownership = false );
  /** \brief Initialize from another <tt>RCP<T></tt> object.
   *
   * After construction, <tt>this</tt> and <tt>r_ptr</tt> will
   * reference the same object.
   *
   * This form of the copy constructor is required even though the
   * below more general templated version is sufficient since some
   * compilers will generate this function automatically which will
   * give an incorrect implementation.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->count() == r_ptr.count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.count()</tt> is incremented by 1
   * </ul>
   */
  RCP(const RCP<T>& r_ptr);
  /** \brief Initialize from another <tt>RCP<T2></tt> object (implicit conversion only).
   *
   * This function allows the implicit conversion of smart pointer objects just
   * like with raw C++ pointers.  Note that this function will only compile
   * if the statement <tt>T1 *ptr = r_ptr.get()</tt> will compile.
   *
   * <b>Postconditons:</b> <ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->count() == r_ptr.count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.count()</tt> is incremented by 1
   * </ul>
   */
  template<class T2>
  RCP(const RCP<T2>& r_ptr);
  /** \brief Removes a reference to a dynamically allocated object and possibly deletes
   * the object if owned.
   *
   * Deletes the object if <tt>this->has_ownership() == true</tt> and
   * <tt>this->count() == 1</tt>.  If <tt>this->count() == 1</tt> but
   * <tt>this->has_ownership() == false</tt> then the object is not deleted.
   * If <tt>this->count() > 1</tt> then the internal reference count shared by
   * all the other related <tt>RCP<...></tt> objects for this shared
   * object is deincremented by one.  If <tt>this->get() == NULL</tt> then
   * nothing happens.
   */
  ~RCP();
  /** \brief Copy the pointer to the referenced object and increment the
   * reference count.
   *
   * If <tt>this->has_ownership() == true</tt> and <tt>this->count() == 1</tt>
   * before this operation is called, then the object pointed to by
   * <tt>this->get()</tt> will be deleted (usually using <tt>delete</tt>)
   * prior to binding to the pointer (possibly <tt>NULL</tt>) pointed to in
   * <tt>r_ptr</tt>.  Assignment to self (i.e. <tt>this->get() ==
   * r_ptr.get()</tt>) is harmless and this function does nothing.
   *
   * <b>Postconditons:</b><ul>
   * <li> <tt>this->get() == r_ptr.get()</tt>
   * <li> <tt>this->count() == r_ptr.count()</tt>
   * <li> <tt>this->has_ownership() == r_ptr.has_ownership()</tt>
   * <li> If <tt>r_ptr.get() != NULL</tt> then <tt>r_ptr.count()</tt> is incremented by 1
   * </ul>
   */
  RCP<T>& operator=(const RCP<T>& r_ptr);
  /** \brief Pointer (<tt>-></tt>) access to members of underlying object.
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>this->get() != NULL</tt> (throws <tt>std::logic_error</tt>)
   * </ul>
   */
  T* operator->() const;
  /** \brief Dereference the underlying object.
   *
   * <b>Preconditions:</b><ul>
   * <li> <tt>this->get() != NULL</tt> (throws <tt>std::logic_error</tt>)
   * </ul>
   */
  T& operator*() const;
    /** \brief Get the raw C++ pointer to the underlying object.
   */
  T* get() const;
  /** \brief Release the ownership of the underlying dynamically allocated object.
   *
   * After this function is called then the client is responsible for
   * deallocating the shared object no matter how many
   * <tt>ref_count_prt<T></tt> objects have a reference to it.  If
   * <tt>this-></tt>get()<tt>== NULL</tt>, then this call is meaningless.
   *
   * Note that this function does not have the exact same semantics as does
   * <tt>auto_ptr<T>::release()</tt>.  In <tt>auto_ptr<T>::release()</tt>,
   * <tt>this</tt> is set to <tt>NULL</tt> while here in RCP<T>::
   * release() only an ownership flag is set and <tt>*this</tt> still points
   * to the same object.  It would be difficult to duplicate the behavior of
   * <tt>auto_ptr<T>::release()</tt> for this class.
   *
   * <b>Postconditions:</b>
   * <ul>
   * <li> <tt>this->has_ownership() == false</tt>
   * </ul>
   *
   * @return Returns the value of <tt>this->get()</tt>
   */
  T* release();
  /** \brief Return the number of <tt>RCP<></tt> objects that have a reference
   * to the underlying pointer that is being shared.
   *
   * @return  If <tt>this->get() == NULL</tt> then this function returns 0.
   * Otherwise, this function returns <tt>0</tt>.
   */
  int count() const;
  /** \brief Give <tt>this</tt> and other <tt>RCP<></tt> objects ownership 
   * of the referenced object <tt>this->get()</tt>.
   *
   * See ~RCP() above.  This function
   * does nothing if <tt>this->get() == NULL</tt>.
   *
   * <b>Postconditions:</b>
   * <ul>
   * <li> If <tt>this->get() == NULL</tt> then
   *   <ul>
   *   <li> <tt>this->has_ownership() == false</tt> (always!).
   *   </ul>
   * <li> else
   *   <ul>
   *   <li> <tt>this->has_ownership() == true</tt>
   *   </ul>
   * </ul>
   */
  void set_has_ownership();
  /** \brief Returns true if <tt>this</tt> has ownership of object pointed to by <tt>this->get()</tt> in order to delete it.
   *
   * See ~RCP() above.
   *
   * @return If this->get() <tt>== NULL</tt> then this function always returns <tt>false</tt>.
   * Otherwise the value returned from this function depends on which function was
   * called most recently, if any; set_has_ownership() (<tt>true</tt>)
   * or release() (<tt>false</tt>).
   */
  bool has_ownership() const;
  /** \brief Returns true if the smart pointers share the same underlying reference-counted object.
   *
   * This method does more than just check if <tt>this->get() == r_ptr.get()</tt>.
   * It also checks to see if the underlying reference counting machinary is the
   * same.
   */
  template<class T2>
  bool shares_resource(const RCP<T2>& r_ptr) const;
  /** \brief Throws <tt>std::logic_error</tt> if <tt>this->get()==NULL</tt>, otherwise returns reference to <tt>*this</tt>. */
  const RCP<T>& assert_not_null() const;

public: // Bad bad bad

  // //////////////////////////////////////
  // Private types

  typedef PrivateUtilityPack::RCP_node      node_t;

private:

  // //////////////////////////////////////////////////////////////
  // Private data members

  T       *ptr_;  // NULL if this pointer is null
  node_t  *node_;  // NULL if this pointer is null

public: // Bad bad bad
#ifndef DOXYGEN_COMPILE
  // These constructors should be private but I have not had good luck making
  // this portable (i.e. using friendship etc.) in the past
  template<class Dealloc_T>
  RCP( T* p, Dealloc_T dealloc, bool has_ownership );
  // This is a very bad breach of encapsulation that is needed since MS VC++ 5.0 will
  // not allow me to declare template functions as friends.
  RCP( T* p, node_t* node);
  T*& access_ptr();
  node_t*& access_node();
  node_t* access_node() const;
#endif

};  // end class RCP<...>

/** \brief Traits specialization.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<typename T>
class TypeNameTraits<RCP<T> > {
public:
  static std::string name() { return "RCP<"+TypeNameTraits<T>::name()+">"; }
};


/** \brief Policy class for deallocator that uses <tt>delete</tt> to delete a
 * pointer which is used by <tt>RCP</tt>.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T>
class DeallocDelete
{
public:
  /// Gives the type (required)
  typedef T ptr_t;
  /// Deallocates a pointer <tt>ptr</tt> using <tt>delete ptr</tt> (required).
  void free( T* ptr ) { if(ptr) delete ptr; }
};

/** \brief Deallocator class that uses <tt>delete []</tt> to delete memory
 * allocated uisng <tt>new []</tt>
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T>
class DeallocArrayDelete
{
public:
  /// Gives the type (required)
  typedef T ptr_t;
  /// Deallocates a pointer <tt>ptr</tt> using <tt>delete [] ptr</tt> (required).
  void free( T* ptr ) { if(ptr) delete [] ptr; }
};

/** \brief Deallocator subclass that Allows any functor object (including a
 * function pointer) to be used to free an object.
 *
 * Note, the only requirement is that deleteFuctor(ptr) can be called (which
 * is true for a function pointer).
 *
 * Note, a client should generally use the function
 * <tt>deallocFunctorDelete()</tt> to create this object and not try to
 * construct it directly.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T, class DeleteFunctor>
class DeallocFunctorDelete
{
public:
  DeallocFunctorDelete( DeleteFunctor deleteFunctor ) : deleteFunctor_(deleteFunctor) {}
  typedef T ptr_t;
  void free( T* ptr ) { if(ptr) deleteFunctor_(ptr); }
private:
  DeleteFunctor deleteFunctor_;
  DeallocFunctorDelete(); // Not defined and not to be called!
};

/** \brief A simple function used to create a functor deallocator object.
 *
 * \relates DeallocFunctorDelete
 */
template<class T, class DeleteFunctor>
DeallocFunctorDelete<T,DeleteFunctor>
deallocFunctorDelete( DeleteFunctor deleteFunctor )
{
  return DeallocFunctorDelete<T,DeleteFunctor>(deleteFunctor);
}

/** \brief Deallocator subclass that Allows any functor object (including a
 * function pointer) to be used to free a handle (i.e. pointer to pointer) to
 * an object.
 *
 * Note, the only requirement is that deleteFuctor(ptrptr) can be called
 * (which is true for a function pointer).
 *
 * Note, a client should generally use the function
 * <tt>deallocFunctorDelete()</tt> to create this object and not try to
 * construct it directly.
 *
 * \ingroup teuchos_mem_mng_grp
 */
template<class T, class DeleteHandleFunctor>
class DeallocFunctorHandleDelete
{
public:
  DeallocFunctorHandleDelete( DeleteHandleFunctor deleteHandleFunctor )
    : deleteHandleFunctor_(deleteHandleFunctor) {}
  typedef T ptr_t;
  void free( T* ptr ) { if(ptr) { T **hdl = &ptr; deleteHandleFunctor_(hdl); } }
private:
  DeleteHandleFunctor deleteHandleFunctor_;
  DeallocFunctorHandleDelete(); // Not defined and not to be called!
};

/** \brief A simple function used to create a functor deallocator object.
 *
 * \relates DeallocFunctorHandleDelete
 */
template<class T, class DeleteHandleFunctor>
DeallocFunctorHandleDelete<T,DeleteHandleFunctor>
deallocFunctorHandleDelete( DeleteHandleFunctor deleteHandleFunctor )
{
  return DeallocFunctorHandleDelete<T,DeleteHandleFunctor>(deleteHandleFunctor);
}

/** \brief Create a <tt>RCP</tt> object properly typed.
 *
 * @param  p  [in] Pointer to an object to be reference counted.
 * @param owns_mem
 *            [in] If <tt>owns_mem==true</tt>  then <tt>delete p</tt>
 *            will be called when the last reference to this object
 *            is removed.  If <tt>owns_mem==false</tt> then nothing
 *            will happen to delete the the object pointed to by
 *            <tt>p</tt> when the last reference is removed.
 *
 * <b>Preconditions:</b><ul>
 * <li> If <tt>owns_mem==true</tt> then <tt>p</tt> must have been
 *      created by calling <tt>new</tt> to create the object since
 *      <tt>delete p</tt> will be called eventually.
 * </ul>
 *
 * If the pointer <tt>p</tt> did not come from <tt>new</tt> then
 * either the client should use the version of <tt>rcp()</tt> that
 * that uses a deallocator policy object or should pass in 
 * <tt>owns_mem = false</tt>.
 *
 * \relates RCP
 */
template<class T>
RCP<T> rcp( T* p, bool owns_mem
#ifndef __sun
  = true
#endif
  );
#ifdef __sun // RAB: 20040303: Sun needs to fix their compiler
template<class T> inline RCP<T> rcp( T* p ) { return rcp(p,true); }
#endif

/** \brief Initialize from a raw pointer with a deallocation policy.
 *
 * @param  p       [in] Raw C++ pointer that \c this will represent.
 * @param  dealloc [in] Deallocator policy object (copied by value) that defines
 *                 a function <tt>void Dealloc_T::free(T* p)</tt> that will
 *                 free the underlying object.
 * @param  owns_mem
 *                 [in] If true then <tt>return</tt> is allowed to delete
 *                 the underlying pointer by calling <tt>dealloc.free(p)</tt>.
 *                 when all references have been removed.
 *
 * <b>Preconditions:</b><ul>
 * <li> The function <tt>void Dealloc_T::free(T* p)</tt> exists.
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> <tt>return.get() == p</tt>
 * <li> If <tt>p == NULL</tt> then
 *   <ul>
 *   <li> <tt>return.count() == 0</tt>
 *   <li> <tt>return.has_ownership() == false</tt>
 *   </ul>
 * <li> else
 *   <ul>
 *   <li> <tt>return.count() == 1</tt>
 *   <li> <tt>return.has_ownership() == owns_mem</tt>
 *   </ul>
 * </ul>
 *
 * By default, <tt>return</tt> has ownership to delete the object
 * pointed to by <tt>p</tt> when <tt>return</tt> is deleted (see
 * <tt>~RCP())</tt>.  If <tt>owns_mem==true</tt>, it is vital
 * that the address <tt>p</tt>
 * passed in is the same address that was returned by <tt>new</tt>.
 * With multiple inheritance this is not always the case.  See the
 * above discussion.  This class is templated to accept a deallocator
 * object that will free the pointer.  The other functions use a
 * default deallocator of type <tt>DeallocDelete</tt> which has a method
 * <tt>DeallocDelete::free()</tt> which just calls <tt>delete p</tt>.
 *
 * \relates RCP
 */
template<class T, class Dealloc_T>
RCP<T> rcp( T* p, Dealloc_T dealloc, bool owns_mem );

/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates RCP
 */
template<class T>
bool is_null( const RCP<T> &p );

/** \brief Returns true if <tt>p.get()==NULL</tt>.
 *
 * \relates RCP
 */
template<class T>
bool operator==( const RCP<T> &p, ENull );

/** \brief Returns true if <tt>p.get()!=NULL</tt>.
 *
 * \relates RCP
 */
template<class T>
bool operator!=( const RCP<T> &p, ENull );

/** \brief Return true if two <tt>RCP</tt> objects point to the same
 * referenced-counted object and have the same node.
 *
 * \relates RCP
 */
template<class T1, class T2>
bool operator==( const RCP<T1> &p1, const RCP<T2> &p2 );

/** \brief Return true if two <tt>RCP</tt> objects do not point to the
 * same referenced-counted object and have the same node.
 *
 * \relates RCP
 */
template<class T1, class T2>
bool operator!=( const RCP<T1> &p1, const RCP<T2> &p2 );

/** \brief Implicit cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>T2* p2 = p1.get();</tt>) compiles.
 *
 * This is to be used for conversions up an inheritance hierarchy and from non-const to
 * const and any other standard implicit pointer conversions allowed by C++.
 *
 * \relates RCP
 */
template<class T2, class T1>
RCP<T2> rcp_implicit_cast(const RCP<T1>& p1);

/** \brief Static cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * The function will compile only if (<tt>static_cast<T2*>(p1.get());</tt>) compiles.
 *
 * This can safely be used for conversion down an inheritance hierarchy
 * with polymorphic types only if <tt>dynamic_cast<T2>(p1.get()) == static_cast<T2>(p1.get())</tt>.
 * If not then you have to use <tt>rcp_dynamic_cast<tt><T2>(p1)</tt>.
 *
 * \relates RCP
 */
template<class T2, class T1>
RCP<T2> rcp_static_cast(const RCP<T1>& p1);

/** \brief Constant cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * This function will compile only if (<tt>const_cast<T2*>(p1.get());</tt>) compiles.
 *
 * \relates RCP
 */
template<class T2, class T1>
RCP<T2> rcp_const_cast(const RCP<T1>& p1);

/** \brief Dynamic cast of underlying <tt>RCP</tt> type from <tt>T1*</tt> to <tt>T2*</tt>.
 *
 * @param  p1             [in] The smart pointer casting from
 * @param  throw_on_fail  [in] If <tt>true</tt> then if the cast fails (for <tt>p1.get()!=NULL) then
 *                        a <tt>std::bad_cast</tt> std::exception is thrown with a very informative
 *                        error message.
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>( p1.get()!=NULL && throw_on_fail==true && dynamic_cast<T2*>(p1.get())==NULL ) == true</tt>
 *      then an <tt>std::bad_cast</tt> std::exception is thrown with a very informative error message.
 * <li> If <tt>( p1.get()!=NULL && dynamic_cast<T2*>(p1.get())!=NULL ) == true</tt>
 *      then <tt>return.get() == dynamic_cast<T2*>(p1.get())</tt>.
 * <li> If <tt>( p1.get()!=NULL && throw_on_fail==false && dynamic_cast<T2*>(p1.get())==NULL ) == true</tt>
 *      then <tt>return.get() == NULL</tt>.
 * <li> If <tt>( p1.get()==NULL ) == true</tt>
 *      then <tt>return.get() == NULL</tt>.
 * </ul>
 *
 * This function will compile only if (<tt>dynamic_cast<T2*>(p1.get());</tt>) compiles.
 *
 * \relates RCP
 */
template<class T2, class T1>
RCP<T2> rcp_dynamic_cast(
  const RCP<T1>& p1
  ,bool throw_on_fail
#ifndef __sun
  = false
#endif
  );
#ifdef __sun // RAB: 20041019: Sun needs to fix their compiler
template<class T2, class T1> inline RCP<T2> rcp_dynamic_cast( const RCP<T1>& p1 )
{ return rcp_dynamic_cast<T2>(p1,false); }
#endif

/** \brief Set extra data associated with a <tt>RCP</tt> object.
 *
 * @param  extra_data
 *               [in] Data object that will be set (copied)
 * @param  name  [in] The name given to the extra data.  The value of
 *               <tt>name</tt> together with the data type <tt>T1</tt> of the
 *               extra data must be unique from any other such data or
 *               the other data will be overwritten.
 * @param  p     [out] On output, will be updated with the input <tt>extra_data</tt>
 * @param  destroy_when
 *               [in] Determines when <tt>extra_data</tt> will be destoryed
 *               in relation to the underlying reference-counted object.
 *               If <tt>destroy_when==PRE_DESTROY</tt> then <tt>extra_data</tt>
 *               will be deleted before the underlying reference-counted object.
 *               If <tt>destroy_when==POST_DESTROY</tt> (the default) then <tt>extra_data</tt>
 *               will be deleted after the underlying reference-counted object.
 * @param  force_unique
 *               [in] Determines if this type and name pair must be unique
 *               in which case if an object with this same type and name
 *               already exists, then an std::exception will be thrown.
 *               The default is <tt>true</tt> for safety.
 *
 * If there is a call to this function with the same type of extra
 * data <tt>T1</tt> and same arguments <tt>p</tt> and <tt>name</tt>
 * has already been made, then the current piece of extra data already
 * set will be overwritten with <tt>extra_data</tt>.  However, if the
 * type of the extra data <tt>T1</tt> is different, then the extra
 * data can be added and not overwrite existing extra data.  This
 * means that extra data is keyed on both the type and name.  This
 * helps to minimize the chance that clients will unexpectedly
 * overwrite data by accident.
 *
 * When the last <tt>RefcountPtr</tt> object is removed and the
 * reference-count node is deleted, then objects are deleted in the following
 * order: (1) All of the extra data that where added with
 * <tt>destroy_when==PRE_DESTROY</tt> are first, (2) then the underlying
 * reference-counted object is deleted, and (3) the rest of the extra data
 * that was added with <tt>destroy_when==PRE_DESTROY</tt> is then deleted.
 * The order in which the objects are destroyed is not guaranteed.  Therefore,
 * clients should be careful not to add extra data that has deletion
 * dependancies (instead consider using nested RCP objects as extra
 * data which will guarantee the order of deletion).
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p->get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * <li> If this function has already been called with the same template
 *      type <tt>T1</tt> for <tt>extra_data</tt> and the same std::string <tt>name</tt>
 *      and <tt>force_unique==true</tt>, then an <tt>std::invalid_argument</tt>
 *      std::exception will be thrown.
 * </ul>
 *
 * Note, this function is made a non-member function to be consistent
 * with the non-member <tt>get_extra_data()</tt> functions.
 *
 * \relates RCP
 */
template<class T1, class T2>
void set_extra_data(
  const T1 &extra_data,
  const std::string& name, RCP<T2> *p,
  EPrePostDestruction destroy_when = POST_DESTROY,
  bool force_unique = true
  );

/** \brief Get a non-const reference to extra data associated with a <tt>RCP</tt> object.
 *
 * @param  p    [in] Smart pointer object that extra data is being extraced from.
 * @param  name [in] Name of the extra data.
 *
 * @return Returns a non-const reference to the extra_data object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * <li> <tt>name</tt> and <tt>T1</tt> must have been used in a previous
 *      call to <tt>set_extra_data()</tt> (throws <tt>std::invalid_argument</tt>).
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * \relates RCP
 */
template<class T1, class T2>
T1& get_extra_data( RCP<T2>& p, const std::string& name );

/** \brief Get a const reference to extra data associated with a <tt>RCP</tt> object.
 *
 * @param  p    [in] Smart pointer object that extra data is being extraced from.
 * @param  name [in] Name of the extra data.
 *
 * @return Returns a const reference to the extra_data object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * <li> <tt>name</tt> and <tt>T1</tt> must have been used in a previous
 *      call to <tt>set_extra_data()</tt> (throws <tt>std::invalid_argument</tt>).
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * Also note that this const version is a false sense of security
 * since a client can always copy a const <tt>RCP</tt> object
 * into a non-const object and then use the non-const version to
 * change the data.  However, its presence will help to avoid some
 * types of accidental changes to this extra data.
 *
 * \relates RCP
 */
template<class T1, class T2>
const T1& get_extra_data( const RCP<T2>& p, const std::string& name );

/** \brief Get a pointer to non-const extra data (if it exists) associated
 * with a <tt>RCP</tt> object.
 *
 * @param  p    [in] Smart pointer object that extra data is being extraced from.
 * @param  name [in] Name of the extra data.
 *
 * @return Returns a non-const pointer to the extra_data object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>name</tt> and <tt>T1</tt> have been used in a previous
 *      call to <tt>set_extra_data()</tt> then <tt>return !=NULL</tt>
 *      and otherwise <tt>return == NULL</tt>.
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * \relates RCP
 */
template<class T1, class T2>
T1* get_optional_extra_data( RCP<T2>& p, const std::string& name );

/** \brief Get a pointer to const extra data (if it exists) associated with a <tt>RCP</tt> object.
 *
 * @param  p    [in] Smart pointer object that extra data is being extraced from.
 * @param  name [in] Name of the extra data.
 *
 * @return Returns a const pointer to the extra_data object if it exists.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If <tt>name</tt> and <tt>T1</tt> have been used in a previous
 *      call to <tt>set_extra_data()</tt> then <tt>return !=NULL</tt>
 *      and otherwise <tt>return == NULL</tt>.
 * </ul>
 *
 * Note, this function must be a non-member function since the client
 * must manually select the first template argument.
 *
 * Also note that this const version is a false sense of security
 * since a client can always copy a const <tt>RCP</tt> object
 * into a non-const object and then use the non-const version to
 * change the data.  However, its presence will help to avoid some
 * types of accidental changes to this extra data.
 *
 * \relates RCP
 */
template<class T1, class T2>
const T1* get_optional_extra_data( const RCP<T2>& p, const std::string& name );

/** \brief Return a non-<tt>const</tt> reference to the underlying deallocator object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * <li> The deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      (throws <tt>std::logic_error</tt>)
 * </ul>
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
Dealloc_T& get_dealloc( RCP<T>& p );

/** \brief Return a <tt>const</tt> reference to the underlying deallocator object.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * <li> The deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      (throws <tt>std::logic_error</tt>)
 * </ul>
 *
 * Note that the <tt>const</tt> version of this function provides only
 * a very ineffective attempt to avoid accidental changes to the
 * deallocation object.  A client can always just create a new
 * non-<tt>const</tt> <tt>RCP<T></tt> object from any
 * <tt>const</tt> <tt>RCP<T></tt> object and then call the
 * non-<tt>const</tt> version of this function.
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
const Dealloc_T& get_dealloc( const RCP<T>& p );

/** \brief Return a pointer to the underlying non-<tt>const</tt> deallocator
 * object if it exists.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If the deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      then <tt>return!=NULL</tt>, otherwise <tt>return==NULL</tt>
 * </ul>
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
Dealloc_T* get_optional_dealloc( RCP<T>& p );

/** \brief Return a pointer to the underlying <tt>const</tt> deallocator
 * object if it exists.
 *
 * <b>Preconditions:</b><ul>
 * <li> <tt>p.get() != NULL</tt> (throws <tt>std::logic_error</tt>)
 * </ul>
 *
 * <b>Postconditions:</b><ul>
 * <li> If the deallocator object type used to construct <tt>p</tt> is same as <tt>Dealloc_T</tt>
 *      then <tt>return!=NULL</tt>, otherwise <tt>return==NULL</tt>
 * </ul>
 *
 * Note that the <tt>const</tt> version of this function provides only
 * a very ineffective attempt to avoid accidental changes to the
 * deallocation object.  A client can always just create a new
 * non-<tt>const</tt> <tt>RCP<T></tt> object from any
 * <tt>const</tt> <tt>RCP<T></tt> object and then call the
 * non-<tt>const</tt> version of this function.
 *
 * \relates RCP
 */
template<class Dealloc_T, class T>
const Dealloc_T* get_optional_dealloc( const RCP<T>& p );

/** \brief Output stream inserter.
 *
 * The implementation of this function just print pointer addresses and
 * therefore puts not restrictions on the data types involved.
 *
 * \relates RCP
 */
template<class T>
std::ostream& operator<<( std::ostream& out, const RCP<T>& p );

/** \brief Print the list of currently active RCP nodes.
 *
 * When the macro <tt>TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE</tt> is
 * defined, this function will print out all of the RCP nodes that are
 * currently active.  This function can be called at any time during a
 * program.
 *
 * When the macro <tt>TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODE_TRACE</tt> is
 * defined this function will get called automatically after the program ends
 * and all of the local and global RCP objects have been destroyed.  If any
 * RCP nodes are printed at that time, then this is an indication that there
 * may be some circular references that will caused memory leaks.  You memory
 * checking tool such as valgrind or purify should complain about this!
 *
 * \relates RCP
 */
void print_active_RCP_nodes(std::ostream &out);

} // end namespace Teuchos

#endif  // TEUCHOS_RCP_DECL_HPP
