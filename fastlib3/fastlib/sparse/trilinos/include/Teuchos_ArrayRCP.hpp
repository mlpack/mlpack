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

#ifndef TEUCHOS_ARRAY_RCP_HPP
#define TEUCHOS_ARRAY_RCP_HPP

#include "Teuchos_ArrayRCPDecl.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_dyn_cast.hpp"
#include "Teuchos_map.hpp"

namespace Teuchos {

// Constructors/Initializers

template<class T>
inline
ArrayRCP<T>::ArrayRCP( ENull )
  : ptr_(NULL)
  , node_(NULL)
  , lowerOffset_(0)
  , upperOffset_(-1)
{}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>::ArrayRCP(const ArrayRCP<T>& r_ptr)
  : ptr_(r_ptr.ptr_), node_(r_ptr.node_)
  , lowerOffset_(r_ptr.lowerOffset_)
  , upperOffset_(r_ptr.upperOffset_)
{
  if(node_) node_->incr_count();
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>::~ArrayRCP()
{
  if(node_ && node_->deincr_count() == 0 ) {
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODES
    printActiveRCPNodes.foo(); // Make sure this object is used!
    remove_RCP_node(node_);
#endif
    delete node_;
  }
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>& ArrayRCP<T>::operator=(const ArrayRCP<T>& r_ptr)
{
  if( this == &r_ptr )
    return *this; // Assignment to self
  if( node_ && !node_->deincr_count() ) {
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODES
    remove_RCP_node(node_);
#endif
    delete node_;
  }
  ptr_   = r_ptr.ptr_;
  node_  = r_ptr.node_;
  lowerOffset_ = r_ptr.lowerOffset_;
  upperOffset_ = r_ptr.upperOffset_;
  if(node_) node_->incr_count();
  return *this;
}

// Object/Pointer Access Functions

template<class T>
inline
T* ArrayRCP<T>::operator->() const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  assert_in_range(0,1);
#endif
  return ptr_;
}

template<class T>
inline
T& ArrayRCP<T>::operator*() const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  assert_in_range(0,1);
#endif
  return *ptr_;
}

template<class T>
inline
T* ArrayRCP<T>::get() const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  if(ptr_) {
    assert_in_range(0,1);
  }
#endif
  return ptr_;
}

template<class T>
REFCOUNTPTR_INLINE
T& ArrayRCP<T>::operator[](Ordinal offset) const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  assert_in_range(offset,1);
#endif
  return ptr_[offset];
}

// Pointer Arithmetic Functions

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>& ArrayRCP<T>::operator++()
{
  if(ptr_) {
    ++ptr_;
    --lowerOffset_;
    --upperOffset_;
  }
  return *this;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T> ArrayRCP<T>::operator++(int)
{
  ArrayRCP<T> r_ptr = *this;
  ++(*this);
  return r_ptr;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>& ArrayRCP<T>::operator--()
{
  if(ptr_) {
    --ptr_;
    ++lowerOffset_;
    ++upperOffset_;
  }
  return *this;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T> ArrayRCP<T>::operator--(int)
{
  ArrayRCP<T> r_ptr = *this;
  --(*this);
  return r_ptr;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>& ArrayRCP<T>::operator+=(Ordinal offset)
{
  if(ptr_) {
    ptr_ += offset;
    lowerOffset_ -= offset;
    upperOffset_ -= offset;
  }
  return *this;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>& ArrayRCP<T>::operator-=(Ordinal offset)
{
  if(ptr_) {
    ptr_ -= offset;
    lowerOffset_ += offset;
    upperOffset_ += offset;
  }
  return *this;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T> ArrayRCP<T>::operator+(Ordinal offset) const
{
  ArrayRCP<T> r_ptr = *this;
  r_ptr+=(offset);
  return r_ptr;
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T> ArrayRCP<T>::operator-(Ordinal offset) const
{
  ArrayRCP<T> r_ptr = *this;
  r_ptr-=offset;
  return r_ptr;
}

// Views

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<const T> ArrayRCP<T>::getConst() const
{
  const T *cptr = ptr_; // Will not compile if not legal!
  return ArrayRCP<const T>(cptr,lowerOffset_,upperOffset_,node_);
}

template<class T>
REFCOUNTPTR_INLINE
ArrayRCP<T>
ArrayRCP<T>::subview( Ordinal lowerOffset, Ordinal size ) const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  assert_in_range(lowerOffset,size);
#endif
  ArrayRCP<T> ptr = *this;
  ptr.ptr_ = ptr.ptr_ + lowerOffset;
  ptr.lowerOffset_ = 0;
  ptr.upperOffset_ = size-1;
  return ptr;
}

// General query functions

template<class T>
REFCOUNTPTR_INLINE
int ArrayRCP<T>::count() const {
  if(node_)
    return node_->count();
  return 0;
}

template<class T>
REFCOUNTPTR_INLINE
template <class T2>
bool ArrayRCP<T>::shares_resource(const ArrayRCP<T2>& r_ptr) const
{
  return node_ == r_ptr.access_node();
  // Note: above, r_ptr is *not* the same class type as *this so we can not
  // access its node_ member directly!  This is an interesting detail to the
  // C++ protected/private protection mechanism!
}

template<class T>
REFCOUNTPTR_INLINE
typename ArrayRCP<T>::Ordinal
ArrayRCP<T>::lowerOffset() const
{
  return lowerOffset_;
}

template<class T>
REFCOUNTPTR_INLINE
typename ArrayRCP<T>::Ordinal
ArrayRCP<T>::upperOffset() const
{
  return upperOffset_;
}

template<class T>
REFCOUNTPTR_INLINE
typename ArrayRCP<T>::Ordinal
ArrayRCP<T>::size() const
{
  return upperOffset_-lowerOffset_+1;
}

// Standard Container-Like Functions

template<class T>
REFCOUNTPTR_INLINE
typename ArrayRCP<T>::const_iterator ArrayRCP<T>::begin() const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  return *this;
#else
  return ptr_;
#endif
}

template<class T>
REFCOUNTPTR_INLINE
typename ArrayRCP<T>::const_iterator
ArrayRCP<T>::end() const
{
#ifdef HAVE_TEUCHOS_ARRAY_BOUNDSCHECK
  return *this + (upperOffset_ + 1);
#else
  return ptr_ + (upperOffset_ + 1);
#endif
}

// Ownership

template<class T>
REFCOUNTPTR_INLINE
T* ArrayRCP<T>::release()
{
  if(node_)
    node_->has_ownership(false);
  return ptr_;
}

template<class T>
REFCOUNTPTR_INLINE
void ArrayRCP<T>::set_has_ownership()
{
  if(node_)
    node_->has_ownership(true);
}

template<class T>
REFCOUNTPTR_INLINE
bool ArrayRCP<T>::has_ownership() const
{
  if(node_)
    return node_->has_ownership();
  return false;
}

// Assertion Functions.

template<class T>
inline
const ArrayRCP<T>&
ArrayRCP<T>::assert_not_null() const
{
  if(!ptr_) PrivateUtilityPack::throw_null(TypeNameTraits<T>::name());
  return *this;
}

template<class T>
inline
const ArrayRCP<T>&
ArrayRCP<T>::assert_in_range( Ordinal lowerOffset, Ordinal size ) const
{
  assert_not_null();
  TEST_FOR_EXCEPTION(
    !( lowerOffset_ <= lowerOffset && lowerOffset+size-1 <= upperOffset_ ), std::logic_error
    ,"Teuchos::ArrayRCP<"<<TypeNameTraits<T>::name()<<">::assert_in_range:"
    " Error, [lowerOffset,lowerOffset+size-1] = ["<<lowerOffset<<","<<(lowerOffset+size-1)<<"] does not lie in the"
    " range ["<<lowerOffset_<<","<<upperOffset_<<"]!"
    );
  return *this;
}

// very bad public functions

template<class T>
inline
ArrayRCP<T>::ArrayRCP(
  T* p, Ordinal lowerOffset, Ordinal upperOffset, bool has_ownership
  )
  : ptr_(p)
  , node_(
    p
    ? new PrivateUtilityPack::RCP_node_tmpl<T,DeallocArrayDelete<T> >(
      p,DeallocArrayDelete<T>(),has_ownership
      )
    : NULL
    )
  ,lowerOffset_(lowerOffset)
  ,upperOffset_(upperOffset)
{
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODES
  if(node_) {
    std::ostringstream os;
    os << "{T=\'"<<TypeNameTraits<T>::name()<<"\',Concrete T=\'"<<typeName(*p)<<"\',p="<<p<<",has_ownership="<<has_ownership<<"}";
    add_new_RCP_node(node_,os.str());
  }
#endif
}

template<class T>
REFCOUNTPTR_INLINE
template<class Dealloc_T>
ArrayRCP<T>::ArrayRCP(
  T* p, Ordinal lowerOffset, Ordinal upperOffset, Dealloc_T dealloc, bool has_ownership
  )
  : ptr_(p)
  , node_( p ? new PrivateUtilityPack::RCP_node_tmpl<T,Dealloc_T>(p,dealloc,has_ownership) : NULL )
  ,lowerOffset_(lowerOffset)
  ,upperOffset_(upperOffset)
{
#ifdef TEUCHOS_SHOW_ACTIVE_REFCOUNTPTR_NODES
  if(node_) {
    std::ostringstream os;
    os << "{T=\'"<<TypeNameTraits<T>::name()<<"\',Concrete T=\'"<<typeName(*p)<<"\',p="<<p<<",has_ownership="<<has_ownership<<"}";
    add_new_ArrayRCP_node(node_,os.str());
  }
#endif
}

template<class T>
inline
ArrayRCP<T>::ArrayRCP(
  T* p, Ordinal lowerOffset, Ordinal upperOffset, node_t* node
  )
  : ptr_(p)
  , node_(node)
  ,lowerOffset_(lowerOffset)
  ,upperOffset_(upperOffset)
{
  if(node_) node_->incr_count();
}

template<class T>
inline
T*& ArrayRCP<T>::access_ptr()
{  return ptr_; }

template<class T>
inline
T* ArrayRCP<T>::access_ptr() const
{  return ptr_; }

template<class T>
inline
typename ArrayRCP<T>::node_t*& ArrayRCP<T>::access_node()
{  return node_; }

template<class T>
inline
typename ArrayRCP<T>::node_t* ArrayRCP<T>::access_node() const
{  return node_; }

}  // end namespace Teuchos

// ///////////////////////////////////////////
// Non-member functions for ArrayRCP

namespace Teuchos {
namespace Utilities {
template<class T1, class T2>
inline void assert_shares_resource(
  const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2
  )
{
#ifdef TEUCHOS_DEBUG
  TEST_FOR_EXCEPT(!p1.shares_resource(p2));
#endif
}
} // namespace Utilities
} // namespace Teuchos

template<class T>
inline
Teuchos::ArrayRCP<T>
Teuchos::arcp(
T* p, typename ArrayRCP<T>::Ordinal lowerOffset
  ,typename ArrayRCP<T>::Ordinal size
  ,bool owns_mem
  )
{
  return ArrayRCP<T>(p,lowerOffset,lowerOffset+size-1,owns_mem);
}

template<class T, class Dealloc_T>
inline
Teuchos::ArrayRCP<T>
Teuchos::arcp(
T* p, typename ArrayRCP<T>::Ordinal lowerOffset
  ,typename ArrayRCP<T>::Ordinal size
  ,Dealloc_T dealloc, bool owns_mem
  )
{
  return ArrayRCP<T>(p,lowerOffset,lowerOffset+size-1,dealloc,owns_mem);
}

template<class T>
inline
Teuchos::ArrayRCP<T>
Teuchos::arcp( typename ArrayRCP<T>::Ordinal size )
{
  return ArrayRCP<T>(new T[size],0,size-1,true);
}

template<class T>
REFCOUNTPTR_INLINE
Teuchos::ArrayRCP<T>
Teuchos::arcp( const RCP<std::vector<T> > &v )
{
  Teuchos::ArrayRCP<T> ptr = arcp(&(*v)[0],0,v->size(),false);
  set_extra_data( v, "std::vector", &ptr );
  return ptr;
}

template<class T>
REFCOUNTPTR_INLINE
Teuchos::RCP<std::vector<T> >
Teuchos::get_std_vector( const ArrayRCP<T> &ptr )
{
  return get_extra_data<RCP<std::vector<T> > >(ptr,"std::vector");
}

template<class T>
REFCOUNTPTR_INLINE
Teuchos::ArrayRCP<const T>
Teuchos::arcp( const RCP<const std::vector<T> > &v )
{
  Teuchos::ArrayRCP<const T> ptr = arcp(&(*v)[0],0,v->size(),false);
  set_extra_data( v, "std::vector", &ptr );
  return ptr;
}

template<class T>
REFCOUNTPTR_INLINE
Teuchos::RCP<const std::vector<T> >
Teuchos::get_std_vector( const ArrayRCP<const T> &ptr )
{
  return get_extra_data<RCP<const std::vector<T> > >(ptr,"std::vector");
}

template<class T>
REFCOUNTPTR_INLINE
bool Teuchos::is_null( const ArrayRCP<T> &p )
{
  return p.access_ptr() == NULL;
}

template<class T>
REFCOUNTPTR_INLINE
bool Teuchos::operator==( const ArrayRCP<T> &p, ENull )
{
  return p.access_ptr() == NULL;
}

template<class T>
REFCOUNTPTR_INLINE
bool Teuchos::operator!=( const ArrayRCP<T> &p, ENull )
{
  return p.access_ptr() != NULL;
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
bool Teuchos::operator==( const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2 )
{
  return p1.access_ptr() == p2.access_ptr();
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
bool Teuchos::operator!=( const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2 )
{
  return p1.access_ptr() != p2.access_ptr();
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
bool Teuchos::operator<( const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2 )
{
  return p1.access_ptr() < p2.access_ptr();
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
bool Teuchos::operator<=( const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2 )
{
  Utilities::assert_shares_resource(p1,p2);
  return p1.access_ptr() <= p2.access_ptr();
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
bool Teuchos::operator>( const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2 )
{
  Utilities::assert_shares_resource(p1,p2);
  return p1.access_ptr() > p2.access_ptr();
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
bool Teuchos::operator>=( const ArrayRCP<T1> &p1, const ArrayRCP<T2> &p2 )
{
  Utilities::assert_shares_resource(p1,p2);
  return p1.access_ptr() >= p2.access_ptr();
}

template<class T2, class T1>
REFCOUNTPTR_INLINE
Teuchos::ArrayRCP<T2>
Teuchos::arcp_reinterpret_cast(const ArrayRCP<T1>& p1)
{
  typedef typename ArrayRCP<T1>::Ordinal Ordinal;
  const int sizeOfT2ToT1 = sizeof(T2)/sizeof(T1);
  Ordinal lowerOffset2 = p1.lowerOffset() / sizeOfT2ToT1;
  Ordinal upperOffset2 = (p1.upperOffset()+1) / sizeOfT2ToT1 -1;
  T2 *ptr2 = reinterpret_cast<T2*>(p1.get());
  return ArrayRCP<T2>(
    ptr2,lowerOffset2,upperOffset2
    ,p1.access_node()
    );
  // Note: Above is just fine even if p1.get()==NULL!
}

template<class T2, class T1>
REFCOUNTPTR_INLINE
Teuchos::ArrayRCP<T2>
Teuchos::arcp_implicit_cast(const ArrayRCP<T1>& p1)
{
  typedef typename ArrayRCP<T1>::Ordinal Ordinal;
  T2 * raw_ptr2 = p1.get();
  return ArrayRCP<T2>(
    raw_ptr2,p1.lowerOffset(),p1.upperOffset()
    ,p1.access_node()
    );
  // Note: Above is just fine even if p1.get()==NULL!
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
void Teuchos::set_extra_data(
  const T1 &extra_data, const std::string& name, Teuchos::ArrayRCP<T2> *p
  ,EPrePostDestruction destroy_when, bool force_unique
  )
{
  p->assert_not_null();
  p->access_node()->set_extra_data( any(extra_data), name, destroy_when, force_unique );
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
T1& Teuchos::get_extra_data( ArrayRCP<T2>& p, const std::string& name )
{
  p.assert_not_null();
  return any_cast<T1>(p.access_node()->get_extra_data(TypeNameTraits<T1>::name(),name));
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
const T1& Teuchos::get_extra_data( const ArrayRCP<T2>& p, const std::string& name )
{
  p.assert_not_null();
  return any_cast<T1>(p.access_node()->get_extra_data(TypeNameTraits<T1>::name(),name));
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
T1* Teuchos::get_optional_extra_data( ArrayRCP<T2>& p, const std::string& name )
{
  p.assert_not_null();
  any *extra_data = p.access_node()->get_optional_extra_data(TypeNameTraits<T1>::name(),name);
  if( extra_data ) return &any_cast<T1>(*extra_data);
  return NULL;
}

template<class T1, class T2>
REFCOUNTPTR_INLINE
const T1* Teuchos::get_optional_extra_data( const ArrayRCP<T2>& p, const std::string& name )
{
  p.assert_not_null();
  any *extra_data = p.access_node()->get_optional_extra_data(TypeNameTraits<T1>::name(),name);
  if( extra_data ) return &any_cast<T1>(*extra_data);
  return NULL;
}

template<class Dealloc_T, class T>
REFCOUNTPTR_INLINE
Dealloc_T&
Teuchos::get_dealloc( ArrayRCP<T>& p )
{
  typedef PrivateUtilityPack::RCP_node_tmpl<typename Dealloc_T::ptr_t,Dealloc_T>  requested_type;
  p.assert_not_null();
  PrivateUtilityPack::RCP_node_tmpl<typename Dealloc_T::ptr_t,Dealloc_T>
    *dnode = dynamic_cast<PrivateUtilityPack::RCP_node_tmpl<typename Dealloc_T::ptr_t,Dealloc_T>*>(p.access_node());
  TEST_FOR_EXCEPTION(
    dnode==NULL, std::logic_error
    ,"get_dealloc<" << TypeNameTraits<Dealloc_T>::name() << "," << TypeNameTraits<T>::name() << ">(p): "
    << "Error, requested type \'" << TypeNameTraits<requested_type>::name()
    << "\' does not match actual type of the node \'" << typeName(*p.access_node()) << "!"
    );
  return dnode->get_dealloc();
}

template<class Dealloc_T, class T>
inline
const Dealloc_T& 
Teuchos::get_dealloc( const Teuchos::ArrayRCP<T>& p )
{
  return get_dealloc<Dealloc_T>(const_cast<ArrayRCP<T>&>(p));
}

template<class Dealloc_T, class T>
REFCOUNTPTR_INLINE
Dealloc_T*
Teuchos::get_optional_dealloc( ArrayRCP<T>& p )
{
  p.assert_not_null();
  typedef PrivateUtilityPack::RCP_node_tmpl<typename Dealloc_T::ptr_t,Dealloc_T>
    RCPNT;
  RCPNT *dnode = dynamic_cast<RCPNT*>(p.access_node());
  if(dnode)
    return &dnode->get_dealloc();
  return NULL;
}

template<class Dealloc_T, class T>
inline
const Dealloc_T*
Teuchos::get_optional_dealloc( const Teuchos::ArrayRCP<T>& p )
{
  return get_optional_dealloc<Dealloc_T>(const_cast<ArrayRCP<T>&>(p));
}

template<class T>
std::ostream& Teuchos::operator<<( std::ostream& out, const ArrayRCP<T>& p )
{
  out
    << TypeNameTraits<ArrayRCP<T> >::name() << "{"
    << "ptr="<<(const void*)(p.get()) // I can't find any alternative to this C cast :-(
    <<",lowerOffset="<<p.lowerOffset()
    <<",upperOffset="<<p.upperOffset()
    <<",size="<<p.size()
    <<",node="<<p.access_node()
    <<",count="<<p.count()
    <<"}";
  return out;
}

#endif // TEUCHOS_ARRAY_RCP_HPP
