////////////////////////////////////////////////////////////////////////////////
// flex_string
// Copyright (c) 2001 by Andrei Alexandrescu
// Permission to use, copy, modify, distribute and sell this software for any
//     purpose is hereby granted without fee, provided that the above copyright
//     notice appear in all copies and that both that copyright notice and this
//     permission notice appear in supporting documentation.
// The author makes no representations about the
//     suitability of this software for any purpose. It is provided "as is"
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////

#ifndef ALLOCATOR_STRING_STORAGE_INC_
#define ALLOCATOR_STRING_STORAGE_INC_

// $Id: allocatorstringstorage.h 754 2006-10-17 19:59:11Z syntheticpp $


/* This is the template for a storage policy
////////////////////////////////////////////////////////////////////////////////
template <typename E, class A = @>
class StoragePolicy
{
    typedef E value_type;
    typedef @ iterator;
    typedef @ const_iterator;
    typedef A allocator_type;
    typedef @ size_type;
    
    StoragePolicy(const StoragePolicy& s);
    StoragePolicy(const A&);
    StoragePolicy(const E* s, size_type len, const A&);
    StoragePolicy(size_type len, E c, const A&);
    ~StoragePolicy();

    iterator begin();
    const_iterator begin() const;
    iterator end();
    const_iterator end() const;
    
    size_type size() const;
    size_type max_size() const;
    size_type capacity() const;

    void reserve(size_type res_arg);

    template <class ForwardIterator>
    void append(ForwardIterator b, ForwardIterator e);

    void resize(size_type newSize, E fill);

    void swap(StoragePolicy& rhs);
    
    const E* c_str() const;
    const E* data() const;
    
    A get_allocator() const;
};
////////////////////////////////////////////////////////////////////////////////
*/

#include <memory>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <stdexcept>
#include "flex_string_details.h"
#include "simplestringstorage.h"

////////////////////////////////////////////////////////////////////////////////
// class template AllocatorStringStorage
// Allocates with your allocator
// Takes advantage of the Empty Base Optimization if available
////////////////////////////////////////////////////////////////////////////////

template <typename E, class A = std::allocator<E> >
class AllocatorStringStorage : public A
{
    typedef typename A::size_type size_type;
    typedef typename SimpleStringStorage<E, A>::Data Data;

    void* Alloc(size_type sz, const void* p = 0)
    {
        return A::allocate(1 + (sz - 1) / sizeof(E), 
            static_cast<const char*>(p));
    }

    void* Realloc(void* p, size_type oldSz, size_type newSz)
    {
        void* r = Alloc(newSz);
        flex_string_details::pod_copy(p, p + Min(oldSz, newSz), r);
        Free(p, oldSz);
        return r;
    }

    void Free(void* p, size_type sz)
    {
        A::deallocate(static_cast<E*>(p), sz);
    }

    Data* pData_;

    void Init(size_type size, size_type cap)
    {
        assert(size <= cap);

        if (cap == 0)
        {
            pData_ = const_cast<Data*>(
                &SimpleStringStorage<E, A>::emptyString_);
        }
        else
        {
            pData_ = static_cast<Data*>(Alloc(
                cap * sizeof(E) + sizeof(Data)));
            pData_->pEnd_ = pData_->buffer_ + size;
            pData_->pEndOfMem_ = pData_->buffer_ + cap;
        }
    }
    
public:
    typedef E value_type;
    typedef A allocator_type;
    typedef typename A::pointer iterator;
    typedef typename A::const_pointer const_iterator;

    AllocatorStringStorage() 
    : A(), pData_(0)
    {
    }

    AllocatorStringStorage(const AllocatorStringStorage& rhs) 
    : A(rhs.get_allocator())
    {
        const size_type sz = rhs.size();
        Init(sz, sz);
        if (sz) flex_string_details::pod_copy(rhs.begin(), rhs.end(), begin());
    }
    
    AllocatorStringStorage(const AllocatorStringStorage& s, 
        flex_string_details::Shallow) 
    : A(s.get_allocator())
    {
        pData_ = s.pData_;
    }
    
    AllocatorStringStorage(const A& a) : A(a)
    { 
        pData_ = const_cast<Data*>(
            &SimpleStringStorage<E, A>::emptyString_);
    }
    
    AllocatorStringStorage(const E* s, size_type len, const A& a)
    : A(a)
    {
        Init(len, len);        
        flex_string_details::pod_copy(s, s + len, begin());
    }

    AllocatorStringStorage(size_type len, E c, const A& a)
    : A(a)
    {
        Init(len, len);
        flex_string_details::pod_fill(&*begin(), &*end(), c);
    }
    
    AllocatorStringStorage& operator=(const AllocatorStringStorage& rhs)
    {
        const size_type sz = rhs.size();
        reserve(sz);
        flex_string_details::pod_copy(&*rhs.begin(), &*rhs.end(), begin());
        pData_->pEnd_ = &*begin() + rhs.size();
        return *this;
    }
    
    ~AllocatorStringStorage()
    {
        if (capacity())
        {
            Free(pData_, 
                sizeof(Data) + capacity() * sizeof(E));
        }
    }
        
    iterator begin()
    { return pData_->buffer_; }
    
    const_iterator begin() const
    { return pData_->buffer_; }
    
    iterator end()
    { return pData_->pEnd_; }
    
    const_iterator end() const
    { return pData_->pEnd_; }
    
    size_type size() const
    { return size_type(end() - begin()); }

    size_type max_size() const
    { return A::max_size(); }

    size_type capacity() const
    { return size_type(pData_->pEndOfMem_ - pData_->buffer_); }

    void resize(size_type n, E c)
    {
        reserve(n);
        iterator newEnd = begin() + n;
        iterator oldEnd = end();
        if (newEnd > oldEnd) 
        {
            // Copy the characters
            flex_string_details::pod_fill(oldEnd, newEnd, c);
        }
        if (capacity()) pData_->pEnd_ = newEnd;
    }

    void reserve(size_type res_arg)
    {
        if (res_arg <= capacity())
        {
            // @@@ shrink to fit here 
            return;
        }
        
        A& myAlloc = *this;
        AllocatorStringStorage newStr(myAlloc);
        newStr.Init(size(), res_arg);
        
        flex_string_details::pod_copy(begin(), end(), newStr.begin());
        
        swap(newStr);
    }

    template <class ForwardIterator>
    void append(ForwardIterator b, ForwardIterator e)
    {
        const size_type 
            sz = std::distance(b, e),
            neededCapacity = size() + sz;

        if (capacity() < neededCapacity)
        {
            static std::less_equal<const E*> le;
            (void) le;
            assert(!(le(begin(), &*b) && le(&*b, end())));
            reserve(neededCapacity);
        }
        std::copy(b, e, end());
        pData_->pEnd_ += sz;
    }
    
    void swap(AllocatorStringStorage& rhs)
    {
        // @@@ The following line is commented due to a bug in MSVC
        //std::swap(lhsAlloc, rhsAlloc);
        std::swap(pData_, rhs.pData_);
    }
    
    const E* c_str() const
    { 
        if (pData_ != &SimpleStringStorage<E, A>::emptyString_) 
        {
            *pData_->pEnd_ = E();
        }
        return &*begin(); 
    }

    const E* data() const
    { return &*begin(); }
    
    A get_allocator() const
    { return *this; }
};

#endif // ALLOCATOR_STRING_STORAGE_INC_
