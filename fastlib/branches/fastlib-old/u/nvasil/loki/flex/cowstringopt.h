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

#ifndef COW_STRING_OPT_INC_
#define COW_STRING_OPT_INC_

// $Id: cowstringopt.h 754 2006-10-17 19:59:11Z syntheticpp $


////////////////////////////////////////////////////////////////////////////////
// class template CowStringOpt
// Implements Copy on Write over any storage
////////////////////////////////////////////////////////////////////////////////


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

    void append(const E* s, size_type sz);
    
    template <class InputIterator>
    void append(InputIterator b, InputIterator e);

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


////////////////////////////////////////////////////////////////////////////////
// class template CowStringOpt
// Implements Copy on Write over any storage
////////////////////////////////////////////////////////////////////////////////

template <class Storage, typename Align = typename Storage::value_type*>
class CowStringOpt
{
    typedef typename Storage::value_type E;
    typedef typename flex_string_details::get_unsigned<E>::result RefCountType;

public:
    typedef E value_type;
    typedef typename Storage::iterator iterator;
    typedef typename Storage::const_iterator const_iterator;
    typedef typename Storage::allocator_type allocator_type;
    typedef typename allocator_type::size_type size_type;
    typedef typename Storage::reference reference;
    
private:
    union
    {
        mutable char buf_[sizeof(Storage)];
        Align align_;
    };

    Storage& Data() const
    { return *reinterpret_cast<Storage*>(buf_); }

    RefCountType GetRefs() const
    {
        const Storage& d = Data();
        assert(d.size() > 0);
        assert(*d.begin() > 0);
        return *d.begin();
    }
    
    RefCountType& Refs()
    {
        Storage& d = Data();
        assert(d.size() > 0);
        return reinterpret_cast<RefCountType&>(*d.begin());
    }
    
    void MakeUnique() const
    {
        assert(GetRefs() >= 1);
        if (GetRefs() == 1) return;

        union
        {
            char buf_[sizeof(Storage)];
            Align align_;
        } temp;

        new(buf_) Storage(
            *new(temp.buf_) Storage(Data()), 
            flex_string_details::Shallow());
        *Data().begin() = 1;
    }

public:
    CowStringOpt(const CowStringOpt& s)
    {
        if (s.GetRefs() == std::numeric_limits<RefCountType>::max())
        {
            // must make a brand new copy
            new(buf_) Storage(s.Data()); // non shallow
            Refs() = 1;
        }
        else
        {
            new(buf_) Storage(s.Data(), flex_string_details::Shallow());
            ++Refs();
        }
        assert(Data().size() > 0);
    }
    
    CowStringOpt(const allocator_type& a)
    {
        new(buf_) Storage(1, 1, a);
    }
    
    CowStringOpt(const E* s, size_type len, const allocator_type& a)
    {
        // Warning - MSVC's debugger has trouble tracing through the code below.
        // It seems to be a const-correctness issue
        //
        new(buf_) Storage(a);
        Data().reserve(len + 1);
        Data().resize(1, 1);
        Data().append(s, s + len);
    }

    CowStringOpt(size_type len, E c, const allocator_type& a)
    {
        new(buf_) Storage(len + 1, c, a);
        Refs() = 1;
    }
    
    CowStringOpt& operator=(const CowStringOpt& rhs)
    {
        CowStringOpt(rhs).swap(*this);
        return *this;
    }

    ~CowStringOpt()
    {
        assert(Data().size() > 0);
        if (--Refs() == 0) Data().~Storage();
    }

    iterator begin()
    {
        assert(Data().size() > 0);
        MakeUnique();
        return Data().begin() + 1; 
    }
    
    const_iterator begin() const
    {
        assert(Data().size() > 0);
        return Data().begin() + 1; 
    }
    
    iterator end()
    {
        MakeUnique();
        return Data().end(); 
    }
    
    const_iterator end() const
    {
        return Data().end(); 
    }
    
    size_type size() const
    {
        assert(Data().size() > 0);
        return Data().size() - 1;
    }

    size_type max_size() const
    { 
        assert(Data().max_size() > 0);
        return Data().max_size() - 1;
    }

    size_type capacity() const
    { 
        assert(Data().capacity() > 0);
        return Data().capacity() - 1;
    }

    void resize(size_type n, E c)
    {
        assert(Data().size() > 0);
        MakeUnique();
        Data().resize(n + 1, c);
    }

    template <class FwdIterator>
    void append(FwdIterator b, FwdIterator e)
    {
        MakeUnique();
        Data().append(b, e);
    }
    
    void reserve(size_type res_arg)
    {
        if (capacity() > res_arg) return;
        MakeUnique();
        Data().reserve(res_arg + 1);
    }
    
    void swap(CowStringOpt& rhs)
    {
        Data().swap(rhs.Data());
    }
    
    const E* c_str() const
    { 
        assert(Data().size() > 0);
        return Data().c_str() + 1;
    }

    const E* data() const
    { 
        assert(Data().size() > 0);
        return Data().data() + 1;
    }
    
    allocator_type get_allocator() const
    { 
        return Data().get_allocator();
    }
};


#endif // COW_STRING_OPT_INC_
