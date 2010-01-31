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

#ifndef VECTOR_STRING_STORAGE_INC_
#define VECTOR_STRING_STORAGE_INC_

// $Id: vectorstringstorage.h 754 2006-10-17 19:59:11Z syntheticpp $


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
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
// class template VectorStringStorage
// Uses std::vector
// Takes advantage of the Empty Base Optimization if available
////////////////////////////////////////////////////////////////////////////////

template <typename E, class A = std::allocator<E> >
class VectorStringStorage : protected std::vector<E, A>
{
    typedef std::vector<E, A> base;

public: // protected:
    typedef E value_type;
    typedef typename base::iterator iterator;
    typedef typename base::const_iterator const_iterator;
    typedef A allocator_type;
    typedef typename A::size_type size_type;
    typedef typename A::reference reference;
    
    VectorStringStorage(const VectorStringStorage& s) : base(s)
    { }
    
    VectorStringStorage(const A& a) : base(1, value_type(), a)
    { }
    
    VectorStringStorage(const value_type* s, size_type len, const A& a)
    : base(a)
    {
        base::reserve(len + 1);
        base::insert(base::end(), s, s + len);
        // Terminating zero
        base::push_back(value_type());
    }

    VectorStringStorage(size_type len, E c, const A& a)
    : base(len + 1, c, a)
    {
        // Terminating zero
        base::back() = value_type();
    }
    
    VectorStringStorage& operator=(const VectorStringStorage& rhs)
    {
        base& v = *this;
        v = rhs;
        return *this;
    }
   
    iterator begin()
    { return base::begin(); }
    
    const_iterator begin() const
    { return base::begin(); }
    
    iterator end()
    { return base::end() - 1; }
    
    const_iterator end() const
    { return base::end() - 1; }
    
    size_type size() const
    { return base::size() - 1; }

    size_type max_size() const
    { return base::max_size() - 1; }

    size_type capacity() const
    { return base::capacity() - 1; }

    void reserve(size_type res_arg)
    { 
        assert(res_arg < max_size());
        base::reserve(res_arg + 1); 
    }
    
    template <class ForwardIterator>
    void append(ForwardIterator b, ForwardIterator e)
    {
        const typename std::iterator_traits<ForwardIterator>::difference_type
            sz = std::distance(b, e);
        assert(sz >= 0);
        if (sz == 0) return;
        base::reserve(base::size() + sz);
        const value_type & v = *b;
        struct OnBlockExit
        {
            VectorStringStorage * that;
            ~OnBlockExit()
            {
                that->base::push_back(value_type());
            }
        } onBlockExit = { this };
        (void) onBlockExit;
        assert(!base::empty());
        assert(base::back() == value_type());
        base::back() = v;
        base::insert(base::end(), ++b, e);
    }

    void resize(size_type n, E c)
    {
        base::reserve(n + 1);
        base::back() = c;
        base::resize(n + 1, c);
        base::back() = E();
    }

    void swap(VectorStringStorage& rhs)
    { base::swap(rhs); }
    
    const E* c_str() const
    { return &*begin(); }

    const E* data() const
    { return &*begin(); }
    
    A get_allocator() const
    { return base::get_allocator(); }
};


#endif // VECTOR_STRING_STORAGE_INC_
