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

#ifndef SMALL_STRING_OPT_INC_
#define SMALL_STRING_OPT_INC_

// $Id: smallstringopt.h 754 2006-10-17 19:59:11Z syntheticpp $


////////////////////////////////////////////////////////////////////////////////
// class template SmallStringOpt
// Builds the small string optimization over any other storage
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
// class template SmallStringOpt
// Builds the small string optimization over any other storage
////////////////////////////////////////////////////////////////////////////////

template <class Storage, unsigned int threshold, 
    typename Align = typename Storage::value_type*>
class SmallStringOpt
{
public:
    typedef typename Storage::value_type value_type;
    typedef value_type* iterator;
    typedef const value_type* const_iterator;
    typedef typename Storage::allocator_type allocator_type;
    typedef typename allocator_type::size_type size_type;
    typedef typename Storage::reference reference;
    
private:
    enum { temp1 = threshold * sizeof(value_type) > sizeof(Storage) 
        ? threshold  * sizeof(value_type) 
        : sizeof(Storage) };
    
    enum { temp2 = temp1 > sizeof(Align) ? temp1 : sizeof(Align) };

public:
    enum { maxSmallString = 
        (temp2 + sizeof(value_type) - 1) / sizeof(value_type) };
    
private:
    enum { magic = maxSmallString + 1 };
    
    union
    {
        mutable value_type buf_[maxSmallString + 1];
        Align align_;
    };
    
    Storage& GetStorage()
    {
        assert(buf_[maxSmallString] == magic);
        Storage* p = reinterpret_cast<Storage*>(&buf_[0]);
        return *p;
    }
    
    const Storage& GetStorage() const
    {
        assert(buf_[maxSmallString] == magic);
        const Storage *p = reinterpret_cast<const Storage*>(&buf_[0]);
        return *p;
    }
    
    bool Small() const
    {
        return buf_[maxSmallString] != magic;
    }
        
public:
    SmallStringOpt(const SmallStringOpt& s)
    {
        if (s.Small())
        {
            flex_string_details::pod_copy(
                s.buf_, 
                s.buf_ + s.size(), 
                buf_);
        }
        else
        {
            new(buf_) Storage(s.GetStorage());
        }
        buf_[maxSmallString] = s.buf_[maxSmallString];
    }
    
    SmallStringOpt(const allocator_type&)
    {
        buf_[maxSmallString] = maxSmallString;
    }
    
    SmallStringOpt(const value_type* s, size_type len, const allocator_type& a)
    {
        if (len <= maxSmallString)
        {
            flex_string_details::pod_copy(s, s + len, buf_);
            buf_[maxSmallString] = value_type(maxSmallString - len);
        }
        else
        {
            new(buf_) Storage(s, len, a);
            buf_[maxSmallString] = magic;
        }
    }

    SmallStringOpt(size_type len, value_type c, const allocator_type& a)
    {
        if (len <= maxSmallString)
        {
            flex_string_details::pod_fill(buf_, buf_ + len, c);
            buf_[maxSmallString] = value_type(maxSmallString - len);
        }
        else
        {
            new(buf_) Storage(len, c, a);
            buf_[maxSmallString] = magic;
        }
    }
    
    SmallStringOpt& operator=(const SmallStringOpt& rhs)
    {
        if (&rhs != this)
        {
            reserve(rhs.size());
            resize(0, 0);
            append(rhs.data(), rhs.data() + rhs.size());
        }
        return *this;
    }

    ~SmallStringOpt()
    {
        if (!Small()) GetStorage().~Storage();
    }

    iterator begin()
    {
        if (Small()) return buf_;
        return &*GetStorage().begin(); 
    }
    
    const_iterator begin() const
    {
        if (Small()) return buf_;
        return &*GetStorage().begin(); 
    }
    
    iterator end()
    {
        if (Small()) return buf_ + maxSmallString - buf_[maxSmallString];
        return &*GetStorage().end(); 
    }
    
    const_iterator end() const
    {
        if (Small()) return buf_ + maxSmallString - buf_[maxSmallString];
        return &*GetStorage().end(); 
    }
    
    size_type size() const
    {
        assert(!Small() || maxSmallString >= buf_[maxSmallString]);
        return Small() 
            ? maxSmallString - buf_[maxSmallString] 
            : GetStorage().size();
    }

    size_type max_size() const
    { return get_allocator().max_size(); }

    size_type capacity() const
    { return Small() ? maxSmallString : GetStorage().capacity(); }

    void reserve(size_type res_arg)
    {
        if (Small())
        {
            if (res_arg <= maxSmallString) return;
            SmallStringOpt temp(*this);
            this->~SmallStringOpt();
            new(buf_) Storage(temp.data(), temp.size(), 
                temp.get_allocator());
            buf_[maxSmallString] = magic;
            GetStorage().reserve(res_arg);
        }
        else
        {
            GetStorage().reserve(res_arg);
        }
        assert(capacity() >= res_arg);
    }
    
    template <class FwdIterator>
    void append(FwdIterator b, FwdIterator e)
    {
        if (!Small())
        {
            GetStorage().append(b, e);
        }
        else
        {
            // append to a small string
            const size_type 
                sz = std::distance(b, e),
                neededCapacity = maxSmallString - buf_[maxSmallString] + sz;

            if (maxSmallString < neededCapacity)
            {
                // need to change storage strategy
                allocator_type alloc;
                Storage temp(alloc);
                temp.reserve(neededCapacity);
                temp.append(buf_, buf_ + maxSmallString - buf_[maxSmallString]);
                temp.append(b, e);
                buf_[maxSmallString] = magic;
                new(buf_) Storage(temp.get_allocator());
                GetStorage().swap(temp);
            }
            else
            {
                std::copy(b, e, buf_ + maxSmallString - buf_[maxSmallString]);
                buf_[maxSmallString] = buf_[maxSmallString] - value_type(sz);
            }
        }
    }

    void resize(size_type n, value_type c)
    {
        if (Small())
        {
            if (n > maxSmallString)
            {
                // Small string resized to big string
                SmallStringOpt temp(*this); // can't throw
                // 11-17-2001: correct exception safety bug
                Storage newString(temp.data(), temp.size(), 
                    temp.get_allocator());
                newString.resize(n, c);
                // We make the reasonable assumption that an empty Storage
                //     constructor won't throw
                this->~SmallStringOpt();
                new(&buf_[0]) Storage(temp.get_allocator());
                buf_[maxSmallString] = value_type(magic);
                GetStorage().swap(newString);
            }
            else
            {
                // Small string resized to small string
                // 11-17-2001: bug fix: terminating zero not copied
                size_type toFill = n > size() ? n - size() : 0;
                flex_string_details::pod_fill(end(), end() + toFill, c);
                buf_[maxSmallString] = value_type(maxSmallString - n);
            }
        }
        else
        {
            if (n > maxSmallString)
            {
                // Big string resized to big string
                GetStorage().resize(n, c);
            }
            else
            {
                // Big string resized to small string
                // 11-17=2001: bug fix in the assertion below
                assert(capacity() > n);
                SmallStringOpt newObj(data(), n, get_allocator());
                newObj.swap(*this);
            }
        }
    }

    void swap(SmallStringOpt& rhs)
    {
        if (Small())
        {
            if (rhs.Small())
            {
                // Small swapped with small
                std::swap_ranges(buf_, buf_ + maxSmallString + 1, 
                    rhs.buf_);
            }
            else
            {
                // Small swapped with big
                // Make a copy of myself - can't throw
                SmallStringOpt temp(*this);
                // Nuke myself
                this->~SmallStringOpt();
                // Make an empty storage for myself (likely won't throw)
                new(buf_) Storage(0, value_type(), rhs.get_allocator());
                buf_[maxSmallString] = magic;
                // Recurse to this same function
                swap(rhs);
                // Nuke rhs
                rhs.~SmallStringOpt();
                // Build the new small string into rhs
                new(&rhs) SmallStringOpt(temp);
            }
        }
        else
        {
            if (rhs.Small())
            {
                // Big swapped with small
                // Already implemented, recurse with reversed args
                rhs.swap(*this);
            }
            else
            {
                // Big swapped with big
                GetStorage().swap(rhs.GetStorage());
            }
        }
    }
    
    const value_type* c_str() const
    { 
        if (!Small()) return GetStorage().c_str(); 
        buf_[maxSmallString - buf_[maxSmallString]] = value_type();
        return buf_;
    }

    const value_type* data() const
    { return Small() ? buf_ : GetStorage().data(); }
    
    allocator_type get_allocator() const
    { return allocator_type(); }
};


#endif // SMALL_STRING_OPT_INC_
