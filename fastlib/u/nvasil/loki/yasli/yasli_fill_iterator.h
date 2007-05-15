#ifndef YASLI_FILL_ITERATOR_H_
#define YASLI_FILL_ITERATOR_H_

// $Id: yasli_fill_iterator.h 754 2006-10-17 19:59:11Z syntheticpp $


#include <iterator>
#include <cstddef>

namespace yasli_nstd
{
    template <class T>
    class fill_iterator_base
        : public std::iterator<
            std::random_access_iterator_tag,
            T,
            ptrdiff_t,
            T*,
            T&>
    {
    };

    template <class T>
    class fill_iterator_base<T&>
        : public std::iterator<
            std::random_access_iterator_tag,
            T,
            ptrdiff_t,
            T*,
            T&>
    {
    };

    template <class T>
    class fill_iterator : public fill_iterator_base<T>
    {
        T value_;
        /*difference_type*/ ptrdiff_t count_;//////////////////////////////////
    
    public:
        typedef std::ptrdiff_t difference_type;
        typedef typename fill_iterator_base<T>::pointer pointer;
        typedef typename fill_iterator_base<T>::reference reference;
        //typedef iterator_type;

        fill_iterator()
        {    
        }

        explicit fill_iterator(reference value, difference_type count = 0)
        : value_(value), count_(count)
        {    
        }

        template<class U>
        fill_iterator(const fill_iterator<U>& rhs)
        : value_(rhs.value_), count_(rhs.count_)
        {
        }

        reference operator*() const
        {
            return value_;
        }

        pointer operator->() const
        {
            return &**this;
        }

        fill_iterator& operator++()
        {    
            ++count_;
            return *this;
        }

        fill_iterator operator++(int)
        {
            fill_iterator it(*this);
            ++*this;
            return it;
        }

        fill_iterator& operator--()
        {    
            --count_;
            return *this;
        }

        fill_iterator operator--(int)
        {
            fill_iterator it(*this);
            --*this;
            return it;
        }

        fill_iterator& operator+=(difference_type d)
        {
            count_ += d;
            return *this;
        }

        fill_iterator operator+(difference_type d) const
        {
            return fill_iterator(*this) += d;
        }

        fill_iterator& operator-=(difference_type d)
        {
            count_ -= d;
            return *this;
        }

        fill_iterator operator-(difference_type d) const
        {
            return fill_iterator(*this) -= d;
        }

        difference_type operator-(const fill_iterator<T>& rhs) const
        {
            return count_ - rhs.count_;
        }

        reference operator[](difference_type) const
        {
            return **this;
        }

        template <class T2> 
        bool operator==(const fill_iterator<T2>& rhs) const
        {
            return count_ == rhs.count_;
        }

    };

    template <class T, class D> 
    inline fill_iterator<T> operator+(D lhs, const fill_iterator<T>& rhs)
    {
        return rhs + lhs;
    }

    template <class T> 
    inline bool operator!=(
        const fill_iterator<T>& lhs,
        const fill_iterator<T>& rhs)
    {    // test for fill_iterator inequality
        return !(lhs == rhs);
    }

    template <class T> 
    inline bool operator<(
        const fill_iterator<T>& lhs,
        const fill_iterator<T>& rhs)
    {
        return lhs.count_ < rhs.count_;
    }

    template <class T> 
    inline bool operator>(
        const fill_iterator<T>& lhs,
        const fill_iterator<T>& rhs)
    {
        return rhs < lhs;
    }

    template <class T> 
    inline bool operator<=(
        const fill_iterator<T>& lhs,
        const fill_iterator<T>& rhs)
    {
        return !(rhs < lhs);
    }

    template <class T> 
    inline bool operator>=(
        const fill_iterator<T>& lhs,
        const fill_iterator<T>& rhs)
    {
        return !(lhs < rhs);
    }
} // namespace yasli_nstd

#endif
