#ifndef YASLI_PROTOCOLS_H_
#define YASLI_PROTOCOLS_H_

// $Id: yasli_protocols.h 754 2006-10-17 19:59:11Z syntheticpp $


#include <complex>
#include <functional>
#include "yasli_memory.h"
#include <memory.h>

namespace yasli_protocols
{
             
    // Most conservative
    template <class T>
    struct safe_move_traits
    {
        static T* destructive_move(
            T* begin, 
            T* end, 
            void* dest)
        {
            T* tdest = static_cast<T*>(dest);
            typedef std::less<T*> ls;
            assert(!ls()(tdest, end) || ls()(tdest, begin - (end - begin)));
            tdest = /*yasli*/std::uninitialized_copy(begin, end, tdest);
            if (yasli_nstd::is_class<T>::value)
            {
                for (; begin != end; ++begin)
                {
                    begin->~T();
                }
            }
            return tdest;
        }
        static T* nondestructive_move(
            T* begin, 
            T* end, 
            void* dest)
        {
            T* d = static_cast<T*>(dest);
            for (; begin != end; ++begin, ++d)
                new(d) T(*begin);
            return d;
        }
        static T* nondestructive_assign_move(
            T* begin, 
            T* end, 
            T* dest)
        {
            if (begin <= dest && dest < end)
            {
                dest += end - begin;
                T* const result = dest;
                while (begin != end)
                    *--dest = *--end;
                return result;
            }
            for (; begin != end; ++begin, ++dest)
                *dest = *begin;
            return dest;
        }
    };
    

    template <class T>
    struct memmove_traits
    {
        static T* destructive_move(
            T* begin, 
            T* end, 
            void* dest)
        {
            memmove(dest, begin, (end - begin) * sizeof(T));
            return static_cast<T*>(dest) + (end - begin);
        };
        static T* nondestructive_move(
            T* begin, 
            T* end, 
            void* dest)
        {
            memmove(dest, begin, (end - begin) * sizeof(T));
            return static_cast<T*>(dest) + (end - begin);
        }
        static T* nondestructive_assign_move(
            T* begin, 
            T* end, 
            T* dest)
        {
            yasli_nstd::destroy_range(begin, end);
            memmove(dest, begin, (end - begin) * sizeof(T));
            return static_cast<T*>(dest) + (end - begin);
        }
    };
    
    // for nonspecialized classes, use safe_move_traits
    template <class T>
    struct move_traits: public 
    yasli_nstd::type_selector<yasli_nstd::is_class<T>::value == 0,
                                          memmove_traits<T>,
                                          safe_move_traits<T>
                                      >::result
    {};    
       
    template <class T>
    struct move_traits<std::complex<T> >:public 
    yasli_nstd::type_selector<sizeof(std::complex<T>) == 2 * sizeof(T),
                                          memmove_traits< std::complex<T> >,
                                          safe_move_traits< std::complex<T> >
                                      >::result
    {
    };
}

#endif // YASLI_PROTOCOLS_H_
