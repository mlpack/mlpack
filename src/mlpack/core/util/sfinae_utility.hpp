/**
 * @file sfinae_utility.hpp
 * @author Trironk Kiatkungwanglai
 *
 * Utilities for the SFINAE Paradigm.
 *
 * Taken from http://stackoverflow.com/a/6324863/391618
 *
 */
#ifndef __MLPACK_CORE_SFINAE_UTILITY
#define __MLPACK_CORE_SFINAE_UTILITY

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>

// Taken from: http://stackoverflow.com/a/264088/391618

#define HAS_MEM_FUNC(func, name)                                               \
    template<typename T, typename sig>                                         \
    struct name {                                                              \
        typedef char yes;                                                      \
        typedef long no;                                                       \
        template <typename U, U> struct type_check;                            \
        template <typename _1> static yes &chk(type_check<sig, &_1::func> *);  \
        template <typename   > static no  &chk(...);                           \
        static bool const value = sizeof(chk<T>(0)) == sizeof(yes);            \
    };


#endif
