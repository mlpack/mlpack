/**
 * @file sfinae_utility.hpp
 * @author Trironk Kiatkungwanglai
 *
 * This file contains macro utilities for the SFINAE Paradigm. These utilities
 * determine if classes passed in as template parameters contain members at
 * compile time, which is useful for changing functionality depending on what
 * operations an object is capable of performing.
 */
#ifndef __MLPACK_CORE_SFINAE_UTILITY
#define __MLPACK_CORE_SFINAE_UTILITY

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>

/*
 * Note: This macro is taken from http://stackoverflow.com/a/264088/391618
 *
 * This macro generates a template struct that is useful for enabling/disabling
 * a method if the template class passed in contains a member function matching
 * a given signature with a specified name.
 * 
 * @param NAME the name of the struct to construct. For example: HasToString
 * @param FUNC the name of the function to check for. For example: ToString
 */
#define HAS_MEM_FUNC(FUNC, NAME)                                               \
    template<typename T, typename sig>                                         \
    struct NAME {                                                              \
        typedef char yes;                                                      \
        typedef long no;                                                       \
        template <typename U, U> struct type_check;                            \
        template <typename _1> static yes &chk(type_check<sig, &_1::FUNC> *);  \
        template <typename   > static no  &chk(...);                           \
        static bool const value = sizeof(chk<T>(0)) == sizeof(yes);            \
    };


#endif
