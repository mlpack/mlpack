////////////////////////////////////////////////////////////////////////////////
/// \file config.hpp
///
/// \brief This header provides configuration data for the bpstd library
////////////////////////////////////////////////////////////////////////////////

/*
  The MIT License (MIT)

  Copyright (c) 2020 Matthew Rodusek All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
#ifndef BPSTD_DETAIL_CONFIG_HPP
#define BPSTD_DETAIL_CONFIG_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#if !defined(__cplusplus)
# error This library requires a C++ compiler
#endif

// _MSC_VER check is due to MSVC not defining __cplusplus to be 201103L
#if !defined(_MSC_VER) && __cplusplus < 201103L
# error This library must be compiled with C++11 support
#endif

#if defined(__cplusplus) && __cplusplus >= 201402L
# define BPSTD_CPP14_CONSTEXPR constexpr
# define BPSTD_HAS_TEMPLATE_VARIABLES 1
#else
# define BPSTD_CPP14_CONSTEXPR
# define BPSTD_HAS_TEMPLATE_VARIABLES 0
#endif

#if defined(__cplusplus) && __cplusplus >= 201703L
# define BPSTD_CPP17_CONSTEXPR constexpr
# define BPSTD_CPP17_INLINE inline
# define BPSTD_HAS_INLINE_VARIABLES 1
#else
# define BPSTD_CPP17_CONSTEXPR
# define BPSTD_CPP17_INLINE
# define BPSTD_HAS_INLINE_VARIABLES 0
#endif

#define BPSTD_UNUSED(x) static_cast<void>(x)

// Use __may_alias__ attribute on gcc and clang
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ > 5)
# define BPSTD_MAY_ALIAS __attribute__((__may_alias__))
#else // defined(__clang__) || defined __GNUC__
# define BPSTD_MAY_ALIAS
#endif // defined __clang__ || defined __GNUC__

#if !defined(BPSTD_INLINE_VISIBILITY)
// When using 'clang-cl', don't forceinline -- since it results in code generation
// failures in 'variant'
# if defined(__clang__) && defined(_MSC_VER)
#  define BPSTD_INLINE_VISIBILITY __attribute__((visibility("hidden"), no_instrument_function))
# elif defined(__clang__) || defined(__GNUC__)
#  define BPSTD_INLINE_VISIBILITY __attribute__((visibility("hidden"), always_inline, no_instrument_function))
# elif defined(_MSC_VER)
#  define BPSTD_INLINE_VISIBILITY __forceinline
# else
#  define BPSTD_INLINE_VISIBILITY
# endif
#endif // !defined(BPSTD_INLINE_VISIBILITY)

#if defined(_MSC_VER)
# define BPSTD_COMPILER_DIAGNOSTIC_PREAMBLE \
  __pragma(warning(push)) \
  __pragma(warning(disable:4714)) \
  __pragma(warning(disable:4100))
#else
# define BPSTD_COMPILER_DIAGNOSTIC_PREAMBLE
#endif

#if defined(_MSC_VER)
# define BPSTD_COMPILER_DIAGNOSTIC_POSTAMBLE \
  __pragma(warning(pop))
#else
# define BPSTD_COMPILER_DIAGNOSTIC_POSTAMBLE
#endif

#endif /* BPSTD_DETAIL_CONFIG_HPP */
