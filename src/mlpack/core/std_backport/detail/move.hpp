////////////////////////////////////////////////////////////////////////////////
/// \file move.hpp
///
/// \brief This internal header provides the definition of the move and forward
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
#ifndef BPSTD_DETAIL_MOVE_HPP
#define BPSTD_DETAIL_MOVE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "config.hpp"

#include <type_traits>

BPSTD_COMPILER_DIAGNOSTIC_PREAMBLE

namespace bpstd {

  //----------------------------------------------------------------------------
  // Utilities
  //----------------------------------------------------------------------------

  /// \{
  /// \brief Forwards a reference \p t
  ///
  /// \tparam T the type to forward
  /// \param t the reference
  /// \return the forwarded reference
  template <typename T>
  constexpr T&& forward(typename std::remove_reference<T>::type& t) noexcept;
  template <typename T>
  constexpr T&& forward(typename std::remove_reference<T>::type&& t) noexcept;
  /// \}

  /// \brief Casts \p x to an rvalue
  ///
  /// \param x the parameter to move
  /// \return rvalue reference to \p x
  template <typename T>
  constexpr T&& move(T& x) noexcept;

  /// \brief Casts \p x to an rvalue
  ///
  /// \param x the parameter to move
  /// \return rvalue reference to \p x
  template <typename T>
  constexpr typename std::remove_reference<T>::type&& move(T&& x) noexcept;

} // namespace bpstd

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

template <typename T>
inline BPSTD_INLINE_VISIBILITY constexpr
T&& bpstd::forward(typename std::remove_reference<T>::type& t)
  noexcept
{
  return static_cast<T&&>(t);
}

template <typename T>
inline BPSTD_INLINE_VISIBILITY constexpr
T&& bpstd::forward(typename std::remove_reference<T>::type&& t)
  noexcept
{
  return static_cast<T&&>(t);
}

template <typename T>
inline BPSTD_INLINE_VISIBILITY constexpr
T&& bpstd::move(T& x)
  noexcept
{
  return static_cast<T&&>(x);
}

template <typename T>
inline BPSTD_INLINE_VISIBILITY constexpr
typename std::remove_reference<T>::type&& bpstd::move(T&& x)
  noexcept
{
  return static_cast<T&&>(x);
}

BPSTD_COMPILER_DIAGNOSTIC_POSTAMBLE

#endif /* BPSTD_DETAIL_MOVE_HPP */
