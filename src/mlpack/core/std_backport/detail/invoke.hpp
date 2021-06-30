////////////////////////////////////////////////////////////////////////////////
/// \file invoke.hpp
///
/// \brief This internal header provides the definition of the INVOKE overload
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
#ifndef BPSTD_DETAIL_INVOKE_HPP
#define BPSTD_DETAIL_INVOKE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "config.hpp"  // BPSTD_INLINE_VISIBILITY
#include "move.hpp"    // forward
#include <type_traits> // std::true_type, std::false_type, etc
#include <functional>  // std::reference_wrapper

#include <utility>

BPSTD_COMPILER_DIAGNOSTIC_PREAMBLE

namespace bpstd {
  namespace detail {

    template<typename T>
    struct is_reference_wrapper : std::false_type {};

    template<typename U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    template <typename Base, typename T, typename Derived, typename... Args>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(T Base::*pmf, Derived&& ref, Args&&... args)
      noexcept(noexcept((::bpstd::forward<Derived>(ref).*pmf)(::bpstd::forward<Args>(args)...)))
      -> typename std::enable_if<std::is_function<T>::value &&
                                 std::is_base_of<Base, typename std::decay<Derived>::type>::value,
          decltype((::bpstd::forward<Derived>(ref).*pmf)(::bpstd::forward<Args>(args)...))>::type
    {
      return (bpstd::forward<Derived>(ref).*pmf)(bpstd::forward<Args>(args)...);
    }

    template <typename Base, typename T, typename RefWrap, typename... Args>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(T Base::*pmf, RefWrap&& ref, Args&&... args)
      noexcept(noexcept((ref.get().*pmf)(std::forward<Args>(args)...)))
      -> typename std::enable_if<std::is_function<T>::value &&
                          is_reference_wrapper<typename std::decay<RefWrap>::type>::value,
          decltype((ref.get().*pmf)(::bpstd::forward<Args>(args)...))>::type
    {
      return (ref.get().*pmf)(bpstd::forward<Args>(args)...);
    }

    template<typename Base, typename T, typename Pointer, typename... Args>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(T Base::*pmf, Pointer&& ptr, Args&&... args)
      noexcept(noexcept(((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...)))
      -> typename std::enable_if<std::is_function<T>::value &&
                          !is_reference_wrapper<typename std::decay<Pointer>::type>::value &&
                          !std::is_base_of<Base, typename std::decay<Pointer>::type>::value,
          decltype(((*::bpstd::forward<Pointer>(ptr)).*pmf)(::bpstd::forward<Args>(args)...))>::type
    {
      return ((*bpstd::forward<Pointer>(ptr)).*pmf)(bpstd::forward<Args>(args)...);
    }

    template<typename Base, typename T, typename Derived>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(T Base::*pmd, Derived&& ref)
      noexcept(noexcept(std::forward<Derived>(ref).*pmd))
      -> typename std::enable_if<!std::is_function<T>::value &&
                          std::is_base_of<Base, typename std::decay<Derived>::type>::value,
          decltype(::bpstd::forward<Derived>(ref).*pmd)>::type
    {
      return bpstd::forward<Derived>(ref).*pmd;
    }

    template<typename Base, typename T, typename RefWrap>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(T Base::*pmd, RefWrap&& ref)
      noexcept(noexcept(ref.get().*pmd))
      -> typename std::enable_if<!std::is_function<T>::value &&
                                 is_reference_wrapper<typename std::decay<RefWrap>::type>::value,
          decltype(ref.get().*pmd)>::type
    {
      return ref.get().*pmd;
    }

    template<typename Base, typename T, typename Pointer>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(T Base::*pmd, Pointer&& ptr)
      noexcept(noexcept((*std::forward<Pointer>(ptr)).*pmd))
      -> typename std::enable_if<!std::is_function<T>::value &&
                          !is_reference_wrapper<typename std::decay<Pointer>::type>::value &&
                          !std::is_base_of<Base, typename std::decay<Pointer>::type>::value,
          decltype((*::bpstd::forward<Pointer>(ptr)).*pmd)>::type
    {
      return (*bpstd::forward<Pointer>(ptr)).*pmd;
    }

    template<typename F, typename... Args>
    inline BPSTD_INLINE_VISIBILITY constexpr
    auto INVOKE(F&& f, Args&&... args)
        noexcept(noexcept(std::forward<F>(f)(std::forward<Args>(args)...)))
      -> typename std::enable_if<!std::is_member_pointer<typename std::decay<F>::type>::value,
        decltype(::bpstd::forward<F>(f)(::bpstd::forward<Args>(args)...))>::type
    {
      return bpstd::forward<F>(f)(bpstd::forward<Args>(args)...);
    }

    //==========================================================================
    // is_nothrow_invocable
    //==========================================================================

    template <typename Fn, typename...Args>
    struct is_nothrow_invocable
    {
      template <typename Fn2, typename...Args2>
      static auto test( Fn2&&, Args2&&... )
        -> decltype(INVOKE(std::declval<Fn2>(), std::declval<Args2>()...),
                    std::integral_constant<bool,noexcept(INVOKE(std::declval<Fn2>(), std::declval<Args2>()...))>{});

      static auto test(...)
        -> std::false_type;

      using type = decltype(test(std::declval<Fn>(), std::declval<Args>()...));
      static constexpr bool value = type::value;
    };

    //==========================================================================
    // is_invocable
    //==========================================================================

    template<typename Fn, typename...Args>
    struct is_invocable
    {
      template <typename Fn2, typename...Args2>
      static auto test( Fn2&&, Args2&&... )
        -> decltype(INVOKE(std::declval<Fn2>(), std::declval<Args2>()...), std::true_type{});

      static auto test(...)
        -> std::false_type;

      using type = decltype(test(std::declval<Fn>(), std::declval<Args>()...));
      static constexpr bool value = type::value;
    };

    // Used to SFINAE away non-invocable types
    template <bool B, typename Fn, typename...Args>
    struct invoke_result_impl{};

    template <typename Fn, typename...Args>
    struct invoke_result_impl<true, Fn, Args...>{
      using type = decltype(INVOKE(std::declval<Fn>(), std::declval<Args>()...));
    };

    template <typename Fn, typename...Args>
    struct invoke_result
      : invoke_result_impl<is_invocable<Fn,Args...>::value, Fn, Args...>{};

  } // namespace detail
} // namespace bpstd

BPSTD_COMPILER_DIAGNOSTIC_POSTAMBLE

#endif /* BPSTD_DETAIL_INVOKE_HPP */
