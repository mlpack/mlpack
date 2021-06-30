////////////////////////////////////////////////////////////////////////////////
/// \file type_traits.hpp
///
/// \brief This header provides definitions from the C++ header <type_traits>
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
#ifndef BPSTD_TYPE_TRAITS_HPP
#define BPSTD_TYPE_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "detail/config.hpp"
#include "detail/move.hpp"   // move, forward
#include "detail/invoke.hpp" // detail::INVOKE

#include <type_traits>
#include <cstddef> // std::size_t

BPSTD_COMPILER_DIAGNOSTIC_PREAMBLE

// GCC versions prior to gcc-5 did not implement the type traits for triviality
// in completion. Several traits are implemented under a different names from
// pre-standardization, such as 'has_trivial_copy_destructor' instead of
// 'is_trivially_destructible'. However, most of these cannot be implemented
// without compiler support.
//
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html#status.iso.2014
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ < 5
# define BPSTD_HAS_TRIVIAL_TYPE_TRAITS 0
#else
# define BPSTD_HAS_TRIVIAL_TYPE_TRAITS 1
#endif

namespace bpstd {

  //============================================================================
  // Type constants
  //============================================================================

  template <typename T, T V>
  using integral_constant = std::integral_constant<T, V>;

  template <bool B>
  using bool_constant = integral_constant<bool,B>;

  using std::true_type;
  using std::false_type;

  template <typename T>
  struct type_identity {
    using type = T;
  };

  namespace detail {
    template <typename T>
    struct make_void : type_identity<void>{};
  } // namespace detail

  template <typename T>
  using void_t = typename detail::make_void<T>::type;

  //============================================================================
  // Metafunctions
  //============================================================================

  template <bool B, typename T = void>
  using enable_if = std::enable_if<B, T>;

  template <bool B, typename T = void>
  using enable_if_t = typename enable_if<B, T>::type;

  //----------------------------------------------------------------------------

  template <bool B, typename True, typename False>
  using conditional = std::conditional<B, True, False>;

  template <bool B, typename True, typename False>
  using conditional_t = typename conditional<B, True, False>::type;

  /// \brief Type trait to determine the bool_constant from a logical
  ///        AND operation of other bool_constants
  ///
  /// The result is aliased as \c ::value
  template<typename...>
  struct conjunction;

  template<typename B1>
  struct conjunction<B1> : B1{};

  template<typename B1, typename... Bn>
  struct conjunction<B1, Bn...>
    : conditional_t<B1::value, conjunction<Bn...>, B1>{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template<typename...Bs>
  BPSTD_CPP17_INLINE constexpr auto disconjunction_v = conjunction<Bs...>::value;
#endif

  //----------------------------------------------------------------------------

  /// \brief Type trait to determine the \c bool_constant from a logical
  ///        OR operations of other bool_constant
  ///
  /// The result is aliased as \c ::value
  template<typename...>
  struct disjunction : false_type { };

  template<typename B1>
  struct disjunction<B1> : B1{};

  template<typename B1, typename... Bn>
  struct disjunction<B1, Bn...>
    : conditional_t<B1::value != false, B1, disjunction<Bn...>>{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template<typename...Bs>
  BPSTD_CPP17_INLINE constexpr auto disjunction_v = disjunction<Bs...>::value;
#endif

  //----------------------------------------------------------------------------

  /// \brief Utility metafunction for negating a bool_constant
  ///
  /// The result is aliased as \c ::value
  ///
  /// \tparam B the constant
  template<typename B>
  struct negation : bool_constant<!static_cast<bool>(B::value)>{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template<typename B>
  BPSTD_CPP17_INLINE constexpr auto negation_v = negation<B>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename Fn, typename...Args>
  using invoke_result = detail::invoke_result<Fn,Args...>;

  template <typename Fn, typename...Args>
  using invoke_result_t = typename invoke_result<Fn,Args...>::type;

  //----------------------------------------------------------------------------

  namespace detail {

    template <bool IsInvocable, typename R, typename Fn, typename...Args>
    struct is_invocable_return : std::is_convertible<invoke_result_t<Fn,Args...>, R>{};

    template <typename R, typename Fn, typename...Args>
    struct is_invocable_return<false, R, Fn, Args...> : false_type{};

  } // namespace detail

  template <typename Fn, typename...Args>
  using is_invocable = detail::is_invocable<Fn, Args...>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template<typename Fn, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_invocable_v = is_invocable<Fn, Args...>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename R, typename Fn, typename...Args>
  struct is_invocable_r
    : detail::is_invocable_return<is_invocable<Fn,Args...>::value, R, Fn, Args...>{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename R, typename Fn, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_invocable_r_v = is_invocable_r<R, Fn, Args...>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename Fn, typename...Args>
  using is_nothrow_invocable = detail::is_nothrow_invocable<Fn, Args...>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template<typename Fn, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_invocable_v = is_nothrow_invocable<Fn, Args...>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename R, typename Fn, typename...Args>
  struct is_nothrow_invocable_r
    : detail::is_invocable_return<is_nothrow_invocable<Fn,Args...>::value, R, Fn, Args...>{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename R, typename Fn, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_invocable_r_v
    = is_nothrow_invocable_r<R, Fn, Args...>::value;
#endif

  //============================================================================
  // Type categories
  //============================================================================

  template <typename T>
  using is_void = std::is_void<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_void_v = is_void<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  struct is_null_pointer : false_type{};

  template <>
  struct is_null_pointer<decltype(nullptr)> : true_type{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_null_pointer_v = is_null_pointer<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_array = std::is_array<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_array_v = is_array<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_pointer = std::is_pointer<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_pointer_v = is_pointer<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_enum = std::is_enum<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_enum_v = is_enum<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_union = std::is_union<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_union_v = is_union<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_class = std::is_class<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_class_v = is_class<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_function = std::is_function<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_function_v = is_function<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_object = std::is_object<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_object_v = is_object<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_scalar = std::is_scalar<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_scalar_v = is_scalar<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_compound = std::is_compound<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_compound_v = is_compound<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_integral = std::is_integral<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_integral_v = is_integral<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_floating_point = std::is_floating_point<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_floating_point_v = is_floating_point<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_fundamental = std::is_fundamental<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_fundamental_v = is_fundamental<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_arithmetic = std::is_arithmetic<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_arithmetic_v = is_arithmetic<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_reference = std::is_reference<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_reference_v = is_reference<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_lvalue_reference = std::is_lvalue_reference<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_lvalue_reference_v = is_lvalue_reference<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_rvalue_reference = std::is_rvalue_reference<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_rvalue_reference_v = is_rvalue_reference<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_member_pointer = std::is_member_pointer<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_member_pointer_v = is_member_pointer<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_member_object_pointer = std::is_member_object_pointer<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_member_object_pointer_v = is_member_object_pointer<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_member_function_pointer = std::is_member_function_pointer<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_member_function_pointer_v = is_member_function_pointer<T>::value;
#endif

  //============================================================================
  // Type properties
  //============================================================================

  template <typename T>
  using is_const = std::is_const<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_const_v = is_const<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_volatile = std::is_volatile<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_volatile_v = is_volatile<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_empty = std::is_empty<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_empty_v = is_empty<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_polymorphic = std::is_polymorphic<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_polymorphic_v = is_polymorphic<T>::value;
#endif

  //----------------------------------------------------------------------------

#if __cplusplus >= 201402L
  // is_final is only defined in C++14
  template <typename T>
  struct is_final : std::is_final<T>{};
#else
  // is_final requires compiler-support to implement.
  // Without this support, the best we can do is require explicit
  // specializations of 'is_final' for any types that are known to be final
  template <typename T>
  struct is_final : false_type{};
#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_final_v = is_final<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_abstract = std::is_abstract<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_abstract_v = is_abstract<T>::value;
#endif

  //----------------------------------------------------------------------------

  // is_aggregate is only defined in C++17
#if __cplusplus >= 201703L

  template <typename T>
  using is_aggregate = std::is_aggregate<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_aggregate_v = is_aggregate<T>::value;
#endif
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_trivial = std::is_trivial<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivial_v = is_trivial<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_copyable = std::is_trivially_copyable<T>;

#else

  // std::is_trivially_copyable is not implemented in gcc < 5, and unfortunately
  // can't be implemented without compiler intrinsics. This definition is
  // left out to avoid problems
  template <typename T>
  using is_trivially_copyable = false_type;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_copyable_v = is_trivially_copyable<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_standard_layout = std::is_standard_layout<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_standard_layout_v = is_standard_layout<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_literal_type = std::is_literal_type<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_literal_type_v = is_literal_type<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_pod = std::is_pod<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_pod_v = is_pod<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_signed = std::is_signed<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_signed_v = is_signed<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_unsigned = std::is_unsigned<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_unsigned_v = is_unsigned<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  struct is_bounded_array : false_type{};

  template <typename T, std::size_t N>
  struct is_bounded_array<T[N]> : true_type{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_bounded_array_v = is_bounded_array<T>::value;
#endif

  //----------------------------------------------------------------------------


  template <typename T>
  struct is_unbounded_array : false_type{};

  template <typename T>
  struct is_unbounded_array<T[]> : true_type{};

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_unbounded_array_v = is_unbounded_array<T>::value;
#endif

  //----------------------------------------------------------------------------

  // has_unique_object_representation only defined in C++17
#if __cplusplus >= 201703L
  template <typename T>
  using has_unique_object_representations = std::has_unique_object_representations<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto has_unique_object_representations_v = has_unique_object_representations<T>::value;
#endif
#endif

  //============================================================================
  // Type Modification
  //============================================================================

  template <typename T>
  using remove_cv = std::remove_cv<T>;

  template <typename T>
  using remove_cv_t = typename remove_cv<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_const = std::remove_const<T>;

  template <typename T>
  using remove_const_t = typename remove_const<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_volatile = std::remove_volatile<T>;

  template <typename T>
  using remove_volatile_t = typename remove_volatile<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using add_cv = std::add_cv<T>;

  template <typename T>
  using add_cv_t = typename add_cv<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using add_const = std::add_const<T>;

  template <typename T>
  using add_const_t = typename add_const<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using add_volatile = std::add_volatile<T>;

  template <typename T>
  using add_volatile_t = typename add_volatile<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using make_signed = std::make_signed<T>;

  template <typename T>
  using make_signed_t = typename make_signed<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using make_unsigned = std::make_unsigned<T>;

  template <typename T>
  using make_unsigned_t = typename make_unsigned<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_reference = std::remove_reference<T>;

  template <typename T>
  using remove_reference_t = typename remove_reference<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using add_lvalue_reference = std::add_lvalue_reference<T>;

  template <typename T>
  using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using add_rvalue_reference = std::add_rvalue_reference<T>;

  template <typename T>
  using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_pointer = std::remove_pointer<T>;

  template <typename T>
  using remove_pointer_t = typename remove_pointer<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using add_pointer = std::add_pointer<T>;

  template <typename T>
  using add_pointer_t = typename add_pointer<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_extent = std::remove_extent<T>;

  template <typename T>
  using remove_extent_t = typename remove_extent<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_all_extents = std::remove_all_extents<T>;

  template <typename T>
  using remove_all_extents_t = typename remove_all_extents<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using remove_cvref = remove_cv<remove_reference_t<T>>;

  template <typename T>
  using remove_cvref_t = typename remove_cvref<T>::type;

  //============================================================================
  // Type Transformation
  //============================================================================

  template <std::size_t Size, std::size_t Align>
  struct aligned_storage
  {
    struct type {
      alignas(Align) char storage[Size];
    };
  };

  template <std::size_t Size, std::size_t Align>
  using aligned_storage_t = typename aligned_storage<Size, Align>::type;

  //----------------------------------------------------------------------------

  namespace detail {

    template <std::size_t...Sizes>
    struct largest;

    template <std::size_t Size0, std::size_t Size1, std::size_t...Sizes>
    struct largest<Size0, Size1, Sizes...>
      : largest<(Size0 > Size1 ? Size0 : Size1), Sizes...>{};

    template <std::size_t Size0>
    struct largest<Size0> : integral_constant<std::size_t,Size0>{};

  } // namespace detail

  // gcc < 5 does not implement 'std::aligned_union', despite it being a type
  // in the C++11 standard -- so it's implemented here to ensure that its
  // available.
  template <std::size_t Len, typename... Ts>
  struct aligned_union
  {
    static constexpr std::size_t alignment_value = detail::largest<alignof(Ts)...>::value;

    struct type
    {
      alignas(alignment_value) char buffer[detail::largest<Len, sizeof(Ts)...>::value];
    };
  };

  template <std::size_t Len, typename... Ts>
  constexpr std::size_t aligned_union<Len,Ts...>::alignment_value;

  template <std::size_t Len, typename...Ts>
  using aligned_union_t = typename aligned_union<Len, Ts...>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using decay = std::decay<T>;

  template <typename T>
  using decay_t = typename decay<T>::type;

  //----------------------------------------------------------------------------

  template <typename...Ts>
  using common_type = std::common_type<Ts...>;

  template <typename...Ts>
  using common_type_t = typename common_type<Ts...>::type;

  //----------------------------------------------------------------------------

  namespace detail {

    template <bool IsEnum, typename T>
    struct underlying_type_impl : type_identity<T>{};

    template <typename T>
    struct underlying_type_impl<false, T>{};

  } // namespace detail

  template <typename T>
  struct underlying_type : detail::underlying_type_impl<is_enum<T>::value, T>{};

  template <typename T>
  using underlying_type_t = typename underlying_type<T>::type;

  //----------------------------------------------------------------------------

  template <typename T>
  using result_of = std::result_of<T>;

  template <typename T>
  using result_of_t = typename result_of<T>::type;

  //============================================================================
  // Supported Operations
  //============================================================================

  template <typename T, typename...Args>
  using is_constructible = std::is_constructible<T, Args...>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_constructible_v = is_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T, typename...Args>
  using is_trivially_constructible = std::is_trivially_constructible<T, Args...>;

#else

  // std::is_trivially_constructible is not implemented in gcc < 5, and
  // there exists no utilities to implement it in the language without extensions.
  // This is left defined to false_type so that the trait may be used, despite
  // yielding incorrect results
  template <typename T, typename...Args>
  using is_trivially_constructible = false_type;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_constructible_v = is_trivially_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T, typename...Args>
  using is_nothrow_constructible = std::is_nothrow_constructible<T, Args...>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename...Args>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_constructible_v = is_nothrow_constructible<T, Args...>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_default_constructible = std::is_default_constructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_default_constructible_v = is_default_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_default_constructible = std::is_trivially_default_constructible<T>;

#else

  // std::is_trivially_default_constructible is not implemented in gcc < 5,
  // however there exists a non-standard
  // 'std::has_trivial_default_constructor' which performs a similar check
  template <typename T>
  using is_trivially_default_constructible = std::has_trivial_default_constructor<T>;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_default_constructible_v = is_trivially_default_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_nothrow_default_constructible = std::is_nothrow_default_constructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_default_constructible_v = is_nothrow_default_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_copy_constructible = std::is_copy_constructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_copy_constructible_v = is_copy_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_copy_constructible = std::is_trivially_copy_constructible<T>;

#else

  // std::is_trivially_copy_constructible is not implemented in gcc < 5,
  // however there exists a non-standard
  // 'std::has_trivial_copy_constructor' which performs a similar check
  template <typename T>
  using is_trivially_copy_constructible = std::has_trivial_copy_constructor<T>;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_copy_constructible_v = is_trivially_copy_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_nothrow_copy_constructible = std::is_nothrow_copy_constructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_copy_constructible_v = is_nothrow_copy_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_move_constructible = std::is_move_constructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_move_constructible_v = is_move_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_move_constructible = std::is_trivially_move_constructible<T>;

#else

  // std::is_trivially_move_constructible is not implemented in gcc < 5, and
  // there exists no utilities to implement it in the language without extensions.
  // This is left defined to false_type so that the trait may be used, despite
  // yielding incorrect results
  template <typename T>
  using is_trivially_move_constructible = false_type;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_move_constructible_v = is_trivially_move_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_nothrow_move_constructible = std::is_nothrow_move_constructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_move_constructible_v = is_nothrow_move_constructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T, typename U>
  using is_assignable = std::is_assignable<T, U>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_assignable_v = is_assignable<T, U>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T, typename U>
  using is_trivially_assignable = std::is_trivially_assignable<T, U>;

#else

  // std::is_trivially_assignable is not implemented in gcc < 5, and
  // there exists no utilities to implement it in the language without extensions.
  // This is left defined to false_type so that the trait may be used, despite
  // yielding incorrect results
  template <typename T, typename U>
  using is_trivially_assignable = false_type;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_assignable_v = is_trivially_assignable<T, U>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T, typename U>
  using is_nothrow_assignable = std::is_nothrow_assignable<T, U>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_assignable_v = is_nothrow_assignable<T, U>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_copy_assignable = std::is_copy_assignable<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_copy_assignable_v = is_copy_assignable<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_copy_assignable = std::is_trivially_copy_assignable<T>;

#else

  // std::is_trivially_copy_assignable is not implemented in gcc < 5,
  // however there exists a non-standard
  // 'std::has_trivial_copy_assign' which performs a similar check
  template <typename T>
  using is_trivially_copy_assignable = std::has_trivial_copy_assign<T>;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_copy_assignable_v = is_trivially_copy_assignable<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_nothrow_copy_assignable = std::is_nothrow_copy_assignable<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_copy_assignable_v = is_nothrow_copy_assignable<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_move_assignable = std::is_move_assignable<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_move_assignable_v = is_move_assignable<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_move_assignable = std::is_trivially_move_assignable<T>;

#else

  // std::is_trivially_move_assignable is not implemented in gcc < 5, and
  // there exists no utilities to implement it in the language without extensions.
  // This is left defined to false_type so that the trait may be used, despite
  // yielding incorrect results
  template <typename T>
  using is_trivially_move_assignable = false_type;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_move_assignable_v = is_trivially_move_assignable<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_nothrow_move_assignable = std::is_nothrow_move_assignable<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_move_assignable_v = is_nothrow_move_assignable<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_destructible = std::is_destructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_destructible_v = is_destructible<T>::value;
#endif

  //----------------------------------------------------------------------------

#if BPSTD_HAS_TRIVIAL_TYPE_TRAITS

  template <typename T>
  using is_trivially_destructible = std::is_trivially_destructible<T>;

#else

  // std::is_trivially_destructible is not implemented in gcc < 5, however there
  // exists a non-standard '__has_trivial_destructor' which performs a
  // similar check
  template <typename T>
  using is_trivially_destructible = bool_constant<(__has_trivial_destructor(T))>;

#endif

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_trivially_destructible_v = is_trivially_destructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using is_nothrow_destructible = std::is_nothrow_destructible<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_destructible_v = is_nothrow_destructible<T>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename T>
  using has_virtual_destructor = std::has_virtual_destructor<T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto has_virtual_destructor_v = has_virtual_destructor<T>::value;
#endif

  //============================================================================
  // Relationship
  //============================================================================

  template <typename T, typename U>
  using is_same = std::is_same<T, U>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_same_v = is_same<T, U>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename Base, typename Derived>
  using is_base_of = std::is_base_of<Base,Derived>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename Base, typename Derived>
  BPSTD_CPP17_INLINE constexpr auto is_base_of_v = is_base_of<Base,Derived>::value;
#endif

  //----------------------------------------------------------------------------

  template <typename From, typename To>
  using is_convertible = std::is_convertible<From, To>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename From, typename To>
  BPSTD_CPP17_INLINE constexpr auto is_convertible_v = is_convertible<From, To>::value;
#endif

  //----------------------------------------------------------------------------

  namespace detail {

    template <bool IsConvertible, typename From, typename To>
    struct is_nothrow_convertible_impl : false_type{};

    template <typename From, typename To>
    struct is_nothrow_convertible_impl<true, From, To>
    {
      static void test(To) noexcept;

      BPSTD_CPP17_INLINE static constexpr auto value =
        noexcept(test(std::declval<From>()));
    };

  } // namespace detail

  template <typename From, typename To>
  using is_nothrow_convertible = bool_constant<
    detail::is_nothrow_convertible_impl<is_convertible<From,To>::value,From,To>::value
  >;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename From, typename To>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_convertible_v = is_nothrow_convertible<From, To>::value;
#endif

  namespace detail {
    namespace adl_swap {

      void swap();

      //------------------------------------------------------------------------

      template <typename T, typename U>
      struct is_std_swappable_with : false_type{};

      template <typename T>
      struct is_std_swappable_with<T,T>
        : conjunction<
            is_move_constructible<remove_cvref_t<remove_extent_t<T>>>,
            is_move_assignable<remove_cvref_t<remove_extent_t<T>>>,
            is_lvalue_reference<T>
          >{};

      template <typename T, typename U>
      struct is_nothrow_std_swappable_with : false_type{};

      template <typename T>
      struct is_nothrow_std_swappable_with<T,T>
        : conjunction<
            is_nothrow_move_constructible<remove_cvref_t<remove_extent_t<T>>>,
            is_nothrow_move_assignable<remove_cvref_t<remove_extent_t<T>>>,
            is_lvalue_reference<T>
          >{};

      //------------------------------------------------------------------------
#if !defined(_MSC_FULL_VER) || _MSC_FULL_VER >= 191426428

      template <typename T, typename U>
      using detect_adl_swap = decltype(swap(std::declval<T>(), std::declval<U>()));

      template <typename T, typename U, template <typename, typename> class Op, typename = void>
      struct is_adl_swappable_with : false_type{};

      template <typename T, typename U, template <typename, typename> class Op>
      struct is_adl_swappable_with<T,U,Op, void_t<Op<T,U>>>
        : true_type{};

      template <typename T, typename U, bool IsSwappable = is_adl_swappable_with<T,U,detect_adl_swap>::value>
      struct is_nothrow_adl_swappable_with : false_type{};

      template <typename T, typename U>
      struct is_nothrow_adl_swappable_with<T,U, true>
        : bool_constant<noexcept(swap(std::declval<T>(), std::declval<U>()))>{};
#endif

    } // namespace adl_swap

#if defined(_MSC_FULL_VER) && _MSC_FULL_VER < 191426428

    // MSVC 2017 15.7 or above is required for expression SFINAE.
    // I'm not sure if 'is_swappable_with' is properly implementable without
    // it, since we need to test calling of 'swap' unqualified.
    // For now, the best we can do is test whether std::swap works, until a
    // more full-featured compiler is used.

    template <typename T, typename U>
    struct is_swappable_with
      : adl_swap::is_std_swappable_with<T,U>{};

    template <typename T, typename U>
    struct is_nothrow_swappable_with
      : adl_swap::is_nothrow_std_swappable_with<T,U>{};

#else

    template <typename T, typename U>
    struct is_swappable_with
      : conditional_t<adl_swap::is_adl_swappable_with<T,U, adl_swap::detect_adl_swap>::value,
          adl_swap::is_adl_swappable_with<T,U, adl_swap::detect_adl_swap>,
          adl_swap::is_std_swappable_with<T,U>
        >{};

    template <typename T, typename U>
    struct is_nothrow_swappable_with
      : conditional_t<adl_swap::is_adl_swappable_with<T,U, adl_swap::detect_adl_swap>::value,
          adl_swap::is_nothrow_adl_swappable_with<T,U>,
          adl_swap::is_nothrow_std_swappable_with<T,U>
        >{};

#endif

  } // namespace detail

  template <typename T, typename U>
  using is_swappable_with = detail::is_swappable_with<remove_cvref_t<T>&,remove_cvref_t<U>&>;

  template <typename T>
  using is_swappable = is_swappable_with<T,T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_swappable_with_v = is_swappable_with<T, U>::value;

  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_swappable_v = is_swappable<T>::value;
#endif

  template <typename T, typename U>
  using is_nothrow_swappable_with = detail::is_nothrow_swappable_with<remove_cvref_t<T>&,remove_cvref_t<U>&>;

  template <typename T>
  using is_nothrow_swappable = is_nothrow_swappable_with<T,T>;

#if BPSTD_HAS_TEMPLATE_VARIABLES
  template <typename T, typename U>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_swappable_with_v = is_nothrow_swappable_with<T, U>::value;

  template <typename T>
  BPSTD_CPP17_INLINE constexpr auto is_nothrow_swappable_v = is_nothrow_swappable_with<T, T>::value;
#endif

} // namespace bpstd

BPSTD_COMPILER_DIAGNOSTIC_POSTAMBLE

#endif /* BPSTD_TYPE_TRAITS_HPP */
