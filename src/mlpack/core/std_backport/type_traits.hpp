#ifndef CORE_TYPE_TRAITS_HPP
#define CORE_TYPE_TRAITS_HPP

#include <type_traits>
#include <utility>
#include <tuple>

#include "internal.hpp"

namespace core {
inline namespace v2 {
namespace impl {

/* union used for variant<Ts...> and implementing aligned_union, which is
 * not provided by gcc 4.8.x, but is provided by clang. (aligned_union_t is
 * the only alias missing from <type_traits>)
 */
template <class... Ts> union discriminate;
template <> union discriminate<> { };
template <class T, class... Ts>
union discriminate<T, Ts...> {
  T value;
  discriminate<Ts...> rest;
};

} /* namespace impl */

/* custom type traits and types */
template <class T> using identity_t = typename meta::identity<T>::type;
template <class T> using identity = meta::identity<T>;

/* extracts the class of a member function ponter */
template <class T> using class_of_t = impl::class_of_t<T>;
template <class T> using class_of = impl::class_of<T>;

template <::std::size_t I, class T>
using tuple_element_t = typename ::std::tuple_element<I, T>::type;
template <class T> using tuple_size_t = typename ::std::tuple_size<T>::type;

/* Implementation of N4389 */
template <bool B> using bool_constant = ::std::integral_constant<bool, B>;

template <class...> struct conjunction;
template <class...> struct disjunction;
template <class...> struct negation;

template <class T, class... Ts>
struct conjunction<T, Ts...> :
  bool_constant<T::value and conjunction<Ts...>::value>
{ };
template <> struct conjunction<> : ::std::true_type { };

template <class T, class... Ts>
struct disjunction<T, Ts...> :
  bool_constant<T::value or disjunction<Ts...>::value>
{ };
template <> struct disjunction<> : ::std::false_type { };

template <class T, class... Ts>
struct negation<T, Ts...> :
  bool_constant<not T::value and negation<Ts...>::value>
{ };
template <> struct negation<> : ::std::false_type { };

/* C++ Library Fundamentals V2 TS detection idiom */
template <class... Ts> using void_t = meta::deduce<Ts...>;

struct nonesuch {
  nonesuch (nonesuch const&) = delete;
  nonesuch () = delete;
  ~nonesuch () = delete;
  void operator = (nonesuch const&) = delete;
};

template <class T, template <class...> class U, class... Args>
using detected_or = impl::make_detect<T, void, U, Args...>;

template <template <class...> class T, class... Args>
using detected_t = typename detected_or<nonesuch, T, Args...>::type;

template <class T, template <class...> class U, class... Args>
using detected_or_t = typename detected_or<T, U, Args...>::type;

template <class To, template <class...> class T, class... Args>
using is_detected_convertible = ::std::is_convertible<
  detected_t<T, Args...>,
  To
>;

template <class T, template <class...> class U, class... Args>
using is_detected_same = ::std::is_same<T, detected_t<U, Args...>>;

template <class T, template<class...> class U, class... Args>
using is_detected_convertible = ::std::is_convertible<
  detected_t<U, Args...>,
  T
>;

template <template <class...> class T, class... Args>
using is_detected = typename detected_or<nonesuch, T, Args...>::value_t;

/* forward declaration */
template <::std::size_t, class...> struct aligned_union;
template <class...> struct invokable;
template <class...> struct invoke_of;
template <class T> struct result_of; /* SFINAE result_of */

/* C++14 style aliases for standard traits */
template <class T>
using remove_volatile_t = typename ::std::remove_volatile<T>::type;

template <class T>
using remove_const_t = typename ::std::remove_const<T>::type;
template <class T> using remove_cv_t = typename ::std::remove_cv<T>::type;

template <class T>
using add_volatile_t = typename ::std::add_volatile<T>::type;
template <class T> using add_const_t = typename ::std::add_const<T>::type;
template <class T> using add_cv_t = typename ::std::add_cv<T>::type;

template <class T>
using add_lvalue_reference_t = typename ::std::add_lvalue_reference<T>::type;

template <class T>
using add_rvalue_reference_t = typename ::std::add_rvalue_reference<T>::type;

template <class T>
using remove_reference_t = typename ::std::remove_reference<T>::type;

template <class T>
using remove_pointer_t = typename ::std::remove_pointer<T>::type;

template <class T> using add_pointer_t = typename ::std::add_pointer<T>::type;

template <class T>
using make_unsigned_t = typename ::std::make_unsigned<T>::type;
template <class T> using make_signed_t = typename ::std::make_signed<T>::type;

template <class T>
using remove_extent_t = typename ::std::remove_extent<T>::type;

template <class T>
using remove_all_extents_t = typename ::std::remove_all_extents<T>::type;

template <
  ::std::size_t Len,
  ::std::size_t Align = alignof(typename ::std::aligned_storage<Len>::type)
> using aligned_storage_t = typename ::std::aligned_storage<Len, Align>::type;

template <::std::size_t Len, class... Types>
using aligned_union_t = typename aligned_union<Len, Types...>::type;

template <class T> using decay_t = impl::decay_t<T>;

template <bool B, class T = void>
using enable_if_t = typename ::std::enable_if<B, T>::type;

template <bool B, class T, class F>
using conditional_t = typename ::std::conditional<B, T, F>::type;

template <class T>
using underlying_type_t = typename ::std::underlying_type<T>::type;

template <::std::size_t Len, class... Types>
struct aligned_union {
  using union_type = impl::discriminate<Types...>;
  static constexpr ::std::size_t size () noexcept {
    return Len > sizeof(union_type) ? Len : sizeof(union_type);
  }

  static constexpr ::std::size_t alignment_value = alignof(
    impl::discriminate<Types...>
  );

  using type = aligned_storage_t<
    (Len > sizeof(union_type) ? Len : sizeof(union_type)),
    alignment_value
  >;
};

/* custom type trait specializations */
template <class... Args> using invoke_of_t = typename invoke_of<Args...>::type;

template <class... Args>
struct invokable : meta::none_t<
  std::is_same<
    decltype(impl::INVOKE(::std::declval<Args>()...)),
    impl::undefined
  >::value
> { };

template <class... Args> struct invoke_of :
  impl::invoke_of<invokable<Args...>::value, Args...>
{ };

template <class F, class... Args>
struct result_of<F(Args...)> : invoke_of<F, Args...> { };

template <class T> using result_of_t = typename result_of<T>::type;

template <class... Ts> struct common_type;

template <class T> struct common_type<T> : identity<decay_t<T>> { };
template <class T, class U>
struct common_type<T, U> : identity<
  decay_t<decltype(true ? ::std::declval<T>() : ::std::declval<U>())>
> { };

template <class T, class U, class... Ts>
struct common_type<T, U, Ts...> : identity<
  typename common_type<
    typename common_type<T, U>::type,
    Ts...
  >::type
> { };

template <class... T> using common_type_t = typename common_type<T...>::type;

/* is_null_pointer */
template <class T> struct is_null_pointer : ::std::false_type { };

template <>
struct is_null_pointer<add_cv_t<::std::nullptr_t>> : ::std::true_type { };
template <>
struct is_null_pointer<::std::nullptr_t volatile> : ::std::true_type { };
template <>
struct is_null_pointer<::std::nullptr_t const> : ::std::true_type { };
template <>
struct is_null_pointer<::std::nullptr_t> : ::std::true_type { };

/* is_nothrow_swappable - N4426 (implemented before paper was proposed) */
template <class T, class U=T>
using is_nothrow_swappable = impl::is_nothrow_swappable<T, U>;

/* propagates const or volatile without using the name propagate :) */
template <class T, class U>
struct transmit_volatile : ::std::conditional<
  ::std::is_volatile<T>::value,
  add_volatile_t<U>,
  U
> { };

template <class T, class U>
struct transmit_const : ::std::conditional<
  ::std::is_const<T>::value,
  add_const_t<U>,
  U
> { };

template <class T, class U>
struct transmit_cv : transmit_volatile<
  T, typename transmit_const<T, U>::type
> { };

template <class T, class U>
using transmit_volatile_t = typename transmit_volatile<T, U>::type;

template <class T, class U>
using transmit_const_t = typename transmit_const<T, U>::type;

template <class T, class U>
using transmit_cv_t = typename transmit_cv<T, U>::type;

}} /* namespace core::v2 */

#endif /* CORE_TYPE_TRAITS_HPP */
