/**
 * Copyright © 2013 - 2015 MNMLSTC
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this software except in compliance with the License. You may 
 * obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing 
 * permissions and limitations under the License.
 */
#ifndef CORE_INTERNAL_HPP
#define CORE_INTERNAL_HPP

/* This is a header containing common implementation specific code, to
 * reduce the complexity of the other headers, especially those that are
 * closely intertwined, such as <core/functional.hpp> and <core/type_traits.hpp>
 *
 * Additionally, some of this code is duplicated elsewhere (such as class_of,
 * and meta::identity), but aliases are placed to lessen any impact that this
 * might have.
 */

#include <type_traits>
#include <functional>
#include <utility>
#include <ciso646>

#include "meta.hpp"

namespace core {
inline namespace v2 {
namespace impl {

template <class T, class Void, template <class...> class, class...>
struct make_detect : meta::identity<T> { using value_t = ::std::false_type; };
template <class T, template <class...> class U, class... Args>
struct make_detect<T, meta::deduce<U<Args...>>, U, Args...> :
  meta::identity<U<Args...>>
{ using value_t = ::std::true_type; };

/* extremely useful custom type traits */
template <class T> struct class_of : meta::identity<T> { };
template <class Signature, class T>
struct class_of<Signature T::*> : meta::identity<T> { };

/* aliases */
template <class T> using class_of_t = typename class_of<T>::type;
template <class T> using decay_t = typename ::std::decay<T>::type;
template <class T>
using remove_reference_t = typename ::std::remove_reference<T>::type;
template <bool B, class T = void>
using enable_if_t = typename ::std::enable_if<B, T>::type;

/* is_nothrow_swappable plumbing */
using ::std::declval;
using ::std::swap;

// MSVC 2015 workaround
template <class T, class U>
struct is_swappable_with {
  template <class X, class Y>
  static auto test (void*) noexcept(true) -> decltype(
    swap(declval<X&>(), declval<Y&>())
  );

  template <class, class>
  static void test (...) noexcept(false);

  static constexpr bool value = noexcept(test<T, U>(nullptr));
};

// MSVC 2015 workaround
template <class T, class U>
struct is_noexcept_swappable_with {
  template <
    class X,
    class Y,
    bool B=noexcept(swap(declval<X&>(), declval<Y&>()))
  > static void test (enable_if_t<B>*) noexcept(true);

  template <class, class>
  static void test (...) noexcept(false);

  static constexpr bool value = noexcept(test<T, U>(nullptr));
};

template <class, class, class=void>
struct is_swappable : ::std::false_type { };

template <class T, class U>
struct is_swappable<
  T,
  U,
  meta::deduce<
    is_swappable_with<T, U>,
    is_swappable_with<U, T>
  >
> : ::std::true_type { };

template <class T, class U=T>
struct is_nothrow_swappable : meta::all_t<
  is_swappable<T, U>::value,
  is_noexcept_swappable_with<T, U>::value,
  is_noexcept_swappable_with<U, T>::value
> { };

/*
 * If I can't amuse myself when working with C++ templates, then life isn't
 * worth living. Bury me with my chevrons.
 */
template <class T>
constexpr T&& pass (remove_reference_t<T>& t) noexcept {
  return static_cast<T&&>(t);
}

template <class T>
constexpr T&& pass (remove_reference_t<T>&& t) noexcept {
  return static_cast<T&&>(t);
}

/* INVOKE pseudo-expression plumbing, *much* more simplified than previous
 * versions of Core
 */
struct undefined { constexpr undefined (...) noexcept { } };

/* We get some weird warnings under clang, so we actually give these functions
 * a body to get rid of it.
 */
template <class... Args>
constexpr undefined INVOKE (undefined, Args&&...) noexcept {
  return undefined { };
}

template <class Functor, class... Args>
constexpr auto INVOKE (Functor&& f, Args&&... args) -> enable_if_t<
  not ::std::is_member_pointer<decay_t<Functor>>::value,
  decltype(pass<Functor>(f)(pass<Args>(args)...))
> { return pass<Functor>(f)(pass<Args>(args)...); }

template <class Functor, class... Args>
auto INVOKE (Functor&& f, Args&&... args) -> enable_if_t<
  ::std::is_member_pointer<decay_t<Functor>>::value,
  decltype(::std::mem_fn(pass<Functor>(f))(pass<Args>(args)...))
> { return ::std::mem_fn(pass<Functor>(f))(pass<Args>(args)...); }

template <bool, class...> struct invoke_of { };
template <class... Args> struct invoke_of<true, Args...> :
  meta::identity<decltype(INVOKE(declval<Args>()...))>
{ };

}}} /* namespace core::v2::impl */

#endif /* CORE_INTERNAL_HPP */
