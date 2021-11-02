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
#ifndef CORE_UTILITY_HPP
#define CORE_UTILITY_HPP

#include <functional>

#include <cstddef>

#include "type_traits.hpp"

namespace core {
inline namespace v2 {

template <class T>
constexpr T&& forward (remove_reference_t<T>& t) noexcept {
  return static_cast<T&&>(t);
}

template <class T>
constexpr T&& forward (remove_reference_t<T>&& t) noexcept {
  return static_cast<T&&>(t);
}

template <class T>
constexpr auto move (T&& t) noexcept -> decltype(
  static_cast<remove_reference_t<T>&&>(t)
) { return static_cast<remove_reference_t<T>&&>(t); }


template <class T, T... I>
using integer_sequence = meta::integer_sequence<T, I...>;

template <::std::size_t... I>
using index_sequence = integer_sequence<::std::size_t, I...>;

template <class T, T N>
using make_integer_sequence = typename meta::iota<T, N, N>::type;

template <::std::size_t N>
using make_index_sequence = make_integer_sequence<::std::size_t, N>;

template <class... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

template <::std::size_t N, class T, class... Ts>
constexpr auto value_at (T&& value, Ts&&...) -> enable_if_t<
  N == 0 and N < (sizeof...(Ts) + 1),
  decltype(::core::forward<T>(value))
> { return ::core::forward<T>(value); }

template <::std::size_t N, class T, class... Ts>
constexpr auto value_at (T&&, Ts&&... values) -> enable_if_t<
  N != 0 and N < (sizeof...(Ts) + 1),
  meta::get<meta::list<T, Ts...>, N>
> { return value_at<N - 1, Ts...>(::core::forward<Ts>(values)...); }

template <class Callable>
struct scope_guard final {

  static_assert(
    ::std::is_nothrow_move_constructible<Callable>::value,
    "Given type must be nothrow move constructible"
  );

  explicit scope_guard (Callable callable) noexcept :
    callable { ::core::move(callable) },
    dismissed { false }
  { }

  scope_guard (scope_guard const&) = delete;
  scope_guard (scope_guard&&) = default;
  scope_guard () = delete;
  ~scope_guard () noexcept { if (not this->dismissed) { callable(); } }

  scope_guard& operator = (scope_guard const&) = delete;
  scope_guard& operator = (scope_guard&&) = default;

  void dismiss () noexcept { this->dismissed = true; }

private:
  Callable callable;
  bool dismissed;
};

template <class Callable>
auto make_scope_guard(Callable&& callable) -> scope_guard<decay_t<Callable>> {
  return scope_guard<decay_t<Callable>> {
    ::core::forward<Callable>(callable)
  };
}

template <class T, class U=T>
T exchange (T& obj, U&& value) noexcept(
  meta::all<
    ::std::is_nothrow_move_constructible<T>,
    ::std::is_nothrow_assignable<add_lvalue_reference_t<T>, U>
  >()
) {
  T old = ::core::move(obj);
  obj = ::core::forward<U>(value);
  return old;
}

inline ::std::uintptr_t as_int (void const* ptr) noexcept {
  return reinterpret_cast<::std::uintptr_t>(ptr);
}

template <class T>
void const* as_void (T const* ptr) { return static_cast<void const*>(ptr); }

template <class T>
void* as_void (T* ptr) { return static_cast<void*>(ptr); }

template <class T>
void const* as_void (T const& ref) { return as_void(::std::addressof(ref)); }

template <class T>
void* as_void (T& ref) { return as_void(::std::addressof(ref)); }

template <class E>
constexpr auto as_under(E e) noexcept -> meta::when<
  std::is_enum<E>::value,
  underlying_type_t<E>
> { return static_cast<underlying_type_t<E>>(e); }

template <class T>
struct capture final {
  static_assert(::std::is_move_constructible<T>::value, "T must be movable");
  using value_type = T;
  using reference = add_lvalue_reference_t<value_type>;
  using pointer = add_pointer_t<value_type>;

  capture (T&& data) : data { core::move(data) } { }

  capture (capture&&) = default;
  capture (capture& that) : data { core::move(that.data) } { }
  capture () = delete;

  capture& operator = (capture const&) = delete;
  capture& operator = (capture&&) = delete;

  operator reference () const noexcept { return this->get(); }
  reference operator * () const noexcept { return this->get(); }
  pointer operator -> () const noexcept {
    return ::std::addressof(this->get());
  }

  reference get () const noexcept { return this->data; }

private:
  value_type data;
};

template <class T>
auto make_capture (remove_reference_t<T>& ref) -> capture<T> {
  return capture<T> { core::move(ref) };
}

template <class T>
auto make_capture (remove_reference_t<T>&& ref) -> capture<T> {
  return capture<T> { core::move(ref) };
}

struct erased_type { };

}} /* namespace core::v2 */

#endif /* CORE_UTILITY_HPP */
