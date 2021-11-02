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
#ifndef CORE_FUNCTIONAL_HPP
#define CORE_FUNCTIONAL_HPP

#include <functional>
#include <tuple>
#include <array>

#include "type_traits.hpp"
#include "utility.hpp"

namespace core {
inline namespace v2 {

template <class T> using is_reference_wrapper = meta::is_specialization_of<
  remove_cv_t<T>,
  ::std::reference_wrapper
>;

template <class F> struct function_traits;

template <class R, class... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)> { };

template <class C, class R>
struct function_traits<R(C::*)> : function_traits<R(C&)> { };

template <class C, class R, class... Args>
struct function_traits<R(C::*)(Args...)> : function_traits<R(C&, Args...)> { };

template <class C, class R, class... Args>
struct function_traits<R(C::*)(Args...) const volatile> :
  function_traits<R(C volatile const&, Args...)>
{ };

template <class C, class R, class... Args>
struct function_traits<R(C::*)(Args...) volatile> :
  function_traits<R(C volatile&, Args...)>
{ };

template <class C, class R, class... Args>
struct function_traits<R(C::*)(Args...) const> :
  function_traits<R(C const&, Args...)>
{ };

template <class R, class... Args>
struct function_traits<R(Args...)> {
  using typelist = meta::list<Args...>;
  using return_type = R;

  using pointer = add_pointer_t<return_type(Args...)>;
  static constexpr auto arity = typelist::size();

  template <::std::size_t N> using argument = meta::get<typelist, N>;
};

template <class F> struct function_traits {
  using functor_type = function_traits<decltype(&decay_t<F>::operator())>;
  using return_type = typename functor_type::return_type;
  using pointer = typename functor_type::pointer;
  static constexpr auto arity = functor_type::arity - 1;
  template <::std::size_t N>
  using argument = typename functor_type::template argument<N>;
};

/* N3727 */
template <class Functor, class... Args>
auto invoke (Functor&& f, Args&&... args) -> enable_if_t<
  ::std::is_member_pointer<decay_t<Functor>>::value,
  result_of_t<Functor&&(Args&&...)>
> { return ::std::mem_fn(f)(core::forward<Args>(args)...); }

template <class Functor, class... Args>
auto invoke (Functor&& f, Args&&... args) -> enable_if_t<
  not ::std::is_member_pointer<decay_t<Functor>>::value,
  result_of_t<Functor&&(Args&&...)>
> { return core::forward<Functor>(f)(core::forward<Args>(args)...); }

template <class F, class T, ::std::size_t... I>
auto apply (F&& f, T&& t, index_sequence<I...>) -> decltype(
  invoke(core::forward<F>(f), ::std::get<I>(core::forward<T>(t))...)
) { return invoke(core::forward<F>(f), ::std::get<I>(core::forward<T>(t))...); }

template <
  class Functor,
  class T,
  class I = make_index_sequence<::std::tuple_size<decay_t<T>>::value>
> auto apply (Functor&& f, T&& t) -> decltype(
  apply(core::forward<Functor>(f), core::forward<T>(t), I { })
) { return apply(core::forward<Functor>(f), core::forward<T>(t), I { }); }

template <class F>
struct apply_functor {
  template <class G>
  explicit apply_functor (G&& g) : f(core::forward<G>(g)) { }

  template <class Applicable>
  auto operator () (Applicable&& args) -> decltype(
    core::apply(core::forward<F>(this->f), core::forward<Applicable>(args))
  ) { return apply(core::forward<F>(f), core::forward<Applicable>(args)); }
private:
  F f;
};

template <class F>
auto make_apply (F&& f) -> apply_functor<F> {
  return apply_functor<F> { core::forward<F>(f) };
}

template <class F>
struct not_fn_functor {
  template <class G>
  explicit not_fn_functor (G&& g) : f(core::forward<G>(g)) { }

  template <class... Args>
  auto operator () (Args&&... args) const -> decltype(
    not (invoke)(::std::declval<F>(), core::forward<Args>(args)...)
  ) { return not (invoke)(f, core::forward<Args>(args)...); }

  template <class... Args>
  auto operator () (Args&&... args) -> decltype(
    not (invoke)(::std::declval<F>(), core::forward<Args>(args)...)
  ) { return not (invoke)(f, core::forward<Args>(args)...); }

private:
  F f;
};

/* Were this C++14, we could just use a lambda with a capture. Oh Well! */
template <class F>
not_fn_functor<decay_t<F>> not_fn (F&& f) {
  return not_fn_functor<decay_t<F>> { core::forward<F>(f) };
}

/* converter function object */
template <class T>
struct converter {
  template <class... Args>
  constexpr T operator () (Args&&... args) const {
    return T(core::forward<Args>(args)...);
  }
};

/* function objects -- arithmetic */
template <class T=void>
struct plus {
  constexpr T operator () (T const& l, T const& r) const { return l + r; }
};

template <class T=void>
struct minus {
  constexpr T operator () (T const& l, T const& r) const { return l - r; }
};

template <class T=void>
struct multiplies {
  constexpr T operator () (T const& l, T const& r) const { return l * r; }
};

template <class T=void>
struct divides {
  constexpr T operator () (T const& l, T const& r) const { return l / r; }
};

template <class T=void>
struct modulus {
  constexpr T operator () (T const& l, T const& r) const { return l % r; }
};

template <class T=void>
struct negate {
  constexpr T operator () (T const& arg) const { return -arg; }
};

/* function objects -- comparisons */
template <class T=void>
struct equal_to {
  constexpr bool operator () (T const& l, T const& r) const { return l == r; }
};

template <class T=void>
struct not_equal_to {
  constexpr bool operator () (T const& l, T const& r) const { return l != r; }
};

template <class T=void>
struct greater_equal {
  constexpr bool operator () (T const& l, T const& r) const { return l >= r; }
};

template <class T=void>
struct less_equal {
  constexpr bool operator () (T const& l, T const& r) const { return l <= r; }
};

template <class T=void>
struct greater {
  constexpr bool operator () (T const& l, T const& r) const { return l > r; }
};

template <class T=void>
struct less {
  constexpr bool operator () (T const& l, T const& r) const { return l < r; }
};

/* function objects -- logical */
template <class T=void>
struct logical_and {
  constexpr bool operator () (T const& l, T const& r) const { return l and r; }
};

template <class T=void>
struct logical_or {
  constexpr bool operator () (T const& l, T const& r) const { return l or r; }
};

template <class T=void>
struct logical_not {
  constexpr bool operator () (T const& arg) const { return not arg; }
};

/* function objects -- bitwise */

template <class T=void>
struct bit_and {
  constexpr bool operator () (T const& l, T const& r) const { return l & r; }
};

template <class T=void>
struct bit_or {
  constexpr bool operator () (T const& l, T const& r) const { return l | r; }
};

template <class T=void>
struct bit_xor {
  constexpr bool operator () (T const& l, T const& r) const { return l ^ r; }
};

template <class T=void>
struct bit_not {
  constexpr bool operator () (T const& arg) const { return ~arg; }
};

/* function objects -- arithmetic specializations */
template <> struct plus<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) + core::forward<U>(u)
  ) { return core::forward<T>(t) + core::forward<U>(u); }
};

template <> struct minus<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) - core::forward<U>(u)
  ) { return core::forward<T>(t) - core::forward<U>(u); }
};

template <> struct multiplies<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) * core::forward<U>(u)
  ) { return core::forward<T>(t) * core::forward<U>(u); }
};

template <> struct divides<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) / core::forward<U>(u)
  ) { return core::forward<T>(t) / core::forward<U>(u); }
};

template <> struct modulus<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) % core::forward<U>(u)
  ) { return core::forward<T>(t) % core::forward<U>(u); }
};

template <> struct negate<void> {
  using is_transparent = void;

  template <class T>
  constexpr auto operator () (T&& t) const -> decltype(core::forward<T>(t)) {
    return core::forward<T>(t);
  }
};

/* function objects -- comparison specialization */
template <> struct equal_to<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) == core::forward<U>(u)
  ) { return core::forward<T>(t) == core::forward<U>(u); }
};

template <> struct not_equal_to<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) != core::forward<U>(u)
  ) { return core::forward<T>(t) != core::forward<U>(u); }
};

template <> struct greater_equal<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) >= core::forward<U>(u)
  ) { return core::forward<T>(t) >= core::forward<U>(u); }
};

template <> struct less_equal<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) <= core::forward<U>(u)
  ) { return core::forward<T>(t) <= core::forward<U>(u); }
};

template <> struct greater<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) > core::forward<U>(u)
  ) { return core::forward<T>(t) > core::forward<U>(u); }
};

template <> struct less<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) < core::forward<U>(u)
  ) { return core::forward<T>(t) < core::forward<U>(u); }
};

/* function objects -- logical specializations */
template <> struct logical_and<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) and core::forward<U>(u)
  ) { return core::forward<T>(t) and core::forward<U>(u); }
};

template <> struct logical_or<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) or core::forward<U>(u)
  ) { return core::forward<T>(t) or core::forward<U>(u); }
};

template <> struct logical_not<void> {
  using is_transparent = void;

  template <class T>
  constexpr auto operator () (T&& t) const -> decltype(
    not core::forward<T>(t)
  ) { return not core::forward<T>(t); }
};

/* function objects -- bitwise specializations */
template <> struct bit_and<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) & core::forward<U>(u)
  ) { return core::forward<T>(t) & core::forward<U>(u); }
};

template <> struct bit_or<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) | core::forward<U>(u)
  ) { return core::forward<T>(t) | core::forward<U>(u); }
};

template <> struct bit_xor<void> {
  using is_transparent = void;

  template <class T, class U>
  constexpr auto operator () (T&& t, U&& u) const -> decltype(
    core::forward<T>(t) ^ core::forward<U>(u)
  ) { return core::forward<T>(t) ^ core::forward<U>(u); }
};

template <> struct bit_not<void> {
  using is_transparent = void;

  template <class T>
  constexpr auto operator () (T&& t) const -> decltype(~core::forward<T>(t)) {
    return ~core::forward<T>(t);
  }
};

/* N3980 Implementation */

}} /* namespace core::v2 */

#endif /* CORE_FUNCTIONAL_HPP */
