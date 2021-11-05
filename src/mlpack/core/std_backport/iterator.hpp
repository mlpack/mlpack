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
#ifndef CORE_ITERATOR_HPP
#define CORE_ITERATOR_HPP

#include <functional>
#include <iterator>
#include <ostream>

#include "type_traits.hpp"
#include "utility.hpp"

namespace core {
inline namespace v2 {

/* capacity */
template <class Container>
constexpr auto size (Container const& container) noexcept -> decltype(
  container.size()
) { return container.size(); }

template <class T, ::std::size_t N>
constexpr ::std::size_t size (T const (&)[N]) noexcept { return N; }

template <class Container>
constexpr bool empty (Container const& container) noexcept {
  return container.empty();
}

template <class T, std::size_t N>
constexpr bool empty (T const (&)[N]) noexcept { return false; }

/* element access */
template <class Container>
constexpr auto front (Container const& container) -> decltype(
  container.front()
) { return container.front(); }

template <class Container>
constexpr auto front (Container& container) -> decltype(container.front()) {
  return container.front();
}

template <class T, ::std::size_t N>
constexpr T const& front (T const (&array)[N]) noexcept { return array[0]; }

template <class T, ::std::size_t N>
constexpr T& front (T (&array)[N]) noexcept { return array[0]; }

template <class Container>
constexpr auto back (Container const& container) -> decltype(
  container.back()
) { return container.back(); }

template <class Container>
constexpr auto back (Container& container) -> decltype(container.back()) {
  return container.back();
}

template <class T, ::std::size_t N>
constexpr T const& back (T const (&array)[N]) noexcept { return array[N - 1]; }

template <class T, ::std::size_t N>
constexpr T& back (T (&array)[N]) noexcept { return array[N - 1]; }

/* data access */
template <class Container>
constexpr auto data (Container const& container) noexcept -> decltype(
  container.data()
) { return container.data(); }

template <class Container>
constexpr auto data (Container& container) noexcept -> decltype(
  container.data()
) { return container.data(); }

template <class T, ::std::size_t N>
constexpr T const* data (T const (&array)[N]) noexcept { return array; }

template <class T, ::std::size_t N>
constexpr T* data (T (&array)[N]) noexcept { return array; }

/* iteration */
template <class Container>
auto cbegin (Container const& container) -> decltype(::std::begin(container)) {
  return ::std::begin(container);
}

template <class Container>
auto cend (Container const& container) -> decltype(::std::end(container)) {
  return ::std::end(container);
}

template <class Container>
auto rbegin (Container const& container) -> decltype(container.rbegin()) {
  return container.rbegin();
}

template <class Container>
auto rbegin (Container& container) -> decltype(container.rbegin()) {
  return container.rbegin();
}

template <class Container>
auto crbegin (Container const& container) -> decltype(rbegin(container)) {
  return rbegin(container);
}

template <class Container>
auto rend (Container const& container) -> decltype(container.rend()) {
  return container.rend();
}

template <class Container>
auto rend (Container& container) -> decltype(container.rend()) {
  return container.rend();
}

template <class Container>
auto crend (Container const& container) -> decltype(rend(container)) {
  return rend(container);
}

template <class Iterator>
::std::reverse_iterator<Iterator> make_reverse_iterator (Iterator iter) {
  return ::std::reverse_iterator<Iterator>(iter);
}

template <
  class DelimT,
  class CharT=char,
  class Traits=::std::char_traits<CharT>
> struct ostream_joiner final : ::std::iterator<
  ::std::output_iterator_tag,
  void,
  void,
  void,
  void
> {
  using delimiter_type = DelimT;
  using ostream_type = ::std::basic_ostream<CharT, Traits>;
  using traits_type = Traits;
  using char_type = CharT;

  ostream_joiner (ostream_type& stream, delimiter_type const& delimiter) :
    stream(stream),
    delimiter { delimiter }
  { }

  ostream_joiner (ostream_type& stream, delimiter_type&& delimiter) :
    stream(stream),
    delimiter { ::core::move(delimiter) },
    first { true }
  { }

  template <class T>
  ostream_joiner& operator = (T const& item) {
    if (not first and delimiter) { this->stream << delimiter; }
    this->stream << item;
    this->first = false;
    return *this;
  }

  ostream_joiner& operator ++ (int) noexcept { return *this; }
  ostream_joiner& operator ++ () noexcept { return *this; }
  ostream_joiner& operator * () noexcept { return *this; }

private:
  ostream_type& stream;
  delimiter_type delimiter;
  bool first;
};

template <class T>
struct number_iterator {
  using iterator_category = ::std::bidirectional_iterator_tag;
  using difference_type = T;
  using value_type = T;
  using reference = add_lvalue_reference_t<T>;
  using pointer = add_pointer_t<T>;

  static_assert(::std::is_integral<value_type>::value, "");

  explicit number_iterator (value_type value, value_type step=1) noexcept :
    value { value },
    step { step }
  { }

  number_iterator (number_iterator const&) noexcept = default;
  number_iterator () noexcept = default;
  ~number_iterator () noexcept = default;

  number_iterator& operator = (number_iterator const&) noexcept = default;

  void swap (number_iterator& that) noexcept {
    ::std::swap(this->value, that.value);
    ::std::swap(this->step, that.step);
  }

  reference operator * () noexcept { return this->value; }

  number_iterator& operator ++ () noexcept {
    this->value += this->step;
    return *this;
  }

  number_iterator& operator -- () noexcept {
    this->value -= this->step;
    return *this;
  }

  number_iterator operator ++ (int) const noexcept {
    return number_iterator { this->value + this->step };
  }

  number_iterator operator -- (int) const noexcept {
    return number_iterator { this->value - this->step };
  }

  bool operator == (number_iterator const& that) const noexcept {
    return this->value == that.value and this->step == that.step;
  }

  bool operator != (number_iterator const& that) const noexcept {
    return this->value != that.value and this->step == that.step;
  }

private:
  value_type value { };
  value_type step { static_cast<value_type>(1) };
};

template <class T>
void swap (number_iterator<T>& lhs, number_iterator<T>& rhs) noexcept {
  lhs.swap(rhs);
}

template <class CharT, class Traits, class DelimT>
ostream_joiner<decay_t<DelimT>, CharT, Traits> make_ostream_joiner (
  ::std::basic_ostream<CharT, Traits>& stream,
  DelimT&& delimiter
) {
  return ostream_joiner<decay_t<DelimT>, CharT, Traits> {
    stream,
    ::core::forward<DelimT>(delimiter)
  };
}

template <class T>
number_iterator<T> make_number_iterator (T value, T step) noexcept {
  return number_iterator<T> { value, step };
}

template <class T>
number_iterator<T> make_number_iterator (T value) noexcept {
  return number_iterator<T> { value };
}

}} /* namespace core::v2 */

#endif /* CORE_ITERATOR_HPP */
