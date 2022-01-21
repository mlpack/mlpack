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
#ifndef CORE_RANGE_HPP
#define CORE_RANGE_HPP

#include <istream>
#include <utility>
#include <memory>

#include <cstdlib>

#include "type_traits.hpp"
#include "iterator.hpp"

namespace core {
inline namespace v2 {
namespace impl {

using ::std::begin;
using ::std::end;

template <class T> using adl_begin_t = decltype(begin(::std::declval<T>()));
template <class T> using adl_end_t = decltype(end(::std::declval<T>()));

template <class T>
adl_begin_t<T> adl_begin (T&& t) {
  using ::std::begin;
  return begin(::core::forward<T>(t));
}

template <class T>
adl_end_t<T> adl_end (T&& t) {
  using ::std::end;
  return end(::core::forward<T>(t));
}

} /* namespace impl */

template <class R>
struct is_range : meta::all_t<
  is_detected<impl::adl_begin_t, R>::value,
  is_detected<impl::adl_end_t, R>::value
> { };

template <class Iterator>
struct range {
  using traits = ::std::iterator_traits<Iterator>;

  using iterator_category = typename traits::iterator_category;

  using difference_type = typename traits::difference_type;
  using value_type = typename traits::value_type;

  using reference = typename traits::reference;
  using pointer = typename traits::pointer;

  using iterator = Iterator;

  static constexpr bool is_input = ::std::is_convertible<
    iterator_category,
    ::std::input_iterator_tag
  >::value;

  static constexpr bool is_output = ::std::is_convertible<
    iterator_category,
    ::std::output_iterator_tag
  >::value;

  static constexpr bool is_forward = ::std::is_convertible<
    iterator_category,
    ::std::forward_iterator_tag
  >::value;

  static constexpr bool is_bidirectional = ::std::is_convertible<
    iterator_category,
    ::std::bidirectional_iterator_tag
  >::value;

  static constexpr bool is_random_access = ::std::is_convertible<
    iterator_category,
    ::std::random_access_iterator_tag
  >::value;

  template <
    class Range,
    class=meta::when<
      meta::all<
        meta::none<
          ::std::is_pointer<iterator>::value,
          ::std::is_same<decay_t<Range>, range>::value
        >(),
        is_range<Range>::value,
        is_detected_convertible<iterator, impl::adl_begin_t, Range>::value
      >()
    >
  > explicit range (Range&& r) noexcept :
    begin_ { impl::adl_begin(::core::forward<Range>(r)) },
    end_ { impl::adl_end(::core::forward<Range>(r)) }
  { }

  range (::std::pair<iterator, iterator> pair) noexcept :
    range { ::std::get<0>(pair), ::std::get<1>(pair) }
  { }

  range (iterator begin_, iterator end_) noexcept :
    begin_ { begin_ },
    end_ { end_ }
  { }

  range (range const& that) :
    range { that.begin_, that.end_ }
  { }

  range (range&& that) noexcept :
    range { ::core::move(that.begin_), ::core::move(that.end_) }
  { that.begin_ = that.end_; }

  range () = default;
  ~range () = default;

  range& operator = (range const& that) {
    return *this = range { that };
  }

  range& operator = (range&& that) {
    range { ::std::move(that) }.swap(*this);
    return *this;
  }

  reference operator [](difference_type idx) const {
    static_assert(is_random_access, "can only subscript into random-access");
    return idx < 0 ? this->end()[idx] : this->begin()[idx];
  }

  iterator begin () const { return this->begin_; }
  iterator end () const { return this->end_; }

  reference front () const { return *this->begin(); }
  reference back () const {
    static_assert(is_bidirectional, "can only get back of bidirectional");
    return *::std::prev(this->end());
  }

  bool empty () const { return this->begin() == this->end(); }

  difference_type size () const {
    static_assert(is_forward, "can only get size of forward-range");
    return ::std::distance(this->begin(), this->end());
  }

  /* Creates an open-ended range of [start, stop) */
  range slice (difference_type start, difference_type stop) const {
    static_assert(is_forward, "can only slice forward-range");
    /* Behavior is:
     * if start is negative, the begin marker is this->end() - start
     * if stop is negative, the end marker is this->end() - stop
     * if start is positive, the begin marker is this->begin() + start
     * if stop is positive, the end marker is this->begin() + stop
     *
     * if start and stop are positive, and stop is less than or equal to start,
     * an empty range is returned.
     *
     * if start and stop are negative and stop is less than or equal to start,
     * an empty range is returned.
     *
     * if start is positive and stop is negative and abs(stop) + start is
     * greater than or equal to this->size(), an empty range is returned.
     *
     * if start is negative and stop is positive and this->size() + start is
     * greater or equal to stop, an empty range is returned.
     *
     * The first two conditions can be computed cheaply, while the third and
     * fourth are a bit more expensive, but WILL be required no matter what
     * iterator type we are. However we don't compute the size until after
     * we've checked the first two conditions
     *
     * An example with python style slicing for each would be:
     * [4:3] -> empty range
     * [-4:-4] -> empty range
     * [7:-4] -> empty range for string of size 11 or more
     * [-4:15] -> empty range for a string of size 19 or less.
     */
    bool const start_positive = start > 0;
    bool const stop_positive = stop > 0;
    bool const stop_less = stop < start;
    bool const first_return_empty = start_positive == stop_positive and stop_less;
    if (first_return_empty) { return range { }; }

    /* now safe to compute size */
    auto const size = this->size();
    auto const third_empty = ::std::abs(stop) + start;

    bool const second_return_empty =
      (start_positive and not stop_positive and third_empty >= size) or
      (not start_positive and stop_positive and size + start >= stop);
    if (second_return_empty) { return range { }; }

    /* While the code below technically works for all iterators it is
     * ineffecient in some cases for bidirectional ranges, where either of
     * start or stop are negative.
     * TODO: Specialize for bidirectional operators
     */
    if (not start_positive) { start += size; }
    if (not stop_positive) { stop += size; }

    auto begin = this->begin();
    ::std::advance(begin, start);

    auto end = begin;
    ::std::advance(end, stop - start);

    return range { begin, end };
  }

  /* Creates an open-ended range of [start, end()) */
  range slice (difference_type start) const {
    static_assert(is_forward, "can only slice forward-range");
    return range { split(start).second };
  }

  ::std::pair<range, range> split (difference_type idx) const {
    static_assert(is_forward,"can only split a forward-range");
    if (idx >= 0) {
      range second { *this };
      second.pop_front_upto(idx);
      return ::std::make_pair(range { this->begin(), second.begin() }, second);
    }

    range first { *this };
    first.pop_back_upto(-idx);
    return ::std::make_pair(first, range { first.end(), this->end() });
  }

  /* mutates range */
  void pop_front (difference_type n) { ::std::advance(this->begin_, n); }
  void pop_front () { ++this->begin_; }

  void pop_back (difference_type n) {
    static_assert(is_bidirectional, "can only pop-back bidirectional-range");
    ::std::advance(this->end_, -n);
  }

  void pop_back () {
    static_assert(is_bidirectional, "can only pop-back bidirectional-range");
    --this->end_;
  }

  /* Negative argument causes no change */
  void pop_front_upto (difference_type n) {
    ::std::advance(
      this->begin_,
      ::std::min(::std::max<difference_type>(0, n), this->size())
    );
  }

  /* Negative argument causes no change */
  void pop_back_upto (difference_type n) {
    static_assert(is_bidirectional, "can only pop-back-upto bidirectional");
    ::std::advance(
      this->end_,
      -::std::min(::std::max<difference_type>(0, n), this->size())
    );
  }

  void swap (range& that) noexcept(is_nothrow_swappable<iterator>::value) {
    using ::std::swap;
    swap(this->begin_, that.begin_);
    swap(this->end_, that.end_);
  }

private:
  iterator begin_;
  iterator end_;
};

template <class T>
auto make_range (T* ptr, ::std::size_t n) -> range<T*> {
  return range<T*> { ptr, ptr + n };
}

template <class Iterator>
auto make_range (Iterator begin, Iterator end) -> range<Iterator> {
  return range<Iterator> { begin, end };
}

template <class Range>
auto make_range (Range&& value) -> range<decltype(begin(value))> {
  using ::std::begin;
  using ::std::end;
  return make_range(begin(value), end(value));
}

/* Used like: core::make_range<char>(::std::cin) */
template <
  class T,
  class CharT,
  class Traits=::std::char_traits<CharT>
> auto make_range (::std::basic_istream<CharT, Traits>& stream) -> range<
  ::std::istream_iterator<T, CharT, Traits>
> {
  using iterator = ::std::istream_iterator<T, CharT, Traits>;
  return make_range(iterator { stream }, iterator { });
}

template <class CharT, class Traits=::std::char_traits<CharT>>
auto make_range (::std::basic_streambuf<CharT, Traits>* buffer) -> range<
  ::std::istreambuf_iterator<CharT, Traits>
> {
  using iterator = ::std::istreambuf_iterator<CharT, Traits>;
  return make_range(iterator { buffer }, iterator { });
}

template <class Iter>
range<::std::move_iterator<Iter>> make_move_range (Iter start, Iter stop) {
  return make_range(
    ::std::make_move_iterator(start),
    ::std::make_move_iterator(stop));
}

template <class T>
range<::std::move_iterator<T*>> make_move_range (T* ptr, ::std::size_t n) {
  return make_move_range(ptr, ptr + n);
}

template <class T>
range<number_iterator<T>> make_number_range(T start, T stop, T step) noexcept {
  auto begin = make_number_iterator(start, step);
  auto end = make_number_iterator(stop, step);
  return make_range(begin, end);
}

template <class T>
range<number_iterator<T>> make_number_range (T start, T stop) noexcept {
  return make_range(make_number_iterator(start), make_number_iterator(stop));
}

template <class Iterator>
void swap (range<Iterator>& lhs, range<Iterator>& rhs) noexcept(
  noexcept(lhs.swap(rhs))
) { lhs.swap(rhs); }

}} /* namespace core::v2 */

#endif /* CORE_RANGE_HPP */
