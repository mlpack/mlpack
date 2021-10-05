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
#ifndef CORE_ALGORITHM_HPP
#define CORE_ALGORITHM_HPP

#include <algorithm>

#include "functional.hpp"
#include "utility.hpp"
#include "range.hpp"

namespace core {
inline namespace v2 {
namespace impl {

template <class InputIt1, class InputIt2, class Predicate>
bool equal (
  range<InputIt1> r1,
  range<InputIt2> r2,
  Predicate&& p,
  ::std::random_access_iterator_tag,
  ::std::random_access_iterator_tag
) {
  if (r1.size() != r2.size()) { return false; }
  return ::std::equal(
    begin(r1),
    end(r1),
    begin(r2),
    ::core::forward<Predicate>(p)
  );
}

template <class InputIt1, class InputIt2, class Predicate>
bool equal (
  range<InputIt1> r1,
  range<InputIt2> r2,
  Predicate&& p,
  ::std::input_iterator_tag,
  ::std::input_iterator_tag
) {
  while (not r1.empty() and not r2.empty()) {
    if (
      not ::core::invoke(
        ::core::forward<Predicate>(p),
        r1.front(),
        r2.front())
    ) { return false; }
    r1.pop_front();
    r2.pop_front();
  }
  return r1.empty() and r2.empty();
}

} /* namespace impl */

/* non-range based algorithms */
template <class T>
constexpr T const& min (T const& lhs, T const& rhs) {
  return (rhs < lhs) ? rhs : lhs;
}

template <class T, class Compare>
constexpr T const& min (T const& lhs, T const& rhs, Compare compare) {
  return compare(rhs, lhs) ? rhs : lhs;
}

template <class T>
constexpr T const& max (T const& lhs, T const& rhs) {
  return (lhs < rhs) ? rhs : lhs;
}

template <class T, class Compare>
constexpr T const& max (T const& lhs, T const& rhs, Compare compare) {
  return compare(lhs, rhs) ? rhs : lhs;
}

/* extensions */
template <class T, class Compare=less<>>
constexpr T const& clamp (
  T const& value,
  T const& low,
  T const& high,
  Compare compare = Compare { }
) {
  return compare(value, low)
    ? low
    : compare(high, value)
      ? high
      : value;
}

/* N4318 (modified) */
template <
  class T,
  class Compare = ::core::less<>,
  class Difference = ::core::minus<>
> constexpr auto abs_diff (
  T const& a,
  T const& b,
  Compare compare = Compare { },
  Difference diff = Difference { }
) -> decltype(compare(a, b) ? diff(b, a) : diff(a, b)) {
  return compare(a, b) ? diff(b, a) : diff(a, b);
}

/* non-modifying sequence algorithms */
template <class Range, class UnaryPredicate>
auto all_of (Range&& rng, UnaryPredicate&& p) -> meta::when<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "all_of requires InputIterators");
  return ::std::all_of(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(p)
  );
}

template <class Range, class UnaryPredicate>
auto any_of (Range&& rng, UnaryPredicate&& p) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "any_of requires InputIterators");
  return ::std::any_of(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(p)
  );
}

template <class Range, class UnaryPredicate>
auto none_of (Range&& rng, UnaryPredicate&& p) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "none_of requires InputIterators");
  return ::std::none_of(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(p)
  );
}

template <class Range, class UnaryFunction>
auto for_each (Range&& rng, UnaryFunction&& f) -> enable_if_t<
  is_range<Range>::value,
  decay_t<UnaryFunction>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "for_each requires InputIterators");
  return ::std::for_each(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryFunction>(f)
  );
}

template <class Range, class UnaryFunction, class UnaryPredicate>
UnaryFunction for_each_if (Range&& r, UnaryFunction uf, UnaryPredicate up) {
  auto range = make_range(::core::forward<Range>(r));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "for_each_if requires InputIterators");
  while (not range.empty()) {
    if (up(range.front())) { uf(range.front()); }
    range.pop_front();
  }
  return uf;
}

template <class Range, class UnaryFunction, class UnaryPredicate>
auto for_each_while (
  Range&& r,
  UnaryFunction f,
  UnaryPredicate p
) -> decltype(begin(make_range(::core::forward<Range>(r)))) {
  auto range = make_range(::core::forward<Range>(r));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "for_each_while requires InputIterators");
  while (not range.empty()) {
    if (not p(range.front())) { break; }
    f(range.front());
    range.pop_front();
  }
  return range.begin();
}

template <class Range, class UnaryFunction, class T>
auto for_each_until (
  Range&& r,
  UnaryFunction f,
  T const& value
) -> decltype(begin(make_range(::core::forward<Range>(r)))) {
  auto range = make_range(::core::forward<Range>(r));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "for_each_until requires InputIterators");
  while (not range.empty()) {
    if (range.front() == value) { break; }
    f(range.front());
    range.pop_front();
  }
  return range.begin();
}

template <class Range, class T>
auto count (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decltype(
    ::std::count(
      ::std::begin(::core::forward<Range>(rng)),
      ::std::end(::core::forward<Range>(rng)),
      value
    )
  )
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "count requires InputIterators");
  return ::std::count(::std::begin(range), ::std::end(range), value);
}

template <class Range, class UnaryPredicate>
auto count_if (Range&& rng, UnaryPredicate&& p) -> enable_if_t<
  is_range<Range>::value,
  decltype(
    ::std::count_if(
      ::std::begin(::core::forward<Range>(rng)),
      ::std::end(::core::forward<Range>(rng)),
      ::core::forward<UnaryPredicate>(p)
    )
  )
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "count_if requires InputIterators");
  return ::std::count_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(p)
  );
}

template <class InputIt1, class InputIt2, class BinaryPredicate>
::std::pair<InputIt1, InputIt2> mismatch (
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  InputIt2 last2,
  BinaryPredicate predicate
) {
  auto r1 = make_range(first1, last1);
  auto r2 = make_range(first2, last2);
  while (not r1.empty() and not r2.empty()) {
    if (not predicate(r1.front(), r2.front())) { break; }
    r1.pop_front();
    r2.pop_front();
  }
  return ::std::make_pair(r1.begin(), r2.begin());
}

template <class InputIt1, class InputIt2>
::std::pair<InputIt1, InputIt2> mismatch (
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  InputIt2 last2
) { return (mismatch)(first1, last1, first2, last2, equal_to<> { }); }

template <
  class Range1,
  class Range2,
  class BinaryPred,
  meta::require<
    meta::all_of<meta::list<Range1, Range2>, is_range>()
  > = __LINE__
> auto mismatch (Range1&& r1, Range2&& r2, BinaryPred&& bp) -> ::std::pair<
  decltype(::std::begin(::core::forward<Range1>(r1))),
  decltype(::std::begin(::core::forward<Range2>(r2)))
> {
  auto range1 = make_range(::core::forward<Range1>(r1));
  auto range2 = make_range(::core::forward<Range2>(r2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1 and is_input2, "mismatch requires InputIterators");
  return (mismatch)(
    range1.begin(),
    range1.end(),
    range2.begin(),
    range2.end(),
    ::core::forward<BinaryPred>(bp));
}

template <
  class Range,
  class InputIt,
  meta::require<
    meta::all<is_range<Range>::value, meta::none<is_range<InputIt>::value>()>()
  > = __LINE__
> auto mismatch(Range&& rng, InputIt&& it) -> ::std::pair<
  decltype(make_range(::core::forward<Range>(rng)).begin()),
  decay_t<InputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "mismatch requires InputIterators");
  return ::std::mismatch(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<InputIt>(it)
  );
}

template <
  class Range,
  class InputIt,
  class BinaryPredicate,
  meta::require<
    meta::all<is_range<Range>::value, meta::none<is_range<InputIt>>()>()
  > = __LINE__
> auto mismatch(Range&& r, InputIt&& it, BinaryPredicate&& bp) -> ::std::pair<
  decltype(core::make_range(::core::forward<Range>(r).begin())),
  decay_t<InputIt>
> {
  auto range = make_range(::core::forward<Range>(r));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "mismatch requires InputIterators");
  return ::std::mismatch(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<InputIt>(it),
    ::core::forward<BinaryPredicate>(bp)
  );
}

template <class InputIt1, class InputIt2, class BinaryPredicate>
bool equal (
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  InputIt2 last2,
  BinaryPredicate bp
) {
  auto r1 = make_range(first1, last1);
  auto r2 = make_range(first2, last2);
  using tag1 = typename decltype(r1)::iterator_category;
  using tag2 = typename decltype(r2)::iterator_category;
  return impl::equal(r1, r2, bp, tag1 { }, tag2 { });
}

template <class InputIt1, class InputIt2>
bool equal (InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2) {
  return equal(first1, last1, first2, last2, equal_to<> { });
}

template <
  class Range1,
  class Range2,
  meta::require<
    meta::all_of<meta::list<Range1, Range2>, is_range>()
  > = __LINE__
> bool equal (Range1&& range1, Range2&& range2) {
  auto r1 = make_range(::core::forward<Range1>(range1));
  auto r2 = make_range(::core::forward<Range2>(range2));
  static constexpr auto is_input1 = decltype(r1)::is_input;
  static constexpr auto is_input2 = decltype(r2)::is_input;
  static_assert(is_input1, "equal requires InputIterators");
  static_assert(is_input2, "equal requires InputIterators");
  return (equal)(r1.begin(), r1.end(), r2.begin(), r2.end());
}

template <
  class Range1,
  class Range2,
  class BinaryPredicate,
  meta::require<
    meta::all_of<meta::list<Range1, Range2>, is_range>()
  > = __LINE__
> bool equal (Range1&& range1, Range2&& range2, BinaryPredicate&& bp) {
  auto r1 = make_range(::core::forward<Range1>(range1));
  auto r2 = make_range(::core::forward<Range2>(range2));
  static constexpr auto is_input1 = decltype(r1)::is_input;
  static constexpr auto is_input2 = decltype(r2)::is_input;
  static_assert(is_input1, "equal requires InputIterators");
  static_assert(is_input2, "equal requires InputIterators");
  return equal(
    r1.begin(),
    r1.end(),
    r2.begin(),
    r2.end(),
    ::core::forward<BinaryPredicate>(bp)
  );
}

template <
  class Range,
  class InputIt,
  meta::require<
    meta::all<is_range<Range>::value, meta::none<is_range<InputIt>::value>()>()
  > = __LINE__
> bool equal (Range&& rng, InputIt&& it) {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "equal requires InputIterators");
  return ::std::equal(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<InputIt>(it)
  );
}

template <
  class Range,
  class InputIt,
  class BinaryPredicate,
  meta::require<
    meta::all<is_range<Range>::value, meta::none<is_range<InputIt>>()>()
  > = __LINE__
> bool equal (Range&& rng, InputIt&& it, BinaryPredicate&& bp) {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "equal requires InputIterators");
  return ::std::equal(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<InputIt>(it),
    ::core::forward<BinaryPredicate>(bp)
  );
}

template <class Range, class T>
auto find (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "find requires InputIterators");
  return ::std::find(::std::begin(range), ::std::end(range), value);
}

template <class Range, class UnaryPredicate>
auto find_if (Range&& rng, UnaryPredicate&& p) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "find_if requires InputIterators");
  return ::std::find_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(p)
  );
}

template <class Range, class UnaryPredicate>
auto find_if_not (Range&& rng, UnaryPredicate&& p) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "find_if_not requires InputIterators");
  return ::std::find_if_not(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(p)
  );
}

template <class Range1, class Range2>
auto find_end (Range1&& rng1, Range2&& rng2) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decltype(::std::begin(::core::forward<Range1>(rng1)))
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "find_end requires ForwardIterators");
  static_assert(is_forward2, "find_end requires ForwardIterators");
  return ::std::find_end(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2)
  );
}

template <class Range1, class Range2, class BinaryPred>
auto find_end (Range1&& rng1, Range2&& rng2, BinaryPred& bp) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decltype(::std::begin(::core::forward<Range1>(rng1)))
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "find_end requires ForwardIterators");
  static_assert(is_forward2, "find_end requires ForwardIterators");
  return ::std::find_end(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<BinaryPred>(bp)
  );
}

template <class IRange, class FRange>
auto find_first_of (IRange&& irng, FRange&& frng) -> meta::when<
  meta::all_of<meta::list<IRange, FRange>, is_range>(),
  decltype(::std::begin(::core::forward<IRange>(irng)))
> {
  auto irange = make_range(::core::forward<IRange>(irng));
  auto frange = make_range(::core::forward<FRange>(frng));
  static constexpr auto is_input = decltype(irange)::is_input;
  static constexpr auto is_forward = decltype(frange)::is_forward;
  static_assert(is_input, "find_first_of requires InputIterators");
  static_assert(is_forward, "find_first_of requires ForwardIterators");
  return ::std::find_first_of(
    ::std::begin(irange),
    ::std::end(irange),
    ::std::begin(frange),
    ::std::end(frange)
  );
}

template <class IRange, class FRange, class BinaryPred>
auto find_first_of (
  IRange&& irng,
  FRange&& frng,
  BinaryPred&& bp
) -> meta::when<
  meta::all_of<meta::list<IRange, FRange>, is_range>(),
  decltype(::std::begin(::core::forward<IRange>(irng)))
> {
  auto irange = make_range(::core::forward<IRange>(irng));
  auto frange = make_range(::core::forward<FRange>(frng));
  static constexpr auto is_input = decltype(irange)::is_input;
  static constexpr auto is_forward = decltype(frange)::is_forward;
  static_assert(is_input, "find_first_of requires InputIterators");
  static_assert(is_forward, "find_first_of requires ForwardIterators");
  return ::std::find_first_of(
    ::std::begin(irange),
    ::std::end(irange),
    ::std::begin(frange),
    ::std::end(frange),
    ::core::forward<BinaryPred>(bp)
  );
}

template <class Range>
auto adjacent_find (Range&& rng) -> meta::when<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "adjacent_find requires ForwardIterators");
  return ::std::adjacent_find(::std::begin(range), ::std::end(range));
}

template <class Range, class BinaryPredicate>
auto adjacent_find (Range&& rng, BinaryPredicate&& bp) -> meta::when<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "adjacent_find requires ForwardIterators");
  return ::std::adjacent_find(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<BinaryPredicate>(bp)
  );
}

template <class Range1, class Range2>
auto search (Range1&& rng1, Range2&& rng2) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decltype(::std::begin(::core::forward<Range1>(rng1)))
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "search requires ForwardIterators");
  static_assert(is_forward2, "search requires ForwardIterators");
  return ::std::search(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2)
  );
}

template <class Range1, class Range2, class BinaryPred>
auto search (Range1&& rng1, Range2&& rng2, BinaryPred&& bp) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decltype(::std::begin(::core::forward<Range1>(rng1)))
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "search requires ForwardIterators");
  static_assert(is_forward2, "search requires ForwardIterators");
  return ::std::search(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<BinaryPred>(bp)
  );
}

template <class Range, class Size, class T>
auto search_n (Range&& rng, Size&& count, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "search_n requires ForwardIterators");
  return ::std::search_n(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Size>(count),
    value
  );
}

template <class Range, class Size, class T, class BinaryPred>
auto search_n (
  Range&& rng,
  Size&& count,
  T const& value,
  BinaryPred&& bp
) -> meta::when<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "search_n requires ForwardIterators");
  return ::std::search_n(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Size>(count),
    value,
    ::core::forward<BinaryPred>(bp)
  );
}

/* modifying sequence algorithms */
template <class Range, class OutputIt>
auto copy (Range&& rng, OutputIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "copy requires InputIterators");
  return ::std::copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range, class OutputIt, class UnaryPredicate>
auto copy_if (Range&& rng, OutputIt&& it, UnaryPredicate&& up) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "copy_if requires InputIterators");
  return ::std::copy_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    ::core::forward<UnaryPredicate>(up)
  );
}

template <class Range, class OutputIt, class T>
OutputIt copy_until (Range&& r, OutputIt it, T const& value) {
  auto range = make_range(::core::forward<Range>(r));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "copy_until requires InputIterators");
  while (not range.empty()) {
    if (range.front() == value) { break; }
    *it++ = range.front();
    range.pop_front();
  }
  return it;
}

template <class Range, class BidirIt>
auto copy_backward (Range&& rng, BidirIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<BidirIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "copy_backward requires BidirectionalIterators");
  return ::std::copy_backward(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<BidirIt>(it)
  );
}

template <class Range, class OutputIt>
auto move (Range&& rng, OutputIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "move requires InputIterators");
  return ::std::move(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range, class BidirIt>
auto move_backward (Range&& rng, BidirIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<BidirIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "move_backward requires BidirectionalIterators");
  return ::std::move_backward(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<BidirIt>(it)
  );
}

template <class Range, class T>
auto fill (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "fill requires ForwardIterators");
  return ::std::fill(::std::begin(range), ::std::end(range), value);
}

template <class Range, class OutputIt, class UnaryOperation>
auto transform (
  Range&& rng,
  OutputIt&& it,
  UnaryOperation&& op
) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "transform requires InputIterators");
  return ::std::transform(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    ::core::forward<UnaryOperation>(op)
  );
}

template <class Range, class OutputIt, class UnaryOperation, class UnaryPred>
auto transform_if (
  Range&& rng,
  OutputIt it,
  UnaryOperation op,
  UnaryPred up
) -> enable_if_t<
  is_range<Range>::value,
  OutputIt
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "transform_if requires ForwardIterators");
  while (not range.empty()) {
    if (invoke(up, range.front())) {
      *it = invoke(op, range.front());
      ++it;
    }
    range.pop_front();
  }
  return it;
}

template <class InputIt, class Size, class OutputIt, class UnaryOp>
OutputIt transform_n (InputIt in, Size count, OutputIt out, UnaryOp op) {
  while (count > 0) {
    *out = invoke(op, *in);
    ++out;
    ++in;
    --count;
  }
  return out;
}

template <
  class InputIt1,
  class InputIt2,
  class Size,
  class OutputIt,
  class UnaryOp
> OutputIt transform_n (
  InputIt1 in1,
  InputIt2 in2,
  Size count,
  OutputIt out,
  UnaryOp op
) {
  while (count > 0) {
    *out = invoke(op, *in1, *in2);
    ++out;
    ++in1;
    ++in2;
    --count;
  }
  return out;
}


template <
  class Range,
  class InputIt,
  class OutputIt,
  class BinaryOperation,
  meta::require<
    meta::all<is_range<Range>::value, meta::none<is_range<InputIt>::value>()>()
  > = __LINE__
> decay_t<OutputIt> transform (
  Range&& r,
  InputIt&& in,
  OutputIt&& out,
  BinaryOperation&& op
) {
  auto range = make_range(::core::forward<Range>(r));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "transform requires InputIterators");
  return ::std::transform(
    range.begin(),
    range.end(),
    ::core::forward<InputIt>(in),
    ::core::forward<OutputIt>(out),
    ::core::forward<BinaryOperation>(op));
}

template <
  class Range1,
  class Range2,
  class OutputIt,
  class BinaryOperation,
  meta::require<
    meta::all_of<meta::list<Range1, Range2>, is_range>()
  > = __LINE__
> decay_t<OutputIt> transform (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt&& it,
  BinaryOperation&& op
) {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "transform requires InputIterators");
  static_assert(is_input2, "transform requires InputIterators");
  return ::std::transform(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::core::forward<OutputIt>(it),
    ::core::forward<BinaryOperation>(op)
  );
}

template <
  class Range1,
  class Range2,
  class OutputIt,
  class BinaryOperation,
  class BinaryPredicate
> auto transform_if (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt it,
  BinaryOperation op,
  BinaryPredicate bp
) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  OutputIt
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "transform_if requires ForwardIterators");
  static_assert(is_forward2, "transform_if requires ForwardIterators");
  while (not range1.empty()) {
    if (bp(range1.front(), range2.front())) {
      *it = op(range1.front(), range2.front());
      ++it;
    }
    range1.pop_front();
    range2.pop_front();
  }
  return it;
}

template <class Range, class T>
auto remove (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "remove requires ForwardIterators");
  return ::std::remove(::std::begin(range), ::std::end(range), value);
}

template <class Range, class UnaryPredicate>
auto remove_if (Range&& rng, UnaryPredicate&& up) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "remove_if requires ForwardIterators");
  return ::std::remove_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(up)
  );
}

template <class Range, class OutputIt, class T>
auto remove_copy (Range&& rng, OutputIt&& it, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "remove_copy requires InputIterators");
  return ::std::remove_copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    value
  );
}

template <class Range, class OutputIt, class UnaryPred>
auto remove_copy_if (Range&& rng, OutputIt&& it, UnaryPred&& up) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "remove_copy_if requires InputIterators");
  return ::std::remove_copy_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    ::core::forward<UnaryPred>(up)
  );
}

template <class Range, class T>
auto remove_erase (Range&& rng, T const& val) -> enable_if_t<
  is_range<Range>::value
> {
  ::core::forward<Range>(rng).erase(
    remove(::core::forward<Range>(rng), val),
    ::std::end(::core::forward<Range>(rng))
  );
}

template <class Range, class UnaryPred>
auto remove_erase_if (Range&& rng, UnaryPred&& up) -> enable_if_t<
  is_range<Range>::value
> {
  ::core::forward<Range>(rng).erase(
    remove_if(
      ::core::forward<Range>(rng),
      ::core::forward<UnaryPred>(up)
    ),
    ::std::end(::core::forward<Range>(rng))
  );
}

template <class Range, class T>
auto replace (Range&& rng, T const& old, T const& value) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_input;
  static_assert(is_forward, "replace requires ForwardIterators");
  return ::std::replace(
    ::std::begin(range),
    ::std::end(range),
    old,
    value
  );
}

template <class Range, class UnaryPred, class T>
auto replace_if (Range&& rng, UnaryPred&& up, T const& value) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "replace_if requires ForwardIterators");
  return ::std::replace_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPred>(up),
    value
  );
}

template <class Range, class OutputIt, class T>
auto replace_copy (
  Range&& rng,
  OutputIt&& it,
  T const& old,
  T const& value
) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "replace_copy requires InputIterators");
  return ::std::replace_copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    old,
    value
  );
}

template <class Range, class OutputIt, class UnaryPred, class T>
auto replace_copy_if (
  Range&& rng,
  OutputIt&& it,
  UnaryPred&& up,
  T const& value
) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "replace_copy_if requires InputIterators");
  return ::std::replace_copy_if(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    ::core::forward<UnaryPred>(up),
    value
  );
}

template <class Range, class ForwardIt>
auto swap_ranges (Range&& rng, ForwardIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<ForwardIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "swap_ranges requires ForwardIterators");
  return ::std::swap_ranges(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<ForwardIt>(it)
  );
}

template <class Range>
auto reverse (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "reverse requires BidirectionalIterators");
  return ::std::reverse(::std::begin(range), ::std::end(range));
}

template <class Range, class OutputIt>
auto reverse_copy (Range&& rng, OutputIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "reverse_copy requires BidirectionalIterators");
  return ::std::reverse_copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range, class ForwardIt>
auto rotate (Range&& rng, ForwardIt&& it) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "rotate requires ForwardIterators");
  ::std::rotate(
    ::std::begin(range),
    ::core::forward<ForwardIt>(it),
    ::std::end(range)
  );
}

template <class Range, class ForwardIt, class OutputIt>
auto rotate_copy (Range&& rng, ForwardIt&& it, OutputIt&& ot) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "rotate_copy requires ForwardIterators");
  return ::std::rotate_copy(
    ::std::begin(range),
    ::core::forward<ForwardIt>(it),
    ::std::end(range),
    ::core::forward<OutputIt>(ot)
  );
}

template <class Range, class URNG>
auto shuffle (Range&& rng, URNG&& g) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "shuffle requires RandomAccessIterators");
  return ::std::shuffle(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<URNG>(g)
  );
}

template <class Range>
auto unique (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "unique requires ForwardIterators");
  return ::std::unique(::std::begin(range), ::std::end(range));
}

template <class Range, class BinaryPredicate>
auto unique (Range&& rng, BinaryPredicate&& bp) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "unique requires ForwardIterators");
  return ::std::unique(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<BinaryPredicate>(bp)
  );
}

template <class Range, class OutputIt>
auto unique_copy (Range&& rng, OutputIt&& it) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "unique_copy requires InputIterators");
  return ::std::unique_copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range, class OutputIt, class BinaryPred>
auto unique_copy (Range&& rng, OutputIt&& it, BinaryPred&& bp) -> enable_if_t<
  is_range<Range>::value,
  decay_t<OutputIt>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "unique_copy requires InputIterators");
  return ::std::unique_copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputIt>(it),
    ::core::forward<BinaryPred>(bp)
  );
}

/* partitioning operations */
template <class Range, class UnaryPredicate>
auto is_partitioned (Range&& rng, UnaryPredicate&& up) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "is_partitioned requires InputIterators");
  return ::std::is_partitioned(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(up)
  );
}

template <class Range, class UnaryPredicate>
auto partition (Range&& rng, UnaryPredicate&& up) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "partition requires ForwardIterators");
  return ::std::partition(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(up)
  );
}

template <class Range, class OutputTrue, class OutputFalse, class UnaryPred>
auto partition_copy (
  Range&& rng,
  OutputTrue&& ot,
  OutputFalse&& of,
  UnaryPred&& up
) -> enable_if_t<
  is_range<Range>::value,
  ::std::pair<decay_t<OutputTrue>, decay_t<OutputFalse>>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_input = decltype(range)::is_input;
  static_assert(is_input, "partition_copy requires InputIterators");
  return ::std::partition_copy(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<OutputTrue>(ot),
    ::core::forward<OutputFalse>(of),
    ::core::forward<UnaryPred>(up)
  );
}

template <class Range, class UnaryPredicate>
auto stable_partition (Range&& rng, UnaryPredicate&& up) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "stable_partition requires BidirectionalIterators");
  return ::std::stable_partition(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(up)
  );
}

template <class Range, class UnaryPredicate>
auto partition_point (Range&& rng, UnaryPredicate&& up) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "partition_point requires ForwardIterators");
  return ::std::partition_point(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<UnaryPredicate>(up)
  );
}

/* sorting operations */

template <class Range>
auto is_sorted (Range&& rng) -> enable_if_t<is_range<Range>::value, bool> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "is_sorted requires ForwardIterators");
  return ::std::is_sorted(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto is_sorted (Range&& rng, Compare&& compare) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "is_sorted requires ForwardIterators");
  return ::std::is_sorted(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(compare)
  );
}

template <class Range>
auto is_sorted_until (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "is_sorted_until requires ForwardIterators");
  return ::std::is_sorted_until(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto is_sorted_until (Range&& rng, Compare&& compare) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "is_sorted_until requires ForwardIterators");
  return ::std::is_sorted_until(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(compare)
  );
}

template <class Range>
auto sort (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "sort requires RandomAccessIterators");
  return ::std::sort(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto sort (Range&& rng, Compare&& cmp) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "sort requires RandomAccessIterators");
  return ::std::sort(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range, class RandomIt>
auto partial_sort (Range&& rng, RandomIt&& it) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "partial_sort requires RandomAccessIterators");
  return ::std::partial_sort(
    ::std::begin(range),
    ::core::forward<RandomIt>(it),
    ::std::end(range)
  );
}

template <class Range, class RandomIt, class Compare>
auto partial_sort (Range&& rng, RandomIt&& it, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "partial_sort requires RandomAccessIterators");
  return ::std::partial_sort(
    ::std::begin(range),
    ::core::forward<RandomIt>(it),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class IRange, class RRange>
auto partial_sort_copy (IRange&& irng, RRange&& rrng) -> meta::when<
  meta::all_of<meta::list<IRange, RRange>, is_range>(),
  decltype(::std::begin(::core::forward<RRange>(rrng)))
> {
  auto irange = make_range(::core::forward<IRange>(irng));
  auto rrange = make_range(::core::forward<RRange>(rrng));
  static constexpr auto is_input = decltype(irange)::is_input;
  static constexpr auto is_random = decltype(rrange)::is_random_access;
  static_assert(is_input, "partial_sort_copy requires InputIterators");
  static_assert(is_random, "partial_sort_copy requires RandomAccessIterators");
  return ::std::partial_sort_copy(
    ::std::begin(irange),
    ::std::end(irange),
    ::std::begin(rrange),
    ::std::end(rrange)
  );
}

template <class IRange, class RRange, class Compare>
auto partial_sort_copy (
  IRange&& irng,
  RRange&& rrng,
  Compare&& cmp
) -> meta::when<
  meta::all_of<meta::list<IRange, RRange>, is_range>(),
  decltype(::std::begin(::core::forward<RRange>(rrng)))
> {
  auto irange = make_range(::core::forward<IRange>(irng));
  auto rrange = make_range(::core::forward<RRange>(rrng));
  static constexpr auto is_input = decltype(irange)::is_input;
  static constexpr auto is_random = decltype(rrange)::is_random_access;
  static_assert(is_input, "partial_sort_copy requires InputIterators");
  static_assert(is_random, "partial_sort_copy requires RandomAccessIterators");
  return ::std::partial_sort_copy(
    ::std::begin(irange),
    ::std::end(irange),
    ::std::begin(rrange),
    ::std::end(rrange),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto stable_sort (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "stable_sort requires RandomAccessIterators");
  return ::std::stable_sort(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto stable_sort (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "stable_sort requires RandomAccessIterators");
  return ::std::stable_sort(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range, class RandomIt>
auto nth_element (Range&& rng, RandomIt&& it) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "nth_element requires RandomAccessIterators");
  return ::std::nth_element(
    ::std::begin(range),
    ::core::forward<RandomIt>(it),
    ::std::end(range)
  );
}

template <class Range, class RandomIt, class Compare>
auto nth_element (Range&& rng, RandomIt&& it, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "nth_element requires RandomAccessIterators");
  return ::std::nth_element(
    ::std::begin(range),
    ::core::forward<RandomIt>(it),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

/* binary search operations (on sorted ranges) */
template <class Range, class T>
auto lower_bound (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "lower_bound requires ForwardIterators");
  return ::std::lower_bound(::std::begin(range), ::std::end(range), value);
}

template <class Range, class T, class Compare>
auto lower_bound (Range&& rng, T const& value, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "lower_bound requires ForwardIterators");
  return ::std::lower_bound(
    ::std::begin(range),
    ::std::end(range),
    value,
    ::core::forward<Compare>(cmp)
  );
}

template <class Range, class T>
auto upper_bound (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "upper_bound requires ForwardIterators");
  return ::std::upper_bound(::std::begin(range), ::std::end(range), value);
}

template <class Range, class T, class Compare>
auto upper_bound (Range&& rng, T const& value, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "upper_bound requires ForwardIterators");
  return ::std::upper_bound(
    ::std::begin(range),
    ::std::end(range),
    value,
    ::core::forward<Compare>(cmp)
  );
}

template <class Range, class T>
auto binary_search (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "binary_search requires ForwardIterators");
  return ::std::binary_search(::std::begin(range), ::std::end(range), value);
}

template <class Range, class T, class Compare>
auto binary_search (Range&& rng, T const& value, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "binary_search requires ForwardIterators");
  return ::std::binary_search(
    ::std::begin(range),
    ::std::end(range),
    value,
    ::core::forward<Compare>(cmp)
  );
}

template <class Range, class T>
auto equal_range (Range&& rng, T const& value) -> enable_if_t<
  is_range<Range>::value,
  range<decltype(::std::begin(::core::forward<Range>(rng)))>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "equal_range requires ForwardIterators");
  return ::std::equal_range(::std::begin(range), ::std::end(range), value);
}

template <class Range, class T, class Compare>
auto equal_range (Range&& rng, T const& value, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  range<decltype(::std::begin(::core::forward<Range>(rng)))>
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "equal_range requires ForwardIterators");
  return ::std::equal_range(
    ::std::begin(range),
    ::std::end(range),
    value,
    ::core::forward<Compare>(cmp)
  );
}

/* set operations (on sorted ranges) */
template <class Range1, class Range2, class OutputIt>
auto merge (Range1&& rng1, Range2&& rng2, OutputIt&& it) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "merge requires InputIterators");
  static_assert(is_input2, "merge requires InputIterators");
  return ::std::merge(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range1, class Range2, class OutputIt, class Compare>
auto merge (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt&& it,
  Compare&& cmp
) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "merge requires InputIterators");
  static_assert(is_input2, "merge requires InputIterators");
  return ::std::merge(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range, class BidirIt>
auto inplace_merge (Range&& rng, BidirIt&& it) -> meta::when<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "inplace_merge requires BidirectionalIterators");
  return ::std::inplace_merge(
    ::std::begin(range),
    ::core::forward<BidirIt>(it),
    ::std::end(range)
  );
}

template <class Range, class BidirIt, class Compare>
auto inplace_merge (Range&& rng, BidirIt&& it, Compare&& cmp) -> meta::when<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "inplace_merge requires BidirectionalIterators");
  return ::std::inplace_merge(
    ::std::begin(range),
    ::core::forward<BidirIt>(it),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2>
auto includes (Range1&& rng1, Range2&& rng2) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  bool
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "includes requires InputIterators");
  static_assert(is_input2, "includes requires InputIterators");
  return ::std::includes(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2)
  );
}

template <class Range1, class Range2, class Compare>
auto includes (Range1&& rng1, Range2&& rng2, Compare&& cmp) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  bool
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "includes requires InputIterators");
  static_assert(is_input2, "includes requires InputIterators");
  return ::std::includes(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2, class OutputIt>
auto set_difference (Range1&& rng1, Range2&& rng2, OutputIt&& it) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_difference requires InputIterators");
  static_assert(is_input2, "set_difference requires InputIterators");
  return ::std::set_difference(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range1, class Range2, class OutputIt, class Compare>
auto set_difference (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt&& it,
  Compare&& cmp
) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_difference requires InputIterators");
  static_assert(is_input2, "set_difference requires InputIterators");
  return ::std::set_difference(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2, class OutputIt>
auto set_intersection (Range1&& rng1, Range2&& rng2, OutputIt&& it) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_intersection requires InputIterators");
  static_assert(is_input2, "set_intersection requires InputIterators");
  return ::std::set_intersection(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range1, class Range2, class OutputIt, class Compare>
auto set_intersection (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt&& it,
  Compare&& cmp
) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_intersection requires InputIterators");
  static_assert(is_input2, "set_intersection requires InputIterators");
  return ::std::set_intersection(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2, class OutputIt>
auto set_symmetric_difference (Range1&& rng1, Range2&& rng2, OutputIt&& it) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_symmetric_difference requires InputIterators");
  static_assert(is_input2, "set_symmetric_difference requires InputIterators");
  return ::std::set_symmetric_difference(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range1, class Range2, class OutputIt, class Compare>
auto set_symmetric_difference (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt&& it,
  Compare&& cmp
) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_symmetric_difference requires InputIterators");
  static_assert(is_input2, "set_symmetric_difference requires InputIterators");
  return ::std::set_symmetric_difference(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2, class OutputIt>
auto set_union (Range1&& rng1, Range2&& rng2, OutputIt&& it) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_union requires InputIterators");
  static_assert(is_input2, "set_union requires InputIterators");
  return ::std::set_union(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it)
  );
}

template <class Range1, class Range2, class OutputIt, class Compare>
auto set_union (
  Range1&& rng1,
  Range2&& rng2,
  OutputIt&& it,
  Compare&& cmp
) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  decay_t<OutputIt>
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "set_union requires InputIterators");
  static_assert(is_input2, "set_union requires InputIterators");
  return ::std::set_union(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<OutputIt>(it),
    ::core::forward<Compare>(cmp)
  );
}

/* heap operations */
template <class Range>
auto is_heap (Range&& rng) -> enable_if_t<is_range<Range>::value, bool> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "is_heap requires RandomAccessIterators");
  return ::std::is_heap(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto is_heap (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "is_heap requires RandomAccessIterators");
  return ::std::is_heap(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto is_heap_until (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "is_heap_until requires RandomAccessIterators");
  return ::std::is_heap_until(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto is_heap_until (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "is_heap_until requires RandomAccessIterators");
  return ::std::is_heap_until(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto make_heap (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "make_heap requires RandomAccessIterators");
  return ::std::make_heap(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto make_heap (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "make_heap requires RandomAccessIterators");
  return ::std::make_heap(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto push_heap (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "push_heap requires RandomAccessIterators");
  return ::std::push_heap(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto push_heap (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "push_heap requires RandomAccessIterators");
  return ::std::push_heap(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto pop_heap (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "pop_heap requires RandomAccessIterators");
  return ::std::pop_heap(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto pop_heap (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "pop_heap requires RandomAccessIterators");
  return ::std::pop_heap(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto sort_heap (Range&& rng) -> enable_if_t<is_range<Range>::value> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "sort_heap requires RandomAccessIterators");
  return ::std::sort_heap(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto sort_heap (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_random = decltype(range)::is_random_access;
  static_assert(is_random, "sort_heap requires RandomAccessIterators");
  return ::std::sort_heap(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

/* min/max operations */
template <class Range>
auto max_element (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "max_element requires ForwardIterators");
  return ::std::max_element(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto max_element (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "max_element requires ForwardIterators");
  return ::std::max_element(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto min_element (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "min_element requires ForwardIterators");
  return ::std::min_element(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto min_element (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  decltype(::std::begin(::core::forward<Range>(rng)))
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "min_element requires ForwardIterators");
  return ::std::min_element(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto minmax_element (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  ::std::pair<
    decltype(::std::begin(::core::forward<Range>(rng))),
    decltype(::std::end(::core::forward<Range>(rng)))
  >
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "minmax_element requires ForwardIterators");
  return ::std::minmax_element(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto minmax_element (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  ::std::pair<
    range<decltype(::std::begin(::core::forward<Range>(rng)))>,
    range<decltype(::std::end(::core::forward<Range>(rng)))>
  >
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_forward = decltype(range)::is_forward;
  static_assert(is_forward, "minmax_element requires ForwardIterators");
  return ::std::minmax_element(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2>
auto lexicographical_compare (Range1&& rng1, Range2&& rng2) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  bool
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "lexicographical_compare requires InputIterators");
  static_assert(is_input2, "lexicographical_compare requires InputIterators");
  return ::std::lexicographical_compare(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2)
  );
}

template <class Range1, class Range2, class Compare>
auto lexicographical_compare (
  Range1&& rng1,
  Range2&& rng2,
  Compare&& cmp
) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  bool
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_input1 = decltype(range1)::is_input;
  static constexpr auto is_input2 = decltype(range2)::is_input;
  static_assert(is_input1, "lexicographical_compare requires InputIterators");
  static_assert(is_input2, "lexicographical_compare requires InputIterators");
  return ::std::lexicographical_compare(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::std::end(range2),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range1, class Range2>
auto is_permutation (Range1&& rng1, Range2&& rng2) -> enable_if_t<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  bool
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "is_permutation requires ForwardIterators");
  static_assert(is_forward2, "is_permutation requires ForwardIterators");
  return ::std::is_permutation(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2)
  );
}

template <class Range1, class Range2, class BinaryPredicate>
auto is_permutation (
  Range1&& rng1,
  Range2&& rng2,
  BinaryPredicate&& bp
) -> meta::when<
  meta::all_of<meta::list<Range1, Range2>, is_range>(),
  bool
> {
  auto range1 = make_range(::core::forward<Range1>(rng1));
  auto range2 = make_range(::core::forward<Range2>(rng2));
  static constexpr auto is_forward1 = decltype(range1)::is_forward;
  static constexpr auto is_forward2 = decltype(range2)::is_forward;
  static_assert(is_forward1, "is_permutation requires ForwardIterators");
  static_assert(is_forward2, "is_permutation requires ForwardIterators");
  return ::std::is_permutation(
    ::std::begin(range1),
    ::std::end(range1),
    ::std::begin(range2),
    ::core::forward<BinaryPredicate>(bp)
  );
}

template <class Range>
auto next_permutation (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "next_permutation requires BidirectionalIterators");
  return ::std::next_permutation(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto next_permutation (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "next_permutation requires BidirectionalIterators");
  return ::std::next_permutation(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

template <class Range>
auto prev_permutation (Range&& rng) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "prev_permutation requires BidirectionalIterators");
  return ::std::prev_permutation(::std::begin(range), ::std::end(range));
}

template <class Range, class Compare>
auto prev_permutation (Range&& rng, Compare&& cmp) -> enable_if_t<
  is_range<Range>::value,
  bool
> {
  auto range = make_range(::core::forward<Range>(rng));
  static constexpr auto is_bidir = decltype(range)::is_bidirectional;
  static_assert(is_bidir, "prev_permutation requires BidirectionalIterators");
  return ::std::prev_permutation(
    ::std::begin(range),
    ::std::end(range),
    ::core::forward<Compare>(cmp)
  );
}

}} /* namespace core::v2 */

#endif /* CORE_ALGORITHM_HPP */
