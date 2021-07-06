#ifndef CORE_STRING_VIEW_HPP
#define CORE_STRING_VIEW_HPP

#include <initializer_list>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <string>
#include <limits>

#include <cstdlib>
#include <ciso646>

#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable:5030)
  #pragma warning(disable:4702)
#endif /* defined(_MSC_VER) */

namespace core {
inline namespace v2 {
namespace impl {

/* implementations of MurmurHash2 *Endian Neutral* (but not alignment!) */
template <::std::size_t=sizeof(::std::size_t)> struct murmur;
template <> struct murmur<4> {
  constexpr murmur () = default;

  ::std::uint32_t operator () (void const* p, ::std::size_t len) const noexcept {
    static constexpr ::std::uint32_t magic = UINT32_C(0x5BD1E995);
    static constexpr auto shift = 24;

    auto hash = static_cast<::std::uint32_t>(len);
    auto data = static_cast<::std::uint8_t const*>(p);

    while (len >= sizeof(::std::uint32_t)) {
      ::std::uint32_t mix = data[0];
      mix |= ::std::uint32_t(data[1]) <<  8;
      mix |= ::std::uint32_t(data[2]) << 16;
      mix |= ::std::uint32_t(data[3]) << 24;

      mix *= magic;
      mix ^= mix >> shift;
      mix *= magic;

      hash *= magic;
      hash ^= mix;

      data += sizeof(::std::uint32_t);
      len -= sizeof(::std::uint32_t);
    }

    switch (len) {
      case 3: hash ^= ::std::uint32_t(data[2]) << 16; [[clang::fallthrough]];
      case 2: hash ^= ::std::uint32_t(data[1]) <<  8; [[clang::fallthrough]];
      case 1: hash ^= ::std::uint32_t(data[0]);
              hash *= magic;
    }

    hash ^= hash >> 13;
    hash *= magic;
    hash ^= hash >> 15;

    return hash;
  }
};

template <> struct murmur<8> {
  constexpr murmur () = default;

  ::std::uint64_t operator () (void const* p, ::std::size_t len) const noexcept {
    static constexpr ::std::uint64_t magic = UINT64_C(0xC6A4A7935BD1E995);
    static constexpr auto shift = 47;

    ::std::uint64_t hash = len * magic;

    auto data = static_cast<::std::uint8_t const*>(p);

    while (len >= sizeof(::std::uint64_t)) {
      ::std::uint64_t mix = data[0];
      mix |= ::std::uint64_t(data[1]) <<  8;
      mix |= ::std::uint64_t(data[2]) << 16;
      mix |= ::std::uint64_t(data[3]) << 24;
      mix |= ::std::uint64_t(data[4]) << 32;
      mix |= ::std::uint64_t(data[5]) << 40;
      mix |= ::std::uint64_t(data[6]) << 48;
      mix |= ::std::uint64_t(data[7]) << 54;

      mix *= magic;
      mix ^= mix >> shift;
      mix *= magic;

      hash ^= mix;
      hash *= magic;

      data += sizeof(::std::uint64_t);
      len -= sizeof(::std::uint64_t);
    }

    switch (len & 7) {
      case 7: hash ^= ::std::uint64_t(data[6]) << 48; [[clang::fallthrough]];
      case 6: hash ^= ::std::uint64_t(data[5]) << 40; [[clang::fallthrough]];
      case 5: hash ^= ::std::uint64_t(data[4]) << 32; [[clang::fallthrough]];
      case 4: hash ^= ::std::uint64_t(data[3]) << 24; [[clang::fallthrough]];
      case 3: hash ^= ::std::uint64_t(data[2]) << 16; [[clang::fallthrough]];
      case 2: hash ^= ::std::uint64_t(data[1]) << 8;  [[clang::fallthrough]];
      case 1: hash ^= ::std::uint64_t(data[0]);
              hash *= magic;
    }

    hash ^= hash >> shift;
    hash *= magic;
    hash ^= hash >> shift;

    return hash;
  }
};

}}} /* namespace core::v2::impl */

namespace core {
inline namespace v2 {

#ifndef CORE_NO_EXCEPTIONS
[[noreturn]] inline void throw_out_of_range (char const* msg) {
  throw ::std::out_of_range { msg };
}
#else /* CORE_NO_EXCEPTIONS */
[[noreturn]] inline void throw_out_of_range (char const*) { ::std::abort(); }
#endif /* CORE_NO_EXCEPTIONS */

template <class CharT, class Traits=::std::char_traits<CharT>>
struct basic_string_view {
  using difference_type = ::std::ptrdiff_t;
  using value_type = CharT;
  using size_type = ::std::size_t;

  using reference = value_type const&;
  using pointer = value_type const*;

  using const_reference = reference;
  using const_pointer = pointer;

  using const_iterator = pointer;
  using iterator = const_iterator;

  using const_reverse_iterator = ::std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;

  using traits = Traits;

  static constexpr size_type npos = ::std::numeric_limits<size_type>::max();

  template <class Allocator>
  basic_string_view (
    ::std::basic_string<CharT, Traits, Allocator> const& that
  ) : str { that.data() }, len { that.size() } { }

  constexpr basic_string_view (pointer str, size_type len) noexcept :
    str { str },
    len { len }
  { }

  basic_string_view (pointer str) noexcept :
    basic_string_view { str, traits::length(str) }
  { }

  constexpr basic_string_view (basic_string_view const&) noexcept = default;
  constexpr basic_string_view () noexcept = default;
  basic_string_view& operator = (basic_string_view const&) noexcept = default;

  template <class Allocator>
  explicit operator ::std::basic_string<CharT, Traits, Allocator> () const {
    return ::std::basic_string<CharT, Traits, Allocator> {
      this->data(),
      this->size()
    };
  }

  template <class Allocator=std::allocator<CharT>>
  ::std::basic_string<CharT, Traits, Allocator> to_string (
    Allocator const& allocator = Allocator()
  ) const {
    return ::std::basic_string<CharT, Traits, Allocator> {
      this->data(),
      this->size(),
      allocator
    };
  }

  constexpr const_iterator begin () const noexcept { return this->data(); }
  constexpr const_iterator end () const noexcept {
    return this->data() + this->size();
  }

  constexpr const_iterator cbegin () const noexcept { return this->begin(); }
  constexpr const_iterator cend () const noexcept { return this->end(); }

  const_reverse_iterator rbegin () const noexcept {
    return const_reverse_iterator { this->end()};
  }

  const_reverse_iterator rend () const noexcept {
    return const_reverse_iterator { this->begin() };
  }

  const_reverse_iterator crbegin () const noexcept { return this->rbegin(); }
  const_reverse_iterator crend () const noexcept { return this->rend(); }

  constexpr size_type max_size () const noexcept {
    return ::std::numeric_limits<size_type>::max();
  }
  constexpr size_type length () const noexcept { return this->size(); }
  constexpr size_type size () const noexcept { return this->len; }

  constexpr bool empty () const noexcept { return this->size() == 0; }

  constexpr reference operator [] (size_type idx) const {
    return this->str[idx];
  }

  constexpr reference front () const { return this->str[0]; }
  constexpr reference back () const { return this->str[this->size() - 1]; }
  constexpr pointer data () const { return this->str; }

  void remove_prefix (size_type n) {
    if (n > this->size()) { n = this->size(); }
    this->str += n;
    this->len -= n;
  }

  void remove_suffix (size_type n) {
    if (n > this->size()) { n = this->size(); }
    this->len -= n;
  }

  void clear () noexcept {
    this->str = nullptr;
    this->len = 0;
  }

  size_type copy (CharT* s, size_type n, size_type pos = 0) const {
    if (pos > this->size()) {
      throw_out_of_range("position greater than size");
    }
    auto const rlen = std::min(n, this->size() - pos);
    ::std::copy_n(this->begin() + pos, rlen, s);
    return rlen;
  }

  constexpr basic_string_view substr (
    size_type pos=0,
    size_type n=npos
  ) const noexcept {
    return pos > this->size()
      ? (throw_out_of_range("start position out of range"), *this)
      : basic_string_view {
        this->data() + pos,
        n == npos or pos + n > this->size()
          ? (this->size() - pos)
          : n
      };
  }

  bool starts_with (value_type value) const noexcept {
    return not this->empty() and traits::eq(value, this->front());
  }

  bool ends_with (value_type value) const noexcept {
    return not this->empty() and traits::eq(value, this->back());
  }

  bool starts_with (basic_string_view that) const noexcept {
    return this->size() >= that.size() and
      traits::compare(this->data(), that.data(), that.size()) == 0;
  }

  bool ends_with (basic_string_view that) const noexcept {
    return this->size() >= that.size() and
      traits::compare(
        this->data() + this->size() - that.size(),
        that.data(),
        that.size()
      ) == 0;
  }

  /* compare */
  difference_type compare (basic_string_view s) const noexcept {
    auto cmp = traits::compare(
      this->data(),
      s.data(),
      ::std::min(this->size(), s.size())
    );

    if (cmp != 0) { return cmp; }
    if (this->size() == s.size()) { return 0; }
    if (this->size() < s.size()) { return -1; }
    return 1;
  }

  difference_type compare (
    size_type pos,
    size_type n,
    basic_string_view s
  ) const noexcept { return this->substr(pos, n).compare(s); }

  difference_type compare (
    size_type pos1,
    size_type n1,
    basic_string_view s,
    size_type pos2,
    size_type n2
  ) const noexcept {
    return this->substr(pos1, n1).compare(s.substr(pos2, n2));
  }

  difference_type compare (pointer s) const noexcept {
    return this->compare(basic_string_view { s });
  }

  difference_type compare (
    size_type pos,
    size_type n,
    pointer s
  ) const noexcept {
    return this->substr(pos, n).compare(basic_string_view { s });
  }

  difference_type compare (
    size_type pos,
    size_type n1,
    pointer s,
    size_type n2
  ) const noexcept {
    return this->substr(pos, n1).compare(basic_string_view { s, n2 });
  }

  reference at (size_type idx) const {
    static constexpr auto error = "requested index out of range";
    if (idx >= this->size()) { throw_out_of_range(error); }
    return this->str[idx];
  }

  /* find-first-not-of */
  size_type find_first_not_of (
    basic_string_view str,
    size_type pos = 0) const noexcept {
    if (pos > this->size()) { return npos; }
    auto begin = this->begin() + pos;
    auto end = this->end();
    auto const predicate = [str] (value_type v) { return str.find(v) == npos; };
    auto iter = std::find_if(begin, end, predicate);
    if (iter == end) { return npos; }
    return static_cast<size_type>(::std::distance(this->begin(), iter));
  }

  size_type find_first_not_of (
    pointer s,
    size_type pos,
    size_type n) const noexcept {
      return this->find_first_not_of(basic_string_view { s, n }, pos);
  }

  size_type find_first_not_of (pointer s, size_type pos = 0) const noexcept {
    return this->find_first_not_of(basic_string_view { s }, pos);
  }

  size_type find_first_not_of (value_type c, size_type pos = 0) const noexcept {
    return this->find_first_not_of(
      basic_string_view { ::std::addressof(c), 1 },
      pos);
  }

  /* find-first-of */
  size_type find_first_of (
    basic_string_view str,
    size_type pos = 0) const noexcept {
    if (pos > this->size()) { return npos; }
    auto iter = ::std::find_first_of(
      this->begin() + pos, this->end(),
      str.begin(), str.end(),
      traits::eq);
    if (iter == this->end()) { return npos; }
    return static_cast<size_type>(::std::distance(this->begin(), iter));
  }

  size_type find_first_of (pointer s, size_type p, size_type n) const noexcept {
    return this->find_first_of(basic_string_view { s, n }, p);
  }

  size_type find_first_of (pointer s, size_type pos = 0) const noexcept {
    return this->find_first_of(basic_string_view { s }, pos);
  }

  size_type find_first_of (value_type c, size_type pos = 0) const noexcept {
    return this->find_first_of(
      basic_string_view { ::std::addressof(c), 1 },
      pos);
  }

  /* find */
  size_type find (basic_string_view str, size_type pos = 0) const noexcept {
    if (pos >= this->size()) { return npos; }
    auto iter = ::std::search(
      this->begin() + pos, this->end(),
      str.begin(), str.end(),
      traits::eq);
    if (iter == this->end()) { return npos; }
    return static_cast<size_type>(::std::distance(this->begin(), iter));
  }

  size_type find (pointer s, size_type p, size_type n) const noexcept {
    return this->find(basic_string_view { s, n }, p);
  }

  size_type find (pointer s, size_type pos = 0) const noexcept {
    return this->find(basic_string_view { s }, pos);
  }

  size_type find (value_type c, size_type pos = 0) const noexcept {
    return this->find(basic_string_view { ::std::addressof(c), 1 }, pos);
  }

  size_type find_last_not_of (
    basic_string_view str,
    size_type pos = npos) const noexcept {
    auto const offset = this->size() - ::std::min(this->size(), pos);
    auto begin = this->rbegin() + static_cast<difference_type>(offset);
    auto end = this->rend();
    auto const predicate = [str] (value_type v) { return str.find(v) == npos; };
    auto iter = ::std::find_if(begin, end, predicate);
    if (iter == end) { return npos; }
    auto const distance = static_cast<size_type>(
      ::std::distance(this->rbegin(), iter));
    return this->size() - distance - 1;
  }

  size_type find_last_not_of (
    pointer s,
    size_type p,
    size_type n) const noexcept {
    return this->find_last_not_of(basic_string_view { s, n }, p);
  }

  size_type find_last_not_of (pointer s, size_type p = npos) const noexcept {
    return this->find_last_not_of(basic_string_view { s }, p);
  }

  size_type find_last_not_of (
    value_type c,
    size_type pos = npos) const noexcept {
    return this->find_last_not_of(
      basic_string_view { ::std::addressof(c), 1 },
      pos);
  }

  size_type find_last_of (
    basic_string_view str,
    size_type pos = npos) const noexcept {
    auto const offset = this->size() - ::std::min(this->size(), pos);
    auto begin = this->rbegin() + static_cast<difference_type>(offset);
    auto end = this->rend();

    auto iter = ::std::find_first_of(
      begin, end,
      str.rbegin(), str.rend(),
      traits::eq);
    if (iter == end) { return npos; }
    auto const distance = static_cast<size_type>(
      ::std::distance(this->rbegin(), iter));
    return this->size() - distance - 1;
  }

  size_type find_last_of (pointer s, size_type p, size_type n) const noexcept {
    return this->find_last_of(basic_string_view { s, n }, p);
  }

  size_type find_last_of (pointer s, size_type p=npos) const noexcept {
    return this->find_last_of(basic_string_view { s }, p);
  }

  size_type find_last_of (value_type c, size_type p=npos) const noexcept {
    return this->find_last_of(basic_string_view { ::std::addressof(c), 1 }, p);
  }

  size_type rfind (basic_string_view str, size_type pos=npos) const noexcept {
    auto const offset = this->size() - ::std::min(this->size(), pos);
    auto begin = this->rbegin() + offset;
    auto end = this->rend();
    auto iter = ::std::search(
      begin, end,
      str.rbegin(), str.rend(),
      traits::eq);
    if (iter == end) { return npos; }
    auto const distance = static_cast<size_type>(
      ::std::distance(this->rbegin(), iter));
    return this->size() - distance - 1;
  }

  size_type rfind (pointer s, size_type p, size_type n) const noexcept {
    return this->rfind(basic_string_view { s, n }, p);
  }

  size_type rfind (pointer s, size_type p=npos) const noexcept {
    return this->rfind(basic_string_view { s }, p);
  }

  size_type rfind (value_type c, size_type p=npos) const noexcept {
    return this->rfind(basic_string_view { ::std::addressof(c), 1 }, p);
  }

  void swap (basic_string_view& that) noexcept {
    using ::std::swap;
    swap(this->str, that.str);
    swap(this->len, that.len);
  }

private:
  pointer str { nullptr };
  size_type len { 0 };
};

using u32string_view = basic_string_view<char32_t>;
using u16string_view = basic_string_view<char16_t>;
using wstring_view = basic_string_view<wchar_t>;
using string_view = basic_string_view<char>;

/* string_view comparison string_view */
template <class CharT, typename Traits>
bool operator == (
  basic_string_view<CharT, Traits> lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return lhs.size() == rhs.size() and lhs.compare(rhs) == 0; }

template <class CharT, typename Traits>
bool operator != (
  basic_string_view<CharT, Traits> lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return lhs.size() != rhs.size() or lhs.compare(rhs) != 0; }

template <class CharT, typename Traits>
bool operator >= (
  basic_string_view<CharT, Traits> lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return lhs.compare(rhs) >= 0; }

template <class CharT, typename Traits>
bool operator <= (
  basic_string_view<CharT, Traits> lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return lhs.compare(rhs) <= 0; }

template <class CharT, typename Traits>
bool operator > (
  basic_string_view<CharT, Traits> lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return lhs.compare(rhs) > 0; }

template <class CharT, typename Traits>
bool operator < (
  basic_string_view<CharT, Traits> lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return lhs.compare(rhs) < 0; }

/* string_view comparison string */
template <class CharT, class Traits, class Allocator>
bool operator == (
  basic_string_view<CharT, Traits> lhs,
  ::std::basic_string<CharT, Traits, Allocator> const& rhs
) noexcept { return lhs == basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits, class Allocator>
bool operator != (
  basic_string_view<CharT, Traits> lhs,
  ::std::basic_string<CharT, Traits, Allocator> const& rhs
) noexcept { return lhs != basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits, class Allocator>
bool operator >= (
  basic_string_view<CharT, Traits> lhs,
  ::std::basic_string<CharT, Traits, Allocator> const& rhs
) noexcept { return lhs >= basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits, class Allocator>
bool operator <= (
  basic_string_view<CharT, Traits> lhs,
  ::std::basic_string<CharT, Traits, Allocator> const& rhs
) noexcept { return lhs <= basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits, class Allocator>
bool operator > (
  basic_string_view<CharT, Traits> lhs,
  ::std::basic_string<CharT, Traits, Allocator> const& rhs
) noexcept { return lhs > basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits, class Allocator>
bool operator < (
  basic_string_view<CharT, Traits> lhs,
  ::std::basic_string<CharT, Traits, Allocator> const& rhs
) noexcept { return lhs < basic_string_view<CharT, Traits> { rhs }; }

/* string comparison string_view */
template <class CharT, class Traits, class Allocator>
bool operator == (
  ::std::basic_string<CharT, Traits, Allocator> const& lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } == rhs; }

template <class CharT, class Traits, class Allocator>
bool operator != (
  ::std::basic_string<CharT, Traits, Allocator> const& lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } != rhs; }

template <class CharT, class Traits, class Allocator>
bool operator >= (
  ::std::basic_string<CharT, Traits, Allocator> const& lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } >= rhs; }

template <class CharT, class Traits, class Allocator>
bool operator <= (
  ::std::basic_string<CharT, Traits, Allocator> const& lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } <= rhs; }

template <class CharT, class Traits, class Allocator>
bool operator > (
  ::std::basic_string<CharT, Traits, Allocator> const& lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } > rhs; }

template <class CharT, class Traits, class Allocator>
bool operator < (
  ::std::basic_string<CharT, Traits, Allocator> const& lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } < rhs; }

/* string_view comparison CharT* */
template <class CharT, class Traits>
bool operator == (
  basic_string_view<CharT, Traits> lhs,
  CharT const* rhs
) noexcept { return lhs == basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits>
bool operator != (
  basic_string_view<CharT, Traits> lhs,
  CharT const* rhs
) noexcept { return lhs != basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits>
bool operator >= (
  basic_string_view<CharT, Traits> lhs,
  CharT const* rhs
) noexcept { return lhs >= basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits>
bool operator <= (
  basic_string_view<CharT, Traits> lhs,
  CharT const* rhs
) noexcept { return lhs <= basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits>
bool operator > (
  basic_string_view<CharT, Traits> lhs,
  CharT const* rhs
) noexcept { return lhs > basic_string_view<CharT, Traits> { rhs }; }

template <class CharT, class Traits>
bool operator < (
  basic_string_view<CharT, Traits> lhs,
  CharT const* rhs
) noexcept { return lhs < basic_string_view<CharT, Traits> { rhs }; }

/* CharT* comparison string_view */
template <class CharT, class Traits>
bool operator == (
  CharT const* lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } == rhs; }

template <class CharT, class Traits>
bool operator != (
  CharT const* lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } != rhs; }

template <class CharT, class Traits>
bool operator >= (
  CharT const* lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } >= rhs; }

template <class CharT, class Traits>
bool operator <= (
  CharT const* lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } <= rhs; }

template <class CharT, class Traits>
bool operator > (
  CharT const* lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } > rhs; }

template <class CharT, class Traits>
bool operator < (
  CharT const* lhs,
  basic_string_view<CharT, Traits> rhs
) noexcept { return basic_string_view<CharT, Traits> { lhs } < rhs; }

template <class CharT, class Traits>
::std::basic_ostream<CharT, Traits>& operator << (
  ::std::basic_ostream<CharT, Traits>& os,
  basic_string_view<CharT, Traits> const& str
) { return os << str.to_string(); }

template <class CharT, class Traits>
void swap (
  basic_string_view<CharT, Traits>& lhs,
  basic_string_view<CharT, Traits>& rhs
) noexcept { return lhs.swap(rhs); }

}} /* namespace core::v2 */

namespace std {

template <typename CharT, typename Traits>
struct hash<core::v2::basic_string_view<CharT, Traits>> {
  using argument_type = core::v2::basic_string_view<CharT, Traits>;
  using result_type = size_t;

  result_type operator ()(argument_type const& ref) const noexcept {
    static constexpr core::impl::murmur<sizeof(size_t)> hasher { };
    return hasher(ref.data(), ref.size());
  }
};

} /* namespace std */


#if defined(_MSC_VER)
  #pragma warning(pop)
#endif /* defined(_MSC_VER) */
#endif /* CORE_STRING_VIEW_HPP */
