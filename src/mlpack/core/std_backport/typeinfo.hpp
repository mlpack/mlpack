#ifndef CORE_TYPEINFO_HPP
#define CORE_TYPEINFO_HPP

#include "type_traits.hpp"
#include "utility.hpp"

#ifndef CORE_NO_RTTI
  #include <typeindex>
  #include <typeinfo>
#endif /* CORE_NO_RTTI */

namespace core {
inline namespace v2 {

#ifndef CORE_NO_RTTI
using type_info = ::std::type_info;

template <class T>
type_info const& type_of () noexcept { return typeid(T); }
#else /* CORE_NO_RTTI */
struct type_info final {

  type_info (type_info const&) = delete;
  type_info () = delete;
  virtual ~type_info () = default;

  type_info& operator = (type_info const&) = delete;

  /* If we had C++14 template variables, this would actually be easier */
  template <class T>
  friend type_info const& type_of () noexcept {
    return type_info::cref<remove_reference_t<remove_cv_t<T>>>();
  }

  bool operator == (type_info const& that) const noexcept {
    return this->id == that.id;
  }

  bool operator != (type_info const& that) const noexcept {
    return this->id != that.id;
  }

  bool before (type_info const& that) const noexcept {
    return this->id < that.id;
  }

  ::std::size_t hash_code () const noexcept { return this->id; }

private:
  type_info (::std::uintptr_t id) noexcept : id { id } { }

  template <class T>
  static type_info const& cref () noexcept {
    static ::std::uintptr_t const value { };
    static type_info const instance { as_int(::std::addressof(value)) };
    return instance;
  }

  ::std::uintptr_t const id;
};

#endif /* CORE_NO_RTTI */

}} /* namespace core::v2 */

#endif /* CORE_TYPEINFO_HPP */
