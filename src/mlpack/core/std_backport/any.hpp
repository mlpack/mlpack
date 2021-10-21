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
#ifndef CORE_ANY_HPP
#define CORE_ANY_HPP

#include <memory>

#include <cstdlib>
#include <cstring>

#include "type_traits.hpp"
#include "algorithm.hpp"
#include "typeinfo.hpp"
#include "utility.hpp"

#ifndef CORE_NO_EXCEPTIONS
#include <stdexcept>
#endif /* CORE_NO_EXCEPTIONS */

namespace core {
inline namespace v2 {
namespace impl {

using data_type = add_pointer_t<void>;

template <class T>
struct is_small final : meta::all_t<
  sizeof(decay_t<T>) <= sizeof(data_type),
  alignof(decay_t<T>) <= alignof(data_type),
  ::std::is_nothrow_copy_constructible<decay_t<T>>::value
> { };
template <> struct is_small<void> final : ::std::true_type { };

template <class T=void, bool=is_small<T>::value> struct dispatch;
template <> struct dispatch<void, true> {
  dispatch () noexcept = default;
  virtual ~dispatch () noexcept = default;

  virtual void clone (data_type const&, data_type&) const { }
  virtual void move (data_type&, data_type&) const noexcept { }
  virtual void destroy (data_type&) const noexcept { }
  virtual type_info const& type () const noexcept { return type_of<void>(); }
};

template <class T>
struct dispatch<T, true> final : dispatch<> {
  using value_type = T;
  using const_pointer = add_pointer_t<add_const_t<value_type>>;
  using pointer = add_pointer_t<value_type>;
  using allocator_type = ::std::allocator<value_type>;
  using allocator_traits = ::std::allocator_traits<allocator_type>;

  virtual void clone (data_type const& src, data_type& dst) const final {
    allocator_type alloc { };
    auto val = reinterpret_cast<add_const_t<const_pointer>>(&src);
    auto ptr = reinterpret_cast<pointer>(&dst);
    allocator_traits::construct(alloc, ptr, *val);
  }

  virtual void move (data_type& src, data_type& dst) const noexcept final {
    allocator_type alloc { };
    auto val = reinterpret_cast<pointer>(&src);
    auto ptr = reinterpret_cast<pointer>(&dst);
    allocator_traits::construct(alloc, ptr, ::core::move(*val));
  }

  virtual void destroy (data_type& src) const noexcept final {
    allocator_type alloc { };
    auto ptr = reinterpret_cast<pointer>(&src);
    allocator_traits::destroy(alloc, ptr);
  }

  virtual type_info const& type () const noexcept final {
    return type_of<value_type>();
  }
};

template <class T>
struct dispatch<T, false> final : dispatch<> {
  using value_type = T;
  using pointer = add_pointer_t<value_type>;
  using allocator_type = ::std::allocator<value_type>;
  using allocator_traits = ::std::allocator_traits<allocator_type>;

  virtual void clone (data_type const& src, data_type& dst) const final {
    allocator_type alloc { };
    auto const& value = *static_cast<add_const_t<pointer>>(src);
    auto ptr = allocator_traits::allocate(alloc, 1);
    auto scope = make_scope_guard([&alloc, ptr] {
      allocator_traits::deallocate(alloc, ptr, 1);
    });
    allocator_traits::construct(alloc, ptr, value);
    scope.dismiss();
    dst = ptr;
  }

  virtual void move (data_type& src, data_type& dst) const noexcept final {
    allocator_type alloc { };
    auto& value = *static_cast<pointer>(src);
    auto ptr = allocator_traits::allocate(alloc, 1);
    auto scope = make_scope_guard([&alloc, ptr] {
      allocator_traits::deallocate(alloc, ptr, 1);
    });
    allocator_traits::construct(alloc, ptr, ::core::move(value));
    scope.dismiss();
    dst = ptr;
  }

  virtual void destroy (data_type& src) const noexcept final {
    allocator_type alloc { };
    auto ptr = static_cast<pointer>(src);
    allocator_traits::destroy(alloc, ptr);
    allocator_traits::deallocate(alloc, ptr, 1);
  }

  virtual type_info const& type () const noexcept final {
    return type_of<value_type>();
  }
};

template <class T> dispatch<> const* lookup () noexcept {
  static dispatch<T> instance;
  return ::std::addressof(instance);
}

template <> inline dispatch<> const* lookup<void> () noexcept {
  static dispatch<> instance;
  return ::std::addressof(instance);
}

} /* namespace impl */

#ifndef CORE_NO_EXCEPTIONS
class bad_any_cast final : public ::std::bad_cast {
public:
  virtual char const* what () const noexcept override {
    return "bad any cast";
  }
};

[[noreturn]] inline void throw_bad_any_cast () { throw bad_any_cast { }; }
#else /* CORE_NO_EXCEPTIONS */
[[noreturn]] inline void throw_bad_any_cast () { ::std::abort(); }
#endif /* CORE_NO_EXCEPTIONS */

struct any final {

  template <class T> friend T const* any_cast (any const*) noexcept;
  template <class T> friend T* any_cast (any*) noexcept;

  any (any const& that) :
    table { that.table },
    data { nullptr }
  { this->table->clone(that.data, this->data); }

  any (any&& that) noexcept :
    table { that.table },
    data { nullptr }
  { this->table->move(that.data, this->data); }

  any () noexcept :
    table { impl::lookup<void>() },
    data { nullptr }
  { }

  template <
    class T,
    class=enable_if_t<not ::std::is_same<any, decay_t<T>>::value>
  > any (T&& value) :
    any { ::std::forward<T>(value), impl::is_small<T> { } }
  { }

  ~any () noexcept { this->clear(); }

  any& operator = (any const& that) {
    any { that }.swap(*this);
    return *this;
  }

  any& operator = (any&& that) noexcept {
    any { ::std::move(that) }.swap(*this);
    return *this;
  }

  template <
    class T,
    class=enable_if_t<not ::std::is_same<any, decay_t<T>>::value>
  > any& operator = (T&& value) {
    any {
      ::std::forward<T>(value),
      impl::is_small<T> { }
    }.swap(*this);
    return *this;
  }

  void swap (any& that) noexcept {
    using ::std::swap;
    swap(this->table, that.table);
    swap(this->data, that.data);
  }

  void clear () noexcept {
    this->table->destroy(this->data);
    this->table = impl::lookup<void>();
  }

  type_info const& type () const noexcept { return this->table->type(); }

  bool empty () const noexcept { return this->table == impl::lookup<void>(); }

private:
  impl::dispatch<> const* table;
  impl::data_type data;

  template <class T>
  any (T&& value, ::std::true_type&&) :
    table { impl::lookup<decay_t<T>>() },
    data { nullptr }
  {
    using value_type = decay_t<T>;
    using allocator_type = ::std::allocator<value_type>;
    using allocator_traits = ::std::allocator_traits<allocator_type>;
    allocator_type alloc { };
    auto pointer = reinterpret_cast<value_type*>(::std::addressof(this->data));
    allocator_traits::construct(alloc, pointer, ::core::forward<T>(value));
  }

  template <class T>
  any (T&& value, ::std::false_type&&) :
    table { impl::lookup<decay_t<T>>() },
    data { nullptr }
  {
    using value_type = decay_t<T>;
    using allocator_type = ::std::allocator<value_type>;
    using allocator_traits = ::std::allocator_traits<allocator_type>;
    allocator_type alloc { };
    auto pointer = allocator_traits::allocate(alloc, 1);
    allocator_traits::construct(alloc, pointer, ::core::forward<T>(value));
    this->data = pointer;
  }

  template <class T>
  T const* cast (::std::true_type&&) const {
    return reinterpret_cast<T const*>(::std::addressof(this->data));
  }

  template <class T>
  T* cast (::std::true_type&&) {
    return reinterpret_cast<T*>(::std::addressof(this->data));
  }

  template <class T>
  T const* cast (::std::false_type&&) const {
    return static_cast<T const*>(this->data);
  }

  template <class T>
  T* cast (::std::false_type&&) {
    return static_cast<T*>(this->data);
  }
};

template <class T>
T const* any_cast (any const* operand) noexcept {
  return operand and operand->type() == type_of<T>()
    ? operand->cast<T>(impl::is_small<T> { })
    : nullptr;
}

template <class T>
T* any_cast (any* operand) noexcept {
  return operand and operand->type() == type_of<T>()
    ? operand->cast<T>(impl::is_small<T> { })
    : nullptr;
}

template <
  class T,
  class=meta::when<
    meta::any<
      ::std::is_reference<T>::value,
      ::std::is_copy_constructible<T>::value
    >()
  >
> T any_cast (any const& operand) {
  using type = remove_reference_t<T>;
  auto pointer = any_cast<add_const_t<type>>(::std::addressof(operand));
  if (not pointer) { throw_bad_any_cast(); }
  return *pointer;
}

template <
  class T,
  class=meta::when<
    meta::any<
      ::std::is_reference<T>::value,
      ::std::is_copy_constructible<T>::value
    >()
  >
> T any_cast (any&& operand) {
  using type = remove_reference_t<T>;
  auto pointer = any_cast<type>(::std::addressof(operand));
  if (not pointer) { throw_bad_any_cast(); }
  return *pointer;
}

template <
  class T,
  class=meta::when<
    meta::any<
      ::std::is_reference<T>::value,
      ::std::is_copy_constructible<T>::value
    >()
  >
> T any_cast (any& operand) {
  using type = remove_reference_t<T>;
  auto pointer = any_cast<type>(::std::addressof(operand));
  if (not pointer) { throw_bad_any_cast(); }
  return *pointer;
}

inline void swap (any& lhs, any& rhs) noexcept { lhs.swap(rhs); }

}} /* namespace core::v2 */

#endif /* CORE_ANY_HPP */
