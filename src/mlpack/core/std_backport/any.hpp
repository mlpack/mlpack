////////////////////////////////////////////////////////////////////////////////
/// \file any.hpp
///
/// \brief This header provides definitions from the C++ header <any>
////////////////////////////////////////////////////////////////////////////////

/*
  The MIT License (MIT)

  Copyright (c) 2020 Matthew Rodusek All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
#ifndef BPSTD_ANY_HPP
#define BPSTD_ANY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "detail/config.hpp"
#include "type_traits.hpp"  // enable_if_t, is_*
#include "utility.hpp"      // in_place_type_t, move, forward

#include <typeinfo>         // std::bad_cast, std::type_info
#include <initializer_list> // std::initializer_list
#include <new>              // placement-new
#include <cassert>          // assert

BPSTD_COMPILER_DIAGNOSTIC_PREAMBLE

namespace bpstd {

  class any;

  //============================================================================
  // class : bad_any_cast
  //============================================================================

  class bad_any_cast : public std::bad_cast
  {
    const char* what() const noexcept override;
  };

  //============================================================================
  // class : any
  //============================================================================

  //////////////////////////////////////////////////////////////////////////////
  /// \brief An object that can hold values of any type via type-erasure
  ///
  /// The class any describes a type-safe container for single values of any
  /// type.
  ///
  /// 1) An object of class any stores an instance of any type that satisfies
  /// the constructor requirements or is empty, and this is referred to as the
  /// state of the class any object. The stored instance is called the
  /// contained object. Two states are equivalent if they are either both
  /// empty or if both are not empty and if the contained objects are
  /// equivalent.
  ///
  /// 2) The non-member any_cast functions provide type-safe access to the
  /// contained object.
  ///
  /// This implementation uses small-buffer optimization to avoid dynamic
  /// memory if the object is below a (4 * sizeof(void*))
  //////////////////////////////////////////////////////////////////////////////
  class any
  {
    //--------------------------------------------------------------------------
    // Constructors / Destructor / Assignment
    //--------------------------------------------------------------------------
  public:

    /// \brief Constructs an any instance that does not contain any value
    any() noexcept;

    /// \brief Moves an any instance by moving the stored underlying value
    ///
    /// \post \p other is left valueless
    ///
    /// \param other the other instance to move
    any(any&& other) noexcept;

    /// \brief Copies an any instance by copying the stored underlying value
    ///
    /// \param other the other instance to copy
    any(const any& other);

    /// \brief Constructs this any using \p value for the underlying instance
    ///
    /// \param value the value to construct this any out of
    template<typename ValueType,
             typename=enable_if_t<!is_same<decay_t<ValueType>,any>::value &&
                                   is_copy_constructible<decay_t<ValueType>>::value>>
    // cppcheck-suppress noExplicitConstructor
    any(ValueType&& value);

    /// \brief Constructs an 'any' of type ValueType by forwarding \p args to
    ///        its constructor
    ///
    /// \note This constructor only participates in overload resolution if
    ///       ValueType is constructible from \p args
    ///
    /// \param args the arguments to forward to ValueType's constructor
    template<typename ValueType, typename...Args,
             typename=enable_if_t<is_constructible<decay_t<ValueType>,Args...>::value &&
                                  is_copy_constructible<decay_t<ValueType>>::value>>
    explicit any(in_place_type_t<ValueType>, Args&&...args);

    /// \brief Constructs an 'any' of type ValueType by forwarding \p args to
    ///        its constructor
    ///
    /// \note This constructor only participates in overload resolution if
    ///       ValueType is constructible from \p args
    ///
    /// \param il an initializer_list of arguments
    /// \param args the arguments to forward to ValueType's constructor
    template<typename ValueType, typename U, typename...Args,
             typename=enable_if_t<is_constructible<decay_t<ValueType>,std::initializer_list<U>,Args...>::value &&
                                  is_copy_constructible<decay_t<ValueType>>::value>>
    explicit any(in_place_type_t<ValueType>,
                 std::initializer_list<U> il,
                 Args&&...args);

    //--------------------------------------------------------------------------

    ~any();

    //--------------------------------------------------------------------------

    /// \brief Assigns the contents of \p other to this any
    ///
    /// \param other the other any to move
    /// \return reference to \c (*this)
    any& operator=(any&& other) noexcept;

    /// \brief Assigns the contents of \p other to this any
    ///
    /// \param other the other any to copy
    /// \return reference to \c (*this)
    any& operator=(const any& other);

    /// \brief Assigns \p value to this any
    ///
    /// \param value the value to assign
    /// \return reference to \c (*this)
    template<typename ValueType,
             typename=enable_if_t<!is_same<decay_t<ValueType>,any>::value &&
                                   is_copy_constructible<decay_t<ValueType>>::value>>
    any& operator=(ValueType&& value);

    //--------------------------------------------------------------------------
    // Modifiers
    //--------------------------------------------------------------------------
  public:

    /// \{
    /// \brief Emplaces a \c ValueType into this any, destroying the previous
    ///        value if it contained one
    ///
    /// \tparam ValueType the type to construct
    /// \param args the arguments to forward to \c ValueType's constructor
    /// \return reference to the constructed value
    template<typename ValueType, typename...Args,
              typename=enable_if_t<is_constructible<decay_t<ValueType>,Args...>::value &&
                                   is_copy_constructible<decay_t<ValueType>>::value>>
    decay_t<ValueType>& emplace(Args&&...args);
    template<typename ValueType, typename U, typename...Args,
              typename=enable_if_t<is_constructible<decay_t<ValueType>,std::initializer_list<U>,Args...>::value &&
                                   is_copy_constructible<decay_t<ValueType>>::value>>
    decay_t<ValueType>& emplace(std::initializer_list<U> il, Args&&...args );
    /// \}

    /// \brief Destroys the underlying stored value, leaving this any
    ///        empty.
    void reset() noexcept;

    /// \brief Swaps the contents of \c this with \p other
    ///
    /// \post \p other contains the old contents of \c this, and \c this
    ///       contains the old contents of \p other
    ///
    /// \param other the other any to swap contents with
    void swap(any& other) noexcept;

    //--------------------------------------------------------------------------
    // Observers
    //--------------------------------------------------------------------------
  public:

    /// \brief Checks whether this any contains a value
    ///
    /// \return \c true if this contains a value
    bool has_value() const noexcept;

    /// \brief Gets the type_info for the underlying stored type, or
    ///        \c typeid(void) if \ref has_value() returns \c false
    ///
    /// \return the typeid of the stored type
    const std::type_info& type() const noexcept;

    //--------------------------------------------------------------------------
    // Private Static Members / Types
    //--------------------------------------------------------------------------
  private:

    // Internal buffer size + alignment
    static constexpr auto buffer_size  = 4u * sizeof(void*);
    static constexpr auto buffer_align = alignof(void*);

    // buffer (for internal storage)
    using internal_buffer = typename aligned_storage<buffer_size,buffer_align>::type;

    union storage
    {
      internal_buffer internal;
      void*           external;
    };

    //--------------------------------------------------------------------------

    // trait to determine if internal storage is required
    template<typename T>
    using requires_internal_storage = bool_constant<
      (sizeof(T) <= buffer_size) &&
      ((buffer_align % alignof(T)) == 0) &&
      is_nothrow_move_constructible<T>::value
    >;

    //-----------------------------------------------------------------------

    template<typename T>
    struct internal_storage_handler;

    template<typename T>
    struct external_storage_handler;

    template<typename T>
    using storage_handler = conditional_t<
      requires_internal_storage<T>::value,
      internal_storage_handler<T>,
      external_storage_handler<T>
    >;

    //-----------------------------------------------------------------------

    enum class operation
    {
      destroy, ///< Operation for calling the underlying's destructor
      copy,    ///< Operation for copying the underlying value
      move,    ///< Operation for moving the underlying value
      value,   ///< Operation for accessing the underlying value
      type,    ///< Operation for accessing the underlying type
    };

    //-----------------------------------------------------------------------

    using storage_handler_ptr = const void*(*)(operation, const storage*,const storage*);

    template<typename T>
    friend T* any_cast(any*) noexcept;
    template<typename T>
    friend const T* any_cast(const any*) noexcept;

    //-----------------------------------------------------------------------
    // Private Members
    //-----------------------------------------------------------------------
  private:

    storage             m_storage;
    storage_handler_ptr m_storage_handler;
  };

  //=========================================================================
  // non-member functions : class : any
  //=========================================================================

  //-------------------------------------------------------------------------
  // utilities
  //-------------------------------------------------------------------------

  /// \brief Swaps the contents of \p lhs and \p rhs
  void swap(any& lhs, any& rhs) noexcept;

  //-------------------------------------------------------------------------
  // casts
  //-------------------------------------------------------------------------

  /// \{
  /// \brief Attempts to cast an any back to the underlying type T
  ///
  /// \throw bad_any_cast if \p any is not exactly of type \p T
  /// \tparam T the type to cast to
  /// \return the object
  template<typename T>
  T any_cast(any& operand);
  template<typename T>
  T any_cast(any&& operand);
  template<typename T>
  T any_cast(const any& operand);
  /// \}

  /// \{
  /// \brief Attempts to cast an any back to the underlying type T
  ///
  /// \tparam T the type to cast to
  /// \return pointer to the object if successfull, nullptr otherwise
  template<typename T>
  T* any_cast(any* operand) noexcept;
  template<typename T>
  const T* any_cast(const any* operand) noexcept;
  /// \}

} // namespace bpstd

//=============================================================================
// definitions : class : bad_any_cast
//=============================================================================

inline
const char* bpstd::bad_any_cast::what()
  const noexcept
{
  return "bad_any_cast";
}

//=============================================================================
// class : any::internal_storage_handler
//=============================================================================

template<typename T>
struct bpstd::any::internal_storage_handler
{
  template<typename...Args>
  static T* construct(storage& s, Args&&...args);

  template<typename U, typename...Args>
  static T* construct(storage& s, std::initializer_list<U> il, Args&&...args);

  static void destroy(storage& s);

  static const void* handle(operation op,
                            const storage* self,
                            const storage* other);
};

//=============================================================================
// definition : class : any::internal_storage_handler
//=============================================================================

template<typename T>
template<typename...Args>
inline BPSTD_INLINE_VISIBILITY
T* bpstd::any::internal_storage_handler<T>
  ::construct(storage& s, Args&&...args)
{
  return ::new(&s.internal) T(bpstd::forward<Args>(args)...);
}

template<typename T>
template<typename U, typename...Args>
inline BPSTD_INLINE_VISIBILITY
T* bpstd::any::internal_storage_handler<T>
  ::construct(storage& s, std::initializer_list<U> il, Args&&...args)
{
  return ::new(&s.internal) T(il, bpstd::forward<Args>(args)...);
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
void bpstd::any::internal_storage_handler<T>
  ::destroy(storage& s)
{
  auto* t = static_cast<T*>(static_cast<void*>(&s.internal));
  t->~T();
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
const void* bpstd::any::internal_storage_handler<T>
  ::handle(operation op,
           const storage* self,
           const storage* other)
{
  switch (op)
  {
    case operation::destroy:
    {
      assert(self != nullptr);
      BPSTD_UNUSED(other);

      destroy(const_cast<storage&>(*self));
      break;
    }

    case operation::copy:
    {
      assert(self != nullptr);
      assert(other != nullptr);

      // Copy construct from the internal storage
      const auto* p = reinterpret_cast<const T*>(&other->internal);
      construct( const_cast<storage&>(*self), *p);
      break;
    }

    case operation::move:
    {
      assert(self != nullptr);
      assert(other != nullptr);

      // Move construct from the internal storage. '
      const auto* p = reinterpret_cast<const T*>(&other->internal);
      construct(const_cast<storage&>(*self), bpstd::move(*const_cast<T*>(p)));
      break;
    }

    case operation::value:
    {
      assert(self != nullptr);
      BPSTD_UNUSED(other);

      // NOTE(bitwize): This seemingly arbitrary conversion is for proper
      //   type-safety/correctness. Otherwise, converting an aligned_storage_t*
      //   to void* and then to T* would violate strict-aliasing -- which
      //   would be undefined-behavior. Behavior is only well-defined for
      //   casts from void* to T* if the the void* originated from a T*.
      const auto* p = reinterpret_cast<const T*>(&self->internal);
      return static_cast<const void*>(p);
    }

    case operation::type:
    {
      BPSTD_UNUSED(self);
      BPSTD_UNUSED(other);

      return static_cast<const void*>(&typeid(T));
    }
  }
  return nullptr;
}

//=============================================================================
// class : any::external_storage_handler
//=============================================================================

template<typename T>
struct bpstd::any::external_storage_handler
{
  template<typename...Args>
  static T* construct(storage& s, Args&&...args);

  template<typename U, typename...Args>
  static T* construct(storage& s, std::initializer_list<U> il, Args&&...args);

  static void destroy(storage& s);

  static const void* handle(operation op,
                            const storage* self,
                            const storage* other);
};


//=============================================================================
// definition : class : any::external_storage_handler
//=============================================================================

template<typename T>
template<typename...Args>
inline BPSTD_INLINE_VISIBILITY
T* bpstd::any::external_storage_handler<T>
  ::construct(storage& s, Args&&...args)
{
  s.external = new T(bpstd::forward<Args>(args)...);
  return static_cast<T*>(s.external);
}

template<typename T>
template<typename U, typename...Args>
inline BPSTD_INLINE_VISIBILITY
T* bpstd::any::external_storage_handler<T>
  ::construct(storage& s, std::initializer_list<U> il, Args&&...args)
{
  s.external = new T(il, bpstd::forward<Args>(args)...);
  return static_cast<T*>(s.external);
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
void bpstd::any::external_storage_handler<T>
  ::destroy(storage& s)
{
  delete static_cast<T*>(s.external);
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
const void* bpstd::any::external_storage_handler<T>
  ::handle( operation op,
            const storage* self,
            const storage* other )
{
  switch (op)
  {
    case operation::destroy:
    {
      assert(self != nullptr);
      BPSTD_UNUSED(other);

      destroy(const_cast<storage&>(*self));
      break;
    }

    case operation::copy:
    {
      assert(self != nullptr);
      assert(other != nullptr);

      // Copy construct from the internal storage
      construct( const_cast<storage&>(*self),
                 *static_cast<const T*>(other->external));
      break;
    }

    case operation::move:
    {
      BPSTD_UNUSED(self != nullptr);
      assert(other != nullptr);

      const auto p = static_cast<const T*>(other->external);
      // Move construct from the internal storage. '
      construct(const_cast<storage&>(*self), bpstd::move(*const_cast<T*>(p)));
      break;
    }

    case operation::value:
    {
      assert(self != nullptr);
      BPSTD_UNUSED(other);

      // self->external was already created as a T*; no need to cast like in
      // internal.
      return self->external;
    }

    case operation::type:
    {
      BPSTD_UNUSED(self);
      BPSTD_UNUSED(other);

      return &typeid(T);
    }
  }
  return nullptr;
}

//=============================================================================
// definitions : class : any
//=============================================================================

//-----------------------------------------------------------------------------
// Constructors / Destructor / Assignment
//-----------------------------------------------------------------------------

inline BPSTD_INLINE_VISIBILITY
bpstd::any::any()
  noexcept
  : m_storage{},
    m_storage_handler{nullptr}
{

}

inline BPSTD_INLINE_VISIBILITY
bpstd::any::any(any&& other)
  noexcept
  : m_storage{},
    m_storage_handler{other.m_storage_handler}
{
  if (m_storage_handler != nullptr) {
    m_storage_handler(operation::move, &m_storage, &other.m_storage);
  }
}

inline BPSTD_INLINE_VISIBILITY
bpstd::any::any(const any& other)
  : m_storage{},
    m_storage_handler{nullptr}
{

  if (other.m_storage_handler != nullptr) {
    // Set handler after constructing, in case of exception
    const auto handler = other.m_storage_handler;

    handler(operation::copy, &m_storage, &other.m_storage);
    m_storage_handler = handler;
  }
}

template<typename ValueType, typename>
inline BPSTD_INLINE_VISIBILITY
bpstd::any::any(ValueType&& value)
  : m_storage{},
    m_storage_handler{nullptr}
{
  // Set handler after constructing, in case of exception
  using handler_type = storage_handler<decay_t<ValueType>>;

  handler_type::construct(m_storage, bpstd::forward<ValueType>(value));
  m_storage_handler = &handler_type::handle;
}

template<typename ValueType, typename...Args, typename>
inline BPSTD_INLINE_VISIBILITY
bpstd::any::any(in_place_type_t<ValueType>, Args&&...args)
  : m_storage{},
    m_storage_handler{nullptr}
{
  // Set handler after constructing, in case of exception
  using handler_type = storage_handler<decay_t<ValueType>>;

  handler_type::construct(m_storage, bpstd::forward<Args>(args)...);
  m_storage_handler = &handler_type::handle;
}

template<typename ValueType, typename U, typename...Args, typename>
inline BPSTD_INLINE_VISIBILITY
bpstd::any::any(in_place_type_t<ValueType>,
                       std::initializer_list<U> il,
                       Args&&...args)
  : m_storage{},
    m_storage_handler{nullptr}
{
  // Set handler after constructing, in case of exception
  using handler_type = storage_handler<decay_t<ValueType>>;

  handler_type::construct(m_storage, il, bpstd::forward<Args>(args)...);
  m_storage_handler = &handler_type::handle;
}

//-----------------------------------------------------------------------------

inline BPSTD_INLINE_VISIBILITY
bpstd::any::~any()
{
  reset();
}

//-----------------------------------------------------------------------------

inline BPSTD_INLINE_VISIBILITY
bpstd::any& bpstd::any::operator=(any&& other)
  noexcept
{
  reset();

  if (other.m_storage_handler != nullptr) {
    m_storage_handler = other.m_storage_handler;
    m_storage_handler(operation::move, &m_storage, &other.m_storage);
  }

  return (*this);
}

inline BPSTD_INLINE_VISIBILITY
bpstd::any& bpstd::any::operator=(const any& other)
{
  reset();

  if (other.m_storage_handler != nullptr) {
    // Set handler after constructing, in case of exception
    const auto handler = other.m_storage_handler;

    handler(operation::copy, &m_storage, &other.m_storage);
    m_storage_handler = handler;
  }

  return (*this);
}

template<typename ValueType, typename>
inline BPSTD_INLINE_VISIBILITY
bpstd::any& bpstd::any::operator=(ValueType&& value)
{
  using handler_type = storage_handler<decay_t<ValueType>>;

  reset();

  handler_type::construct(m_storage, bpstd::forward<ValueType>(value));
  m_storage_handler = &handler_type::handle;

  return (*this);
}

//-----------------------------------------------------------------------------
// Modifiers
//-----------------------------------------------------------------------------

template<typename ValueType, typename...Args, typename>
inline BPSTD_INLINE_VISIBILITY
bpstd::decay_t<ValueType>&
  bpstd::any::emplace(Args&&...args)
{
  using handler_type = storage_handler<decay_t<ValueType>>;

  reset();

  auto& result = *handler_type::construct(m_storage,
                                          bpstd::forward<Args>(args)...);
  m_storage_handler = &handler_type::handle;

  return result;
}

template<typename ValueType, typename U, typename...Args, typename>
inline BPSTD_INLINE_VISIBILITY
bpstd::decay_t<ValueType>&
  bpstd::any::emplace(std::initializer_list<U> il,
                      Args&&...args)
{
  using handler_type = storage_handler<decay_t<ValueType>>;

  reset();

  auto& result = *handler_type::construct(m_storage,
                                          il,
                                          bpstd::forward<Args>(args)...);
  m_storage_handler = &handler_type::handle;

  return result;
}

inline BPSTD_INLINE_VISIBILITY
void bpstd::any::reset()
  noexcept
{
  if (m_storage_handler != nullptr) {
    m_storage_handler(operation::destroy, &m_storage, nullptr);
    m_storage_handler = nullptr;
  }
}

inline BPSTD_INLINE_VISIBILITY
void bpstd::any::swap(any& other)
  noexcept
{
  using std::swap;

  if (m_storage_handler != nullptr && other.m_storage_handler != nullptr)
  {
    auto tmp = any{};

    // tmp := self
    tmp.m_storage_handler = m_storage_handler;
    m_storage_handler(operation::move, &tmp.m_storage, &m_storage);
    m_storage_handler(operation::destroy, &m_storage, nullptr);

    // self := other
    m_storage_handler = other.m_storage_handler;
    m_storage_handler(operation::move, &m_storage, &other.m_storage);
    m_storage_handler(operation::destroy, &other.m_storage, nullptr);

    // other := tmp
    other.m_storage_handler = tmp.m_storage_handler;
    other.m_storage_handler(operation::move, &other.m_storage, &tmp.m_storage);
  }
  else if (other.m_storage_handler != nullptr)
  {
    swap(m_storage_handler, other.m_storage_handler);

    // self := other
    m_storage_handler(operation::move, &m_storage, &other.m_storage);
    m_storage_handler(operation::destroy, &other.m_storage, nullptr);
  }
  else if (m_storage_handler != nullptr)
  {
    swap(m_storage_handler, other.m_storage_handler);

    // other := self
    other.m_storage_handler(operation::move, &other.m_storage, &m_storage);
    other.m_storage_handler(operation::destroy, &m_storage, nullptr);
  }
}

//-----------------------------------------------------------------------------
// Observers
//-----------------------------------------------------------------------------

inline BPSTD_INLINE_VISIBILITY
bool bpstd::any::has_value()
  const noexcept
{
  return m_storage_handler != nullptr;
}

inline BPSTD_INLINE_VISIBILITY
const std::type_info& bpstd::any::type()
  const noexcept
{
  if (has_value()) {
    auto* p = m_storage_handler(operation::type, nullptr, nullptr);
    return (*static_cast<const std::type_info*>(p));
  }
  return typeid(void);
}

//=============================================================================
// definition : non-member functions : class : any
//=============================================================================

//-----------------------------------------------------------------------------
// utilities
//-----------------------------------------------------------------------------

inline BPSTD_INLINE_VISIBILITY
void bpstd::swap(any& lhs, any& rhs)
  noexcept
{
  lhs.swap(rhs);
}

//-----------------------------------------------------------------------------
// casts
//-----------------------------------------------------------------------------

template<typename T>
inline BPSTD_INLINE_VISIBILITY
T bpstd::any_cast(any& operand)
{
  using underlying_type = remove_cvref_t<T>;

  static_assert(
    is_constructible<T, underlying_type&>::value,
    "A program is ill-formed if T is not constructible from U&"
  );

  auto* p = any_cast<underlying_type>(&operand);
  if (p == nullptr) {
    throw bad_any_cast{};
  }
  return static_cast<T>(*p);
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
T bpstd::any_cast(any&& operand)
{
  using underlying_type = remove_cvref_t<T>;

  static_assert(
    is_constructible<T, underlying_type>::value,
    "A program is ill-formed if T is not constructible from U"
  );

  auto* p = any_cast<underlying_type>(&operand);
  if (p == nullptr) {
    throw bad_any_cast{};
  }
  return static_cast<T>(bpstd::move(*p));
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
T bpstd::any_cast(const any& operand)
{
  using underlying_type = remove_cvref_t<T>;

  static_assert(
    is_constructible<T, const underlying_type&>::value,
    "A program is ill-formed if T is not constructible from const U&"
  );

  const auto* p = any_cast<underlying_type>(&operand);
  if (p == nullptr) {
    throw bad_any_cast{};
  }
  return static_cast<T>(*p);
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
T* bpstd::any_cast(any* operand)
  noexcept
{
  if (!operand) {
    return nullptr;
  }
  if (operand->type() != typeid(T)) {
    return nullptr;
  }

  auto p = operand->m_storage_handler(any::operation::value,
                                      &operand->m_storage,
                                      nullptr);
  return const_cast<T*>(static_cast<const T*>(p));
}

template<typename T>
inline BPSTD_INLINE_VISIBILITY
const T* bpstd::any_cast(const any* operand)
  noexcept
{
  if (!operand) {
    return nullptr;
  }
  if (operand->type() != typeid(T)) {
    return nullptr;
  }

  auto* p = operand->m_storage_handler(any::operation::value,
                                       &operand->m_storage,
                                       nullptr);
  return static_cast<const T*>(p);
}

BPSTD_COMPILER_DIAGNOSTIC_POSTAMBLE

#endif /* BPSTD_ANY_HPP */
