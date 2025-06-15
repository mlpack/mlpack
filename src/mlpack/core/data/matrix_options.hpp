/**
 * @file core/data/matrix_options.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Data options, all possible options to load different data types and format
 * with specific settings into mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MATRIX_OPTIONS_HPP
#define MLPACK_CORE_DATA_MATRIX_OPTIONS_HPP

#include <mlpack/prereqs.hpp>

#include "data_options.hpp"

namespace mlpack {
namespace data {

template<typename Derived>
class MatrixOptionsBase : public DataOptionsBase<MatrixOptionsBase<Derived>>
{
 protected:
  // Users should not construct a MatrixOptionsBase directly.
  MatrixOptionsBase(std::optional<bool> noTranspose = std::nullopt) :
      DataOptionsBase<MatrixOptionsBase<Derived>>(),
      noTranspose(noTranspose)
  { }

 public:
  //
  // Handling for copying and moving MatrixOptionsBase of the exact same type.
  //

  MatrixOptionsBase(const DataOptionsBase<MatrixOptionsBase<Derived>>& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  MatrixOptionsBase(DataOptionsBase<MatrixOptionsBase<Derived>>&& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  MatrixOptionsBase& operator=(
      const DataOptionsBase<MatrixOptionsBase<Derived>>& otherIn)
  {
    const MatrixOptionsBase& other =
        static_cast<const MatrixOptionsBase&>(otherIn);

    if (&other == this)
      return *this;

    if (other.noTranspose.has_value())
      noTranspose = other.noTranspose;
    DataOptionsBase<MatrixOptionsBase<Derived>>::CopyOptions(other);

    return *this;
  }

  MatrixOptionsBase& operator=(
      DataOptionsBase<MatrixOptionsBase<Derived>>&& otherIn)
  {
    MatrixOptionsBase&& other = static_cast<MatrixOptionsBase&&>(otherIn);

    if (&other == this)
      return *this;

    noTranspose = std::move(other.noTranspose);
    DataOptionsBase<MatrixOptionsBase<Derived>>::MoveOptions(std::move(other));

    return *this;
  }

  //
  // Handling for copying and moving entirely different DataOptionsBase types.
  //

  template<typename Derived2>
  explicit MatrixOptionsBase(const DataOptionsBase<Derived2>& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  template<typename Derived2>
  explicit MatrixOptionsBase(DataOptionsBase<Derived2>&& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(const DataOptionsBase<Derived2>& otherIn)
  {
    // Call out to base operator=.
    return static_cast<MatrixOptionsBase&>(
        DataOptionsBase<MatrixOptionsBase>::operator=(otherIn));
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(DataOptionsBase<Derived2>&& otherIn)
  {
    // Call out to base operator=.
    return static_cast<MatrixOptionsBase&>(
        DataOptionsBase<MatrixOptionsBase>::operator=(std::move(otherIn)));
  }

  //
  // Handling for copying and moving different child types of MatrixOptionsBase.
  //

  template<typename Derived2>
  MatrixOptionsBase(const DataOptionsBase<MatrixOptionsBase<Derived2>>& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  template<typename Derived2>
  MatrixOptionsBase(DataOptionsBase<MatrixOptionsBase<Derived2>>&& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(
      const DataOptionsBase<MatrixOptionsBase<Derived2>>& otherIn)
  {
    const MatrixOptionsBase<Derived2>& other =
        static_cast<const MatrixOptionsBase<Derived2>&>(otherIn);

    if ((void*) &other == (void*) this)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    // Only copy options that have been set in the other object.
    if (other.noTranspose.has_value())
      noTranspose = other.NoTranspose();

    // Copy base members.
    DataOptionsBase<MatrixOptionsBase<Derived>>::CopyOptions(other);

    return *this;
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(
      DataOptionsBase<MatrixOptionsBase<Derived2>>&& otherIn)
  {
    MatrixOptionsBase<Derived2>&& other =
        static_cast<MatrixOptionsBase<Derived2>&&>(otherIn);

    if ((void*) this == (void*) &other)
      return *this;

    // Print warnings for any members that cannot be converted.
    const char* dataDesc = static_cast<const Derived&>(*this).DataDescription();
    static_cast<const Derived2&>(other).WarnBaseConversion(dataDesc);

    noTranspose = std::move(other.noTranspose);

    // Move base members.
    DataOptionsBase<MatrixOptionsBase<Derived>>::MoveOptions(std::move(other));

    return *this;
  }

  // Augment with the options of the other `MatrixOptionsBase`.
  template<typename Derived2>
  void Combine(const MatrixOptionsBase<Derived2>& other)
  {
    // Combine the noTranspose option.
    noTranspose =
        DataOptionsBase<MatrixOptionsBase<Derived>>::CombineBooleanOption(
        noTranspose, other.noTranspose, "NoTranspose()");

    // If the derived type is the same, we can take any options from it.
    if constexpr (std::is_same_v<Derived, Derived2>)
    {
      static_cast<Derived&>(*this).Combine(static_cast<const Derived2&>(other));
    }

    // If Derived is not the same as Derived2, we will have printed warnings in
    // the standalone operator+().
  }

  void WarnBaseConversion(const char* dataDescription) const
  {
    if (noTranspose.has_value() && noTranspose != defaultNoTranspose)
      this->WarnOptionConversion("noTranspose", dataDescription);

    // We may potentially need to print warnings for any other converted members
    // of the derived type.
    static_cast<const Derived&>(*this).WarnBaseConversion(dataDescription);
  }

  static const char* DataDescription() { return "matrix data"; }

  void Reset()
  {
    noTranspose.reset();

    // Reset any child members.
    static_cast<Derived&>(*this).Reset();
  }

  // Get whether or not we will avoid transposing the matrix during load.
  bool NoTranspose() const
  {
    return this->AccessMember(noTranspose, defaultNoTranspose);
  }
  // Modify whether or not we will avoid transposing the matrix during load.
  bool& NoTranspose()
  {
    return this->ModifyMember(noTranspose, defaultNoTranspose);
  }

 private:
  std::optional<bool> noTranspose;

  constexpr static const bool defaultNoTranspose = false;

  // For access to internal optional members.
  template<typename Derived2>
  friend class MatrixOptionsBase;
};

// This utility class is meant to be used as the Derived parameter for a matrix
// option that is not actually a derived type.  It provides the
// WarnBaseConversion() member, which does nothing.
class PlainMatrixOptions : public MatrixOptionsBase<PlainMatrixOptions>
{
 public:
  // Allow access to all public MatrixOptionsBase constructors and operators,
  // but with the PlainMatrixOptions type name.
  using MatrixOptionsBase::MatrixOptionsBase;
  using MatrixOptionsBase::operator=;

  // However, C++ does not allow inheriting copy and move constructors or
  // operators, and the inherited protected constructor will still be protected,
  // so forward those manually.
  PlainMatrixOptions(const std::optional<bool> noTranspose = std::nullopt) :
      MatrixOptionsBase(noTranspose) { }
  PlainMatrixOptions(const MatrixOptionsBase<PlainMatrixOptions>& other) :
      MatrixOptionsBase(other) { }
  PlainMatrixOptions(MatrixOptionsBase<PlainMatrixOptions>&& other) :
      MatrixOptionsBase(std::move(other)) { }

  PlainMatrixOptions& operator=(
      const MatrixOptionsBase<PlainMatrixOptions>& other)
  {
    return static_cast<PlainMatrixOptions&>(
        MatrixOptionsBase::operator=(other));
  }

  PlainMatrixOptions& operator=(MatrixOptionsBase<PlainMatrixOptions>&& other)
  {
    return static_cast<PlainMatrixOptions&>(
        MatrixOptionsBase::operator=(std::move(other)));
  }

  void WarnBaseConversion(const char* /* dataDescription */) const { }
  static const char* DataDescription() { return "general data"; }
  void Reset() { }
  void Combine(const PlainMatrixOptions&) { }
};

using MatrixOptions = PlainMatrixOptions;

// Boolean options.
static const MatrixOptions Transpose   = MatrixOptions(false);
static const MatrixOptions NoTranspose = MatrixOptions(true);

template<typename T>
struct IsDataOptions<MatrixOptionsBase<T>>
{
  constexpr static bool value = true;
};

template<>
struct IsDataOptions<PlainMatrixOptions>
{
  constexpr static bool value = true;
};

} // namespace data
} // namespace mlpack

#endif
