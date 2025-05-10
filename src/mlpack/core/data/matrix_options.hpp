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
 public:
  MatrixOptionsBase(bool noTranspose = defaultNoTranspose) :
      DataOptionsBase<MatrixOptionsBase<Derived>>(),
      noTranspose(noTranspose)
  {
    // Do Nothing.
  }

  template<typename Derived2>
  explicit MatrixOptionsBase(const MatrixOptionsBase<Derived2>& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  template<typename Derived2>
  explicit MatrixOptionsBase(MatrixOptionsBase<Derived2>&& opts) :
      DataOptionsBase<MatrixOptionsBase<Derived>>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  // Inherit base class constructors.
  using DataOptionsBase<MatrixOptionsBase<Derived>>::DataOptionsBase;

  MatrixOptionsBase& operator=(const MatrixOptionsBase& other)
  {
    if (&other == this)
      return *this;

    if (other.noTranspose.has_value())
      noTranspose = other.noTranspose;
    DataOptionsBase<MatrixOptionsBase<Derived>>::CopyOptions(other);

    return *this;
  }

  MatrixOptionsBase& operator=(MatrixOptionsBase&& other)
  {
    if (&other == this)
      return *this;

    noTranspose = std::move(other.noTranspose);
    DataOptionsBase<MatrixOptionsBase<Derived>>::MoveOptions(std::move(other));

    return *this;
  }

  template<typename Derived2>
  MatrixOptionsBase& operator=(const MatrixOptionsBase<Derived2>& other)
  {
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
  MatrixOptionsBase& operator=(MatrixOptionsBase<Derived2>&& other)
  {
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
class EmptyMatrixOptions : public MatrixOptionsBase<EmptyMatrixOptions>
{
 public:
  void WarnBaseConversion(const char* /* dataDescription */) const { }
  static const char* DataDescription() { return "general data"; }
  void Reset() { }
};

using MatrixOptions = MatrixOptionsBase<EmptyMatrixOptions>;

} // namespace data
} // namespace mlpack

#endif
