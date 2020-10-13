/**
 * @file core/cereal/pointer_wrapper.hpp
 * @author Omar Shrit
 *
 * Implementation of a pointer wrapper to enable the serialization of
 * raw pointers in cereal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_POINTER_WRAPPER_HPP
#define MLPACK_CORE_CEREAL_POINTER_WRAPPER_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>

#if __cplusplus <= 201103L && !defined(_MSC_VER)
namespace std {
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namepace std
#endif

namespace cereal {

/**
 * The objective of this class is to create a wrapper for
 * raw pointer by encapsulating them in a smart pointer of type unique_ptr.
 *
 * Cereal does not support the serialization of raw pointer out of the box.
 * Therefore, we have created this wrapper to serialize raw pointer in cereal
 * as a smart pointer since because it will be difficult to change all pointer
 * type in mlpack.
 */
template<class T>
class PointerWrapper
{
 public:
  PointerWrapper(T*& pointer)
    : localPointer(pointer)
  {}

  template<class Archive>
  void save(Archive& ar, const uint32_t /*version*/) const
  {
    std::unique_ptr<T> smartPointer;
    if (this->localPointer != NULL)
      smartPointer = std::unique_ptr<T>(localPointer);
    ar(CEREAL_NVP(smartPointer));
    localPointer = smartPointer.release();
  }

  template<class Archive>
  void load(Archive& ar, const uint32_t /*version*/)
  {
    std::unique_ptr<T> smartPointer;
    ar(CEREAL_NVP(smartPointer));
    localPointer = smartPointer.release();
  }

  T*& release() { return localPointer; }

 private:
  T*& localPointer;
};

/**
 * Serialize raw pointer object by encapsulating the pointer into a smart
 * pointer.
 *
 * @param t A reference to raw pointer to be serialized.
 */
template<class T>
inline PointerWrapper<T>
make_pointer(T*& t)
{
  return PointerWrapper<T>(t);
}

/**
 * Cereal does not support the serialization of raw pointer.
 * This macro enable developers to serialize a raw pointer by using the
 * above PointerWrapper class which replace the raw pointer by a smart pointer
 * internally.
 *
 * @param T Raw pointer to be serialized.
 */
#define CEREAL_POINTER(T) cereal::make_pointer(T)

} // namespace cereal

#endif // CEREAL_POINTER_WRAPPER_HPP
