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
#ifndef MLPACK_CORE_CEREAL_PONTER_WRAPPER_HPP
#define MLPACK_CORE_CEREAL_PONTER_WRAPPER_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>

#if __cplusplus <= 201103L
namespace std {
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)

{
      return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namepace std
#endif

namespace cereal {

template<class T>
class pointer_wrapper
{
/*
 * The objective of this class is to create a wrapper for
 * raw pointer by encapsulating them in a smart pointer unique_ptr
 * This will allow to serialize raw pointer in cereal as a smart pointer
 * because it will be difficult to change all pointer type in mlpack
 */
 public:
  pointer_wrapper(T*& pointer)
    : localPointer(pointer)
  {}

  template<class Archive>
  void save(Archive& ar, const unsigned int /*version*/) const
  {
    std::unique_ptr<T> smartPointer;
    if (this->localPointer != NULL)
      smartPointer = std::make_unique<T>(std::move(*this->localPointer));
    ar(CEREAL_NVP(smartPointer));
    localPointer = smartPointer.release();
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int /*version*/)
  {
    std::unique_ptr<T> smartPointer;
    ar(CEREAL_NVP(smartPointer));
    localPointer = smartPointer.release();
  }

  T*& release() { return localPointer; }

 private:
  T*& localPointer;
};

template<class T>
inline pointer_wrapper<T>
make_pointer(T*& t)
{
  return pointer_wrapper<T>(t);
}

#define CEREAL_POINTER(T) cereal::make_pointer(T)

} // namespace cereal

#endif // CEREAL_POINTER_WRAPPER_HPP
