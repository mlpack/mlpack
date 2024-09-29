/**
 * @file core/cereal/pointer_vector_wrapper.hpp
 * @author Omar Shrit
 *
 * Implementation of a vector wrapper to enable the serialization of
 * the size of the vector in cereal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_POINTER_VECTOR_WRAPPER_HPP
#define MLPACK_CORE_CEREAL_POINTER_VECTOR_WRAPPER_HPP

#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/vector.hpp>

#include "pointer_wrapper.hpp"

namespace cereal {

/**
 * The objective of this class is to create a wrapper for
 * std::vector that hold pointers by adding also the size of the vector.
 * Cereal supports the serialization of the std vector, but 
 * we need to serialize the vector if it holds raw pointer.
 * This wrapper uses the PointerWrapper we have already created, it serialize
 * each pointer in the vector independently.
 *
 * We need to add the size of the vector if it holds a raw pointers, during the
 * serialization, so we can know the size of the number pointer to allocated
 * during the deserialization process.
 */
template<class T>
class PointerVectorWrapper
{
 public:
  PointerVectorWrapper(std::vector<T*>& pointerVec)
    : pointerVector(pointerVec)
  {}

  template<class Archive>
  void save(Archive& ar) const
  {
    size_t vecSize = pointerVector.size();
    ar(CEREAL_NVP(vecSize));
    for (size_t i = 0; i < pointerVector.size(); ++i)
    {
      ar(CEREAL_POINTER(pointerVector.at(i)));
    }
  }

  template<class Archive>
  void load(Archive& ar)
  {
    size_t vecSize = 0;
    ar(CEREAL_NVP(vecSize));
    pointerVector.resize(vecSize);
    for (size_t i = 0; i < pointerVector.size(); ++i)
    {
      ar(CEREAL_POINTER(pointerVector.at(i)));
    }
  }

 private:
  std::vector<T*>& pointerVector;
};

/**
 * Serialize an std::vector that holds raw pointer object by encapsulating them
 * into a smart pointer.
 *
 * @param t A reference to std::vector that holds raw pointer to be serialized.
 */
template<class T>
inline PointerVectorWrapper<T>
make_pointer_vector(std::vector<T*>& t)
{
  return PointerVectorWrapper<T>(t);
}

/**
 * Cereal does not support the serialization of raw pointer.
 * This macro enable developers to serialize std vectors that holds raw
 * pointers by using the above PointerVectorWrapper class which replace the internal
 * raw pointers by smart pointer internally.
 *
 * @param T std::vector that holds raw pointer to be serialized.
 */
#define CEREAL_VECTOR_POINTER(T) cereal::make_pointer_vector(T)

} // namespace cereal

#endif // CEREAL_POINTER_VECTOR_WRAPPER_HPP
