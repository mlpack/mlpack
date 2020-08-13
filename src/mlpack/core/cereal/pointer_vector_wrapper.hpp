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

template<class T>
class pointer_vector_wrapper
{
/*
 * The objective of this class is to create a wrapper for
 * std::vector that hold pointers by adding also the size of the vector.
 * Cereal supports the serialization of the std vector, but 
 * we need to add the size of the vector if it holds a raw pointers.
 */
 public:
  pointer_vector_wrapper(std::vector<T*>& pointerVec)
    : pointerVector(pointerVec)
  {}

  template<class Archive>
  void save(Archive& ar) const
  {
    size_t vecSize = pointerVector.size();
    ar & CEREAL_NVP(vecSize);
    for (size_t i = 0; i < pointerVector.size(); ++i)
    {
      ar & CEREAL_POINTER(pointerVector.at(i));
    }
  }

  template<class Archive>
  void load(Archive& ar)
  {
    size_t vecSize = 0;
    ar & CEREAL_NVP(vecSize);
    pointerVector.resize(vecSize);
    for (size_t i = 0; i < pointerVector.size(); ++i)
    {
      ar & CEREAL_POINTER(pointerVector.at(i));
    }
  }

 private:
  std::vector<T*>& pointerVector;
};

template<class T>
inline pointer_vector_wrapper<T>
make_pointer_vector(std::vector<T*>& t)
{
  return pointer_vector_wrapper<T>(t);
}

#define CEREAL_VECTOR_POINTER(T) cereal::make_pointer_vector(T)

} // namespace cereal

#endif // CEREAL_POINTER_VECTOR_WRAPPER_HPP
