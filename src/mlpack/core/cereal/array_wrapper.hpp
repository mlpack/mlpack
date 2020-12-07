/**
 * @file core/cereal/array_wrapper.hpp
 * @author Omar Shrit
 *
 * Implementation of an array wrapper.
 *
 * This implementation allows to seriliaze an array easily using cereal.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_ARRAY_WRAPPER_HPP
#define MLPACK_CORE_CEREAL_ARRAY_WRAPPER_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {

/**
 * This class is used as a shim for cereal to be able to serialize a raw pointer array.
 */
template<class T>
class ArrayWrapper
{
 public:
  ArrayWrapper(T*& addr, std::size_t& size) :
      arrayAddress(addr),
      arraySize(size)
  {}

  template<class Archive>
  void save(Archive& ar) const
  {
    ar(CEREAL_NVP(arraySize));
    for (size_t i = 0; i < arraySize; ++i)
      ar(cereal::make_nvp("item", arrayAddress[i]));
  }

  template<class Archive>
  void load(Archive& ar)
  {
    ar(CEREAL_NVP(arraySize));
    delete[] arrayAddress;
    if (arraySize == 0)
    {
      arrayAddress = NULL;
      return;
    }
    arrayAddress = new T[arraySize];
    for (size_t i = 0; i < arraySize; ++i)
      ar(cereal::make_nvp("item", arrayAddress[i]));
  }

 private:
  ArrayWrapper& operator=(ArrayWrapper rhs);

  T*& arrayAddress;
  size_t& arraySize;
};

/**
 * This function is used to serialized old c-style array
 *
 * @param t C Style array.
 * @param s the size of the array.
 */
template<class T, class S>
inline
ArrayWrapper<T> make_array(T*& t, S& s)
{
  return ArrayWrapper<T>(t, s);
}

/**
 * Cereal does not support the serialization of raw pointer.
 * This macro enable developers to serialize old c-style array by using the
 * above ArrayWrapper class which serialize each member independently.
 *
 * @param T C Style array.
 * @param S Size of the array.
 */
#define CEREAL_POINTER_ARRAY(T, S) cereal::make_array(T, S)

} // namespace cereal

#endif // CEREAL_ARRAY_WRAPPER_HPP
