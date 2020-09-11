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
  // note: I would like to make the copy constructor private but this breaks
  // make_array.  So I make make_array a friend
  template<class Tx, class S>
  friend const cereal::ArrayWrapper<Tx> make_array(Tx*& t, S& s);

  T*& arrayAddress;
  size_t& arraySize;
};

/* This function is used to serialized old c-style array */
template<class T, class S>
inline
ArrayWrapper<T> make_array(T*& t, S& s)
{
  return ArrayWrapper<T>(t, s);
}

} // namespace cereal

#endif // CEREAL_ARRAY_WRAPPER_HPP
