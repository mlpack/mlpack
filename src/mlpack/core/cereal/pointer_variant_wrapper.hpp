/**
 * @file core/cereal/pointer_variant_wrapper.hpp
 * @author Omar Shrit
 *
 * Implementation of a boost::variant wrapper to enable the serialization of
 * the pointers inside boost variant in cereal
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_POINTER_VARIANT_WRAPPER_HPP
#define MLPACK_CORE_CEREAL_POINTER_VARIANT_WRAPPER_HPP

#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/boost_variant.hpp>

#include <boost/variant.hpp>
#include <boost/variant/variant_fwd.hpp>
#include <boost/variant/static_visitor.hpp>

#include "pointer_wrapper.hpp"

namespace cereal {

// Forward declaration.
template<typename... VariantTypes>
class pointer_variant_wrapper;

template<typename... VariantTypes>
inline pointer_variant_wrapper<VariantTypes...>
make_pointer_variant(boost::variant<VariantTypes...>& t)
{
  return pointer_variant_wrapper<VariantTypes...>(t);
}

template<class Archive>
struct save_visitor : public boost::static_visitor<void>
{
  save_visitor(Archive& ar) : ar(ar) {}

  template<class T>
  void operator()(const T* value) const
  {
    ar & CEREAL_POINTER(value);
  }

  template<typename... Types>
  void operator()(boost::variant<Types*...>& value) const
  {
    ar & make_pointer_variant(value);
  }

  Archive& ar;
};

template<typename T>
struct load_visitor : public boost::static_visitor<void>
{
  template<typename Archive, typename VariantType>
  static void load_impl(Archive& ar, VariantType& variant, std::true_type)
  {
    // Note that T will be a pointer type.
    T loadVariant;
    ar & CEREAL_POINTER(loadVariant);
    variant = loadVariant;
  }

  template<typename Archive, typename VariantType>
  static void load_impl(Archive& ar, VariantType& value, std::false_type)
  {
    // This must be a nested boost::variant.
    ar & make_pointer_variant(value);
  }

  template<typename Archive, typename VariantType>
  static void load(Archive& ar, VariantType& variant)
  {
    // Delegate to the proper load_impl() overload depending on whether T is a
    // pointer type.  If T is not a pointer type, then we expect it to be a
    // nested boost::variant.
    load_impl(ar, variant, typename std::is_pointer<T>::type());
  }
};

template<typename... VariantTypes>
class pointer_variant_wrapper
{
/*
 * The objective of this class is to create a wrapper for
 * boost::variant. 
 * Cereal supports the serialization of boost::variant, but 
 * we need to serialize it if it holds a raw pointers.
 */
 public:
  pointer_variant_wrapper(boost::variant<VariantTypes...>& pointerVar)
    : pointerVariant(pointerVar)
  {}

  template<class Archive>
  void save(Archive& ar) const
  {
    // which represent the index in std::variant.
    int which = pointerVariant.which();
    ar & CEREAL_NVP(which);
    save_visitor<Archive> s(ar);
    boost::apply_visitor(s, pointerVariant);
  }

  template<class Archive>
  void load(Archive& ar)
  {
    // Load the size of the serialized type.
    int which;
    ar & CEREAL_NVP(which);

    // Create function pointers to each overload of load_visitor<T>::load, for
    // all T in VariantTypes.
    using LoadFuncType = void(*)(Archive&, boost::variant<VariantTypes...>&);
    LoadFuncType loadFuncArray[] = { &load_visitor<VariantTypes>::load... };

    if (which >= int(sizeof(loadFuncArray)/sizeof(loadFuncArray[0])))
      throw std::runtime_error("Invalid 'which' selector when"
          "deserializing boost::variant");

    loadFuncArray[which](ar, pointerVariant);
  }

 private:
  boost::variant<VariantTypes...>& pointerVariant;
};

#define CEREAL_VARIANT_POINTER(T) cereal::make_pointer_variant(T)

} // namespace cereal

#endif // CEREAL_POINTER_VARIANT_WRAPPER_HPP
