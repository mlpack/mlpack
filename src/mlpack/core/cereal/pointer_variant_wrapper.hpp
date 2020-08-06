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

template<class Archive>
struct save_visitor : public boost::static_visitor<void>
{
  save_visitor(Archive& ar) : ar_(ar) {}
  
  template<class T>
  void operator()(const T* value) const 
  {
    ar_ & CEREAL_POINTER(value);
  }
  
  template<typename... VariantTypes>
  void operator()(Archive& ar, boost::variant<VariantTypes...>& variant) const
  {
    ar & CEREAL_VARIANT_POINTER(variant);
  }

  Archive& ar_;
};

struct load_visitor : public boost::static_visitor<void>
{
  template<class Archive, class VariantType>
  void operator()(Archive& ar, VariantType* variant) const 
  {
    VariantType* loadVariant;
    ar & CEREAL_POINTER(loadVariant);
    variant = loadVariant;
  }

  template<typename Archive, typename... VariantTypes>
  void operator()(Archive& ar, boost::variant<VariantTypes...>& variant) const
  {
    ar & CEREAL_VARIANT_POINTER(variant);
  }
};

template<typename... VariantTypes>
class pointer_variant_wrapper
{
/*
 * The objective of this class is to create a wrapper for
 * boost::variant. 
 * Cereal supports the serialization of boost::vairnat, but 
 * we need to serialize it if it holds a raw pointers.
 */
 public:
   pointer_variant_wrapper(boost::variant<VariantTypes*...>& PointerVar)
    : PointerVariant(PointerVar)
  {}

  template<class Archive>
  void save(Archive& ar) const
  {
    // which represent the index in std::variant.
    int which = PointerVariant.which();
    ar & CEREAL_NVP(which);
    save_visitor<Archive> s(ar);
    boost::apply_visitor(s, PointerVariant);
  }

  template<class Archive>
  void load(Archive& ar)
  {
    // Load the size of the serialized type.
    int which;
    ar & CEREAL_NVP(which);

    // A function pointer used to define which type is used from boost::visitor
    using LoadFuncType = 
        std::function<void(Archive &, 
            boost::variant<VariantTypes*...> &)>;

    // Basically I have inspired myself from the cereal Implementation.
    LoadFuncType loadFuncArray[0] = load_visitor(); 

    if(which >= int(sizeof(loadFuncArray)/sizeof(loadFuncArray[0])))
      throw std::runtime_error("Invalid 'which' selector when deserializing boost::variant");

    loadFuncArray[which](ar, PointerVariant);
  }

private:
   boost::variant<VariantTypes*...>& PointerVariant;
};

template<typename... VariantTypes>
inline pointer_variant_wrapper<VariantTypes...>
make_pointer_variant(boost::variant<VariantTypes*...>& t)
{
  return pointer_variant_wrapper<VariantTypes...>(t);
}

#define CEREAL_VARIANT_POINTER(T) cereal::make_pointer_variant(T)

} // end namespace cereal

#endif // CEREAL_POINTER_VARIANT_WRAPPER_HPP
