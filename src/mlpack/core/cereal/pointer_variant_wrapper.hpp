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

#include "pointer_wrapper.hpp"

namespace cereal {

struct save : public boost::static_visitor<>
{
  template<class T>
  void operator()(T t) const 
  {
    CEREAL_NVP_POINTER(t);
  }
};

struct load : public boost::static_visitor<>
{
  template<class T>
  void operator()(T t) const 
  {
    CEREAL_NVP_POINTER(t);
  }
};

template<typename VariantType1, typename... VariantTypes>
class pointer_variant_wrapper
{
/*
 * The objective of this class is to create a wrapper for
 * boost::variant. 
 * Cereal supports the serialization of boost::vairnat, but 
 * we need to serialize it if it holds a raw pointers.
 */
 public:
   pointer_variant_wrapper(boost::variant<VariantType1*, VariantTypes*...>& PointerVar)
    : PointerVariant(PointerVar)
  {}

  template<class Archive>
  void save(Archive& ar) const
  { 
    boost::apply_visitor(save(), PointerVariant);
  }

  template<class Archive>
  void load(Archive& ar)
  {
    boost::apply_visitor(load(), PointerVariant);
  }

private:
   boost::variant<VariantType1*, VariantTypes*...>& PointerVariant;
};

template<typename VariantType1, typename... VariantTypes>
inline pointer_variant_wrapper<VariantType1, VariantTypes...>
make_pointer_variant(boost::variant<VariantType1*, VariantTypes*...>& t)
{
  return pointer_variant_wrapper<VariantType1, VariantTypes...>(t);
}

#define CEREAL_VARIANT_POINTER(T) cereal::make_pointer_variant(T)

} // end namespace cereal

#endif // CEREAL_POINTER_VARIANT_WRAPPER_HPP
