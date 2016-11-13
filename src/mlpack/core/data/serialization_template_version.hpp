/**
 * @file serialization_template_version.hpp
 * @author Ryan Curtin
 *
 * A better version of the BOOST_CLASS_VERSION() macro that supports templated
 * classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SERIALIZATION_TEMPLATE_VERSION_HPP
#define MLPACK_CORE_DATA_SERIALIZATION_TEMPLATE_VERSION_HPP

/**
 * Use this like BOOST_CLASS_VERSION(), but for templated classes.  The first
 * argument is the signature for the template.  Here is an example for
 * math::Range<eT>:
 *
 * BOOST_TEMPLATE_CLASS_VERSION(template<typename eT>, math::Range<eT>, 1);
 */
#define BOOST_TEMPLATE_CLASS_VERSION(SIGNATURE, T, N) \
namespace boost { \
namespace serialization { \
SIGNATURE \
struct version<mlpack::data::SecondShim<T>> \
{ \
  typedef mpl::int_<N> type; \
  typedef mpl::integral_c_tag tag; \
  BOOST_STATIC_CONSTANT(int, value = version::type::value); \
  BOOST_MPL_ASSERT(( \
      boost::mpl::less< \
          boost::mpl::int_<N>, \
          boost::mpl::int_<256> \
      > \
  )); \
}; \
} \
}

#endif
