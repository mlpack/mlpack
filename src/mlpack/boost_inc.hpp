#pragma once
#ifndef MLPACK_BOOST_INC_HPP
#define MLPACK_BOOST_INC_HPP

// Increase the number of template arguments for the boost list class.
#undef BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#undef BOOST_MPL_LIMIT_LIST_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 40

// We'll need the necessary boost::serialization features, as well as what we
// use with mlpack.  In Boost 1.59 and newer, the BOOST_PFTO code is no longer
// defined, but we still need to define it (as nothing) so that the mlpack
// serialization shim compiles.
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
// boost_backport.hpp handles the version and backporting of serialization (and
// other) features.
#include "mlpack/core/boost_backport/boost_backport_serialization.hpp"
// Boost 1.59 and newer don't use BOOST_PFTO, but our shims do.  We can resolve
// any issue by setting BOOST_PFTO to nothing.
#ifndef BOOST_PFTO
  #define BOOST_PFTO
#endif
#include <mlpack/core/data/has_serialize.hpp>
#include <mlpack/core/data/serialization_template_version.hpp>

// If we have Boost 1.58 or older and are using C++14, the compilation is likely
// to fail due to boost::visitor issues.  We will pre-emptively fail.
#if __cplusplus > 201103L && BOOST_VERSION < 105900
#error Use of C++14 mode with Boost < 1.59 is known to cause compilation \
problems.  Instead specify the C++11 standard (-std=c++11 with gcc or clang), \
or upgrade Boost to 1.59 or newer.
#endif

#endif
