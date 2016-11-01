/**
 * @file deprecated.hpp
 * @author Marcos Pividori.
 *
 * Definition of the mlpack_deprecated macro.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_DEPRECATED_HPP
#define MLPACK_CORE_UTIL_DEPRECATED_HPP

#ifdef __GNUG__
#define mlpack_deprecated __attribute__((deprecated))
#elif defined(_MSC_VER)
#define mlpack_deprecated __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement mlpack_deprecated for this "
    "compiler")
#define mlpack_deprecated
#endif

#endif
