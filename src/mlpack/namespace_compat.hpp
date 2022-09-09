/**
 * @file namespace_compat.hpp
 * @author Ryan Curtin
 *
 * This file is included for reverse compatibility with mlpack 3 and older code:
 * it introduces all the namespaces that were removed in mlpack 4.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_NAMESPACE_COMPAT_HPP
#define MLPACK_NAMESPACE_COMPAT_HPP

namespace mlpack {

namespace math { using namespace mlpack; }

}

#endif
