/**
 * @file core/util/util.hpp
 * @author Omar Shrit 
 *
 * Convenience include for everything in the mlpack/core.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_UTIL_HPP
#define MLPACK_CORE_UTIL_UTIL_HPP

#include "arma_config.hpp"
#include "arma_traits.hpp"
// mlpack::backtrace only for linux
#ifdef MLPACK_HAS_BFD_DL
  #include "backtrace.hpp"
#endif
#include "binding_details.hpp"
#include "conv_to.hpp"
#include "deprecated.hpp"
#include "ens_traits.hpp"
#include "forward.hpp"
#include "hyphenate_string.hpp"
#include "io.hpp"
#include "is_std_vector.hpp"
#include "log.hpp"
#include "nulloutstream.hpp"
#include "param.hpp"
#include "param_data.hpp"
#include "params.hpp"
#include "prefixedoutstream.hpp"
#include "program_doc.hpp"
#include "sfinae_utility.hpp"
#include "size_checks.hpp"
#include "timers.hpp"
#include "to_lower.hpp"
#include "version.hpp"

#endif
