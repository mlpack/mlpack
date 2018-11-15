/**
 * @file ensmallen.hpp
 *
 * This is the main header to include if you want to use the ensmallen library.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// NOTE: When using the ensmallen library in your code, only include the ensmallen.hpp header.
// NOTE: Do not include any of the files in the ensmallen_bits folder.
 
#ifndef ENSMALLEN_HPP
#define ENSMALLEN_HPP

// certain compilers are way behind the curve
#if (defined(_MSVC_LANG) && (_MSVC_LANG >= 201402L))
  #undef  ARMA_USE_CXX11
  #define ARMA_USE_CXX11
#endif

#include <armadillo>

#if !defined(ARMA_USE_CXX11)
  // armadillo automatically enables ARMA_USE_CXX11
  // when a C++11/C++14/C++17/etc compiler is detected
  #error "please enable C++11/C++14 mode in your compiler"
#endif

#if ((ARMA_VERSION_MAJOR < 6) || ((ARMA_VERSION_MAJOR == 6) && (ARMA_VERSION_MINOR < 500)))
  #error "need Armadillo version 6.500 or later"
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <climits>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <iostream>
#include <string>
#include <sstream>

// On Visual Studio, disable C4519 (default arguments for function templates)
// since it's by default an error, which doesn't even make any sense because
// it's part of the C++11 standard.
#ifdef _MSC_VER
  #pragma warning(disable : 4519)
#endif

#include "ensmallen_bits/config.hpp"
#include "ensmallen_bits/ens_version.hpp"
#include "ensmallen_bits/log.hpp" // TODO: should move to another place

#include "ensmallen_bits/problems/problems.hpp" // TODO: should move to another place

// TODO: remove mlpack bits from each of these files
#include "ensmallen_bits/ada_delta/ada_delta.hpp"
#include "ensmallen_bits/ada_grad/ada_grad.hpp"
#include "ensmallen_bits/adam/adam.hpp"
#include "ensmallen_bits/aug_lagrangian/aug_lagrangian.hpp"
#include "ensmallen_bits/bigbatch_sgd/bigbatch_sgd.hpp"
#include "ensmallen_bits/cmaes/cmaes.hpp"
#include "ensmallen_bits/cne/cne.hpp"

#include "ensmallen_bits/function.hpp" // TODO: should move to function/

#include "ensmallen_bits/fw/frank_wolfe.hpp"
#include "ensmallen_bits/gradient_descent/gradient_descent.hpp"
//#include "ensmallen_bits/grid_search/grid_search.hpp"
#include "ensmallen_bits/iqn/iqn.hpp"
#include "ensmallen_bits/katyusha/katyusha.hpp"
#include "ensmallen_bits/lbfgs/lbfgs.hpp"
#include "ensmallen_bits/line_search/line_search.hpp"
#include "ensmallen_bits/parallel_sgd/parallel_sgd.hpp"
#include "ensmallen_bits/proximal/proximal.hpp"
#include "ensmallen_bits/rmsprop/rmsprop.hpp"

#include "ensmallen_bits/sa/sa.hpp"
#include "ensmallen_bits/sarah/sarah.hpp"
#include "ensmallen_bits/scd/scd.hpp"
#include "ensmallen_bits/sdp/sdp.hpp"
#include "ensmallen_bits/sdp/lrsdp.hpp"
#include "ensmallen_bits/sdp/primal_dual.hpp"

#include "ensmallen_bits/sgd/sgd.hpp"
// TODO: this should probably be included in sgd.hpp
#include "ensmallen_bits/sgd/update_policies/gradient_clipping.hpp"
#include "ensmallen_bits/sgdr/sgdr.hpp"
#include "ensmallen_bits/sgdr/snapshot_ensembles.hpp"
#include "ensmallen_bits/sgdr/snapshot_sgdr.hpp"
#include "ensmallen_bits/smorms3/smorms3.hpp"
#include "ensmallen_bits/spalera_sgd/spalera_sgd.hpp"
#include "ensmallen_bits/svrg/svrg.hpp"

#endif
