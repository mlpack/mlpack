#pragma once
#ifndef MLPACK_ARMA_INC_HPP
#define MLPACK_ARMA_INC_HPP

// First, check if Armadillo was included before, warning if so.
#ifdef ARMA_INCLUDES
#pragma message "Armadillo was included before mlpack; this can sometimes cause\
 problems.  It should only be necessary to include <mlpack/core.hpp> and not \
<armadillo>."
#endif

// Now include Armadillo through the special mlpack extensions.
#include <mlpack/core/arma_extend/arma_extend.hpp>
#include <mlpack/core/util/arma_traits.hpp>

// Ensure that the user isn't doing something stupid with their Armadillo
// defines.
#include <mlpack/core/util/arma_config_check.hpp>

// On Visual Studio, disable C4519 (default arguments for function templates)
// since it's by default an error, which doesn't even make any sense because
// it's part of the C++11 standard.
#ifdef _MSC_VER
  #pragma warning(disable : 4519)
  #define ARMA_USE_CXX11
#endif

#endif
