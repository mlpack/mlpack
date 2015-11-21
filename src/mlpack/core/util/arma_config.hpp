/**
 * @file arma_config_check.hpp
 * @author Ryan Curtin
 *
 * Using the contents of arma_config.hpp, try to catch the condition where the
 * user has included mlpack with ARMA_64BIT_WORD enabled but mlpack was compiled
 * without ARMA_64BIT_WORD enabled.  This should help prevent a long, drawn-out
 * debugging process where nobody can figure out why the stack is getting
 * mangled.
 */
#ifndef __MLPACK_CORE_UTIL_ARMA_CHECK_HPP
#define __MLPACK_CORE_UTIL_ARMA_CHECK_HPP



#endif
