/**
 * @file deprecated.hpp
 * @author Marcos Pividori.
 *
 * Definition of the DEPRECATED macro.
 */
#ifndef MLPACK_CORE_UTIL_DEPRECATED_HPP
#define MLPACK_CORE_UTIL_DEPRECATED_HPP

#ifdef __GNUG__
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

#endif
