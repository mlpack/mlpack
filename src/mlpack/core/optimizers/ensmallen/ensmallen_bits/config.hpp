/**
 * @file config.hpp
 * @author Conrad Sanderson
 * @author Marcus Edel
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#if !defined(ENS_PRINT_INFO)
  // #define ENS_PRINT_INFO
#endif

#if !defined(ENS_PRINT_WARN)
  // #define ENS_PRINT_WARN
#endif

#if defined(ARMA_USE_OPENMP)
  #undef  ENS_USE_OPENMP
  #define ENS_USE_OPENMP
#endif


//


#if defined(ENS_DONT_PRINT_INFO)
  #undef ENS_PRINT_INFO
#endif

#if defined(ENS_DONT_PRINT_WARN)
  #undef ENS_PRINT_WARN
#endif

#if defined(ENS_DONT_USE_OPENMP)
  #undef ENS_USE_OPENMP
#endif


//


#if defined(ENS_USE_OPENMP)
  #define ENS_PRAGMA_OMP_PARALLEL _Pragma("omp parallel")
  #define ENS_PRAGMA_OMP_ATOMIC   _Pragma("omp atomic")
#else
  #define ENS_PRAGMA_OMP_PARALLEL
  #define ENS_PRAGMA_OMP_ATOMIC
#endif
