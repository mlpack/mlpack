/***
 * @file mlpack_core.h
 *
 * Include all of the base components required to write MLPACK methods.
 */
#ifndef __MLPACK_CORE_H
#define __MLPACK_CORE_H

// First, standard includes.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <iostream>

// Defining __USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <math.h>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

// And then the Armadillo library.
#include <armadillo>

// Now MLPACK-specific includes.
#include <mlpack/core/data/dataset.h>
#include <mlpack/core/math/math_lib.h>
#include <mlpack/core/math/range.h>
#include <mlpack/core/math/kernel.h>
#include <mlpack/core/model/model.hpp>
#include <mlpack/core/model/save_restore_model.hpp>
#include <mlpack/core/file/textfile.h>
#include <mlpack/core/io/io.h>
#include <mlpack/core/io/log.h>
#include <mlpack/core/arma_extend/arma_extend.h>
#include <mlpack/core/kernels/kernel.h>

#endif
