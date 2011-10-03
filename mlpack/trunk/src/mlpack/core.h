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
#include <cmath>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <iostream>

// And then the Armadillo library.
#include <armadillo>

// Now MLPACK-specific includes.
#include <mlpack/core/data/dataset.h>
#include <mlpack/core/math/math_lib.h>
#include <mlpack/core/math/range.h>
#include <mlpack/core/math/kernel.h>
#include <mlpack/core/file/textfile.h>
#include <mlpack/core/io/io.h>
#include <mlpack/core/arma_extend/arma_extend.h>

#endif
