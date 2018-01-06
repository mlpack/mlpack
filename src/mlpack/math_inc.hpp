#pragma once
#ifndef MLPACK_MATH_INC_HPP
#define MLPACK_MATH_INC_HPP

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <math.h>

// But if it's not defined, we'll do it.
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#endif
