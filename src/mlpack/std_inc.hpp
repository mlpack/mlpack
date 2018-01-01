#pragma once
#ifndef MLPACK_STD_INC_HPP
#define MLPACK_STD_INC_HPP

// Defining _USE_MATH_DEFINES should set M_PI.
#define _USE_MATH_DEFINES
#include <cmath>

// But if it's not defined, we'll do it.
#ifndef M_PI
  #define M_PI 3.141592653589793238462643383279
#endif

// Next, standard includes.
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

#endif
