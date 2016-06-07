/**
 * @file utils.hpp
 * @author Bang Liu
 *
 * Definition of utils.
 */
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>

namespace mlpack {
namespace ne {

/**
 * Definitions of different activate functions.
 */
inline double sigmoid(double x) {  // TODO: make it faster. More parameters?
  return 1.0 / (1.0 + exp(-x));
}

inline double relu(double x) {
  return (x > 0)? x:0;
}

}  // namespace mlpack
}  // namespace ne

#endif  // UTILS_HPP