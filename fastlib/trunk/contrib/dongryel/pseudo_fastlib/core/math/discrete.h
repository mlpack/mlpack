/**
 * @file discrete.h
 *
 * Discrete math helpers.
 */

#ifndef CORE_MATH_DISCRETE_H
#define CORE_MATH_DISCRETE_H

#include <vector>
#include <math.h>

namespace core {
namespace math {
/**
 * Computes the factorial of an integer.
 */
__attribute__((const)) double Factorial(int d);

/**
 * Computes the binomial coefficient, n choose k for nonnegative integers
 * n and k
 *
 * @param n the first nonnegative integer argument
 * @param k the second nonnegative integer argument
 * @return the binomial coefficient n choose k
 */
__attribute__((const)) double BinomialCoefficient(int n, int k);

template<typename TAnyIntegerType>
inline bool IsPowerTwo(TAnyIntegerType i) {
  return (i & (i - 1)) == 0;
}
};
};

#endif
