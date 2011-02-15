/** @file math_lib.h
 *
 *  Includes additional math utilities not provided by the Boost
 *  library.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MATH_MATH_LIB_H
#define CORE_MATH_MATH_LIB_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <boost/math/special_functions/binomial.hpp>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include "core/table/dense_point.h"

namespace core {
namespace math {

class RandomNumberInit {
  private:
    const gsl_rng_type *global_generator_type_;
    gsl_rng *global_generator_;

  public:
    void set_seed(unsigned int seed_in) {
      gsl_rng_set(global_generator_, seed_in);
    }

    RandomNumberInit() {
      gsl_rng_env_setup();
      global_generator_type_ = gsl_rng_default;
      global_generator_ = gsl_rng_alloc(global_generator_type_);
      gsl_rng_set(global_generator_, time(NULL));
      std::cerr << "Random number generator initialized.\n";
    }

    ~RandomNumberInit() {
      gsl_rng_free(global_generator_);
    }

    template<typename T>
    T Random() {
      return gsl_rng_uniform(global_generator_);
    }

    double RandGaussian(double sigma) {
      return gsl_ran_gaussian(global_generator_, sigma);
    }

    /** @brief Returns a random integer from 0 to n - 1 inclusive.
     */
    int RandInt(int max_exclusive) {
      return gsl_rng_uniform_int(global_generator_, max_exclusive);
    }
};

extern core::math::RandomNumberInit global_random_number_state_;

template<typename T>
T BinomialCoefficient(unsigned n, unsigned k) {
  return (n < k) ? 0 : boost::math::binomial_coefficient<T>(n, k);
}

/** Squares a number. */
template<typename T>
inline T Sqr(T v) {
  return v * v;
}

/**
 * Forces a number to be non-negative, turning negative numbers into zero.
 *
 * Avoids branching costs (yes, we've discovered measurable improvements).
 */
template<typename T>
inline double ClampNonNegative(T d) {
  return (d + fabs(d)) / 2;
}

/**
 * Forces a number to be non-positive, turning positive numbers into zero.
 *
 * Avoids branching costs (yes, we've discovered measurable improvements).
 */
template<typename T>
inline double ClampNonPositive(T d) {
  return (d - fabs(d)) / 2;
}

/**
 * Clips a number between a particular range.
 *
 * @param value the number to clip
 * @param range_min the first of the range
 * @param range_max the last of the range
 * @return max(range_min, min(range_max, d))
 */
template<typename T>
inline double ClampRange(T value, T range_min, T range_max) {
  if(value <= range_min) {
    return range_min;
  }
  else if(value >= range_max) {
    return range_max;
  }
  else {
    return value;
  }
}

/**
 * Generates a uniform random number between 0 and 1.
 */
template<typename T>
inline double Random() {
  return global_random_number_state_.Random<T>();
}

/**
 * Generates a uniform random number in the specified range.
 */
template<typename T>
inline double Random(T lo, T hi) {
  return core::math::Random<T>() * (hi - lo) + lo;
}

template<typename T>
inline double RandGaussian(T sigma) {
  return global_random_number_state_.RandGaussian(sigma);
}

/**
 * Generates a uniform random integer in [0, n).
 */
template<typename T>
inline int RandInt(T hi_exclusive) {
  return global_random_number_state_.RandInt(hi_exclusive);
}

/**
 * Generates a uniform random integer in [lo, hi_exclusive).
 */
template<typename T>
inline int RandInt(T lo, T hi_exclusive) {
  return core::math::RandInt(hi_exclusive - lo) + lo;
}
};
};

#include "math_lib_impl.h"

namespace core {
namespace math {
/**
 * Calculates a relatively small power using template metaprogramming.
 *
 * This allows a numerator and denominator.  In the case where the
 * numerator and denominator are equal, this will not do anything, or in
 * the case where the denominator is one.
 */
template<int t_numerator, int t_denominator>
inline double Pow(double d) {
  return
    core::math__private::ZPowImpl<t_numerator, t_denominator>::Calculate(d);
}

/**
 * Calculates a small power of the absolute value of a number
 * using template metaprogramming.
 *
 * This allows a numerator and denominator.  In the case where the
 * numerator and denominator are equal, this will not do anything, or in
 * the case where the denominator is one.  For even powers, this will
 * avoid calling the absolute value function.
 */
template<int t_numerator, int t_denominator>
inline double PowAbs(double d) {
  // we specify whether it's an even function -- if so, we can sometimes
  // avoid the absolute value sign
  return core::math__private::ZPowAbsImpl < t_numerator, t_denominator,
         (t_numerator % t_denominator == 0) && ((t_numerator / t_denominator) % 2 == 0) >::Calculate(fabs(d));
}

template<typename T>
void RandomCombination(
  int begin, int end, int num_elements, std::vector<T> *combination) {

  for(int i = end - begin - num_elements; i < end - begin; i++) {
    int t = core::math::RandInt(0, i + 1) + begin;
    bool already_in_list = false;
    for(unsigned int j = 0; already_in_list == false &&
        j < combination->size(); j++) {
      already_in_list = ((*combination)[j] == t);
    }

    if(already_in_list == false) {
      combination->push_back(t);
    }
    else {
      combination->push_back(i + begin);
    }
  }
}

/** @brief Implements Algorithm 1 in "Fast Construction of $k$-Nearest
 *         Neighbor Graphs for Point Clouds" by Connor and Kumar, TVCG
 *         2009.
 */
template<typename T>
static int XorMsb(T a, T b) {

  typedef union {
    T float_rep_;
    int int_rep_;
  } float_helper;

  int a_exp, b_exp;
  float_helper a_mantissa, b_mantissa;
  a_mantissa.float_rep_ = frexp(a, &a_exp);
  b_mantissa.float_rep_ = frexp(b, &b_exp);
  if(a_exp == b_exp) {

    // Take the XOR of the two Mantissa bit representations and find
    // the most significant bit.
    int a_mantissa_xor_b_mantissa = a_mantissa.int_rep_ ^ b_mantissa.int_rep_;
    int most_significant_bit = 0;
    int shift_bit = 1;
    int num_bits_in_int = std::numeric_limits<int>::digits;
    for(int i = 0; i < num_bits_in_int; i++) {
      if((shift_bit & a_mantissa_xor_b_mantissa) != 0) {
        most_significant_bit = shift_bit;
      }
      shift_bit = shift_bit << 1;
    }
    return a_exp - shift_bit;
  }
  if(b_exp < a_exp) {
    return a_exp;
  }
  else {
    return b_exp;
  }
}

/** @brief Compares two vectors based on their Morton order.
 */
template<typename PointType>
inline bool MortonOrderPoints(const PointType &a, const PointType &b) {
  int x = 0;
  int selected_dim = 0;

  for(int d = 0; d < core::table::LengthTrait<PointType>::length(a); d++) {
    long int y = XorMsb(a[d], b[d]);
    if(x < y) {
      x = y;
      selected_dim = d;
    }
  }
  return a[selected_dim] < b[selected_dim];
}

template<typename T>
T SphereVolume(T r, int d) {
  int n = d / 2;
  double val;

  if(d % 2 == 0) {
    val = pow(r * boost::math::constants::root_pi<double>(), d) /
          boost::math::factorial<double>(n);
  }
  else {
    val = pow(2 * r, d) * pow(boost::math::constants::pi<double>(), n) *
          boost::math::factorial<double>(n) / boost::math::factorial<double>(d);
  }
  return val;
}
}
}

#endif
