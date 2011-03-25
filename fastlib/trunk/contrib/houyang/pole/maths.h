#ifndef MATHS_H
#define MATHS_H

#include <cstdio>
#include <cmath>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#include "sparsela.h"

namespace MathsPrivate {
  template < int t_numerator, int t_denominator = 1 > struct ZPowImpl {
    static double Calculate(double d) {
      return pow(d, t_numerator * 1.0 / t_denominator);
    }
  };

  template<int t_equal> struct ZPowImpl<t_equal, t_equal> {
    static double Calculate(double d) {
      return d;
    }
  };

  template<> struct ZPowImpl<1, 1> {
    static double Calculate(double d) {
      return d;
    }
  };

  template<> struct ZPowImpl<1, 2> {
    static double Calculate(double d) {
      return sqrt(d);
    }
  };

  template<> struct ZPowImpl<1, 3> {
    static double Calculate(double d) {
      return cbrt(d);
    }
  };

  template<int t_denominator> struct ZPowImpl<0, t_denominator> {
    static double Calculate(double d) {
      return 1;
    }
  };

  template<int t_numerator> struct ZPowImpl<t_numerator, 1> {
    static double Calculate(double d) {
      return ZPowImpl < t_numerator - 1, 1 >::Calculate(d) * d;
    }
  };
  // odd powers
  template<int t_numerator, int t_denominator, bool is_even> struct ZPowAbsImpl;

  template<int t_numerator, int t_denominator> struct ZPowAbsImpl<t_numerator, t_denominator, false> {
    static double Calculate(double d) {
      return ZPowImpl<t_numerator, t_denominator>::Calculate(fabs(d));
    }
  };
  // even powers
  template<int t_numerator, int t_denominator> struct ZPowAbsImpl<t_numerator, t_denominator, true> {
    static double Calculate(double d) {
      return ZPowImpl<t_numerator, t_denominator>::Calculate(d);
    }
  };
};

namespace Maths {
  // The square root of 2.
  const double SQRT2 = 1.41421356237309504880;
  // Base of the natural logarithm.
  const double E = 2.7182818284590452354;
  // Log base 2 of E.
  const double LOG2_E = 1.4426950408889634074;
  // Log base 10 of E.
  const double LOG10_E = 0.43429448190325182765;
  // Natural log of 2.
  const double LN_2 = 0.69314718055994530942;
  // Natural log of 10.
  const double LN_10 = 2.30258509299404568402;
  // The ratio of the circumference of a circle to its diameter.
  const double PI = 3.141592653589793238462643383279;
  // The ratio of the circumference of a circle to its radius.
  const double PI_2 = 1.57079632679489661923;

  //////////////////////////////////////////////////////////////////////////////
  // Calculates a small power of the absolute value of a double number
  // using template metaprogramming. This allows a numerator and denominator.  
  // In the case where the numerator and denominator are equal, 
  // this will not do anything, or in the case where the denominator is one. 
  // For even powers, this will avoid calling the absolute value function.
  //////////////////////////////////////////////////////////////////////////////
  template<int t_numerator, int t_denominator> inline double PowAbs(double d) {
    // specify whether it's an even function; if so, avoid the absolute value sign
    return MathsPrivate::ZPowAbsImpl < t_numerator, t_denominator,
      (t_numerator % t_denominator == 0) && ((t_numerator / t_denominator) % 2 == 0) >::Calculate(fabs(d));
  }
};
  
class RandomNumber {
 private:
  const gsl_rng_type *global_generator_type_;
  gsl_rng *global_generator_;
 public:
  void SetSeed(unsigned int seed_in) {
    gsl_rng_set(global_generator_, seed_in);
  }
    
  RandomNumber() {
    gsl_rng_env_setup();
    // By default, unless GSL_RNG_TYPE environment variable is
    // specified, gsl_rng_mt19937 (MT19937 generator of Makoto
    // Matsumoto and Takuji Nishimura) is the default.
    global_generator_type_ = gsl_rng_default;
    global_generator_ = gsl_rng_alloc(global_generator_type_);
    gsl_rng_set(global_generator_, time(NULL));
    std::cerr << "Random number generator initialized.\n";
  }

  ~RandomNumber() {
    gsl_rng_free(global_generator_);
  }
    
  inline T_VAL Random() {
    return gsl_rng_uniform(global_generator_);
  }
  
  inline T_VAL RandGaussian(double sigma) {
    return gsl_ran_gaussian(global_generator_, sigma);
  }
};

#endif
