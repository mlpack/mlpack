/**
 * @file cmaes.h
 * @author Kartik Nighania
 *
 * Covariance Matrix Adaptation Evolution Strategy
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */


#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_RANDOM_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_RANDOM_HPP

#include <iostream>
#include <ctime>
#include <cmath>

/**
 * @class Random
 * A simple random number generator from a normal distribution in the range [0,1]
 */

namespace mlpack {
namespace optimization {

template<typename T>
class Random 
{
  // variables for uniform()
  long int startseed;
  long int aktseed;
  long int aktrand;
  long int rgrand[32];
  bool stored;
  T hold;

public:
  /**
   * @param seed use clock if 0
   */
  Random(long unsigned seed = 0) : hold(0.0)
  {
    stored = false;
    if (seed < 1)
    {
      long int t = 100*time(0) + clock();
      seed = (long unsigned) (t < 0 ? -t : t);
    }
    start(seed);
  }
  
  void start(long unsigned seed)
  {
    stored = false;
    startseed = seed;
    if (seed < 1) seed = 1;
    aktseed = seed;
    for (int i = 39; i >= 0; --i)
    {
      long tmp = aktseed / 127773;
      aktseed = 16807* (aktseed - tmp* 127773) - 2836* tmp;
      if (aktseed < 0) aktseed += 2147483647;
      if (i < 32) rgrand[i] = aktseed;
    }
    aktrand = rgrand[0];
  }

  /**
   * @return (0,1)-normally distributed random number 
   */
  T gauss(void)
  {
    if (stored)
    {
      stored = false;
      return hold;
    }
    stored = true;
    T x1, x2, rquad;
    do {
      x1 = 2.0*uniform() - 1.0;
      x2 = 2.0*uniform() - 1.0;
      rquad = x1*x1 + x2*x2;
    } while (rquad >= 1 || rquad <= 0);
    const register T fac = std::sqrt(T(-2)*std::log(rquad)/rquad);
    hold = fac*x1;
    return fac*x2;
  }
  /**
   * @return (0,1)-uniform distributed random number
   */
  T uniform(void)
  {
    long tmp = aktseed / 127773;
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed += 2147483647;
    tmp = aktrand / 67108865;
    aktrand = rgrand[tmp];
    rgrand[tmp] = aktseed;
    return (T) aktrand / T(2.147483647e9);
  }
};

} //cmaes
} // optimizer
#endif