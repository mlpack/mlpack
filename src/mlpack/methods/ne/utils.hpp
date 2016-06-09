/**
 * @file utils.hpp
 * @author Bang Liu
 *
 * Definition of utils.
 */
#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cmath>
#include <ctime>

#include <boost/random.hpp>

typedef boost::mt19937 RNGType;  // Choose random number generator.

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

/**
 * Definitions of random number generation functions.
 */

// Random number generator.
RNGType rng;

// Set seed for random number generator
void Seed(int seedVal) {
  rng.seed(seedVal);
}

// Set seed by time for random number generator
void TimeSeed() {
  rng.seed(time(0));
}

// Returns randomly either 1 or -1
int RandPosNeg() {
  boost::random::uniform_int_distribution<> dist(0, 1);
  return dist(rng);
}

// Returns a random integer between [x, y]
// in case of ( 0 .. 1 ) returns 0
int RandInt(int x, int y) {
	boost::random::uniform_int_distribution<> dist(x, y);
    return dist(rng);
}

// Return a random float between [0, 1]
double RandFloat(double x, double y) {
  boost::random::uniform_01<> dist;
  return dist(rng);
  
}

// Return a random float between [x, y]
double RandFloat(double x, double y) {
  boost::random::uniform_real_distribution<> dist(x, y);
  return dist(rng);
}

}  // namespace mlpack
}  // namespace ne

#endif  // UTILS_HPP