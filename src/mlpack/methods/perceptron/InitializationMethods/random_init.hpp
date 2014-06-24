/*
 *  @file: randominit.hpp
 *  @author: Udit Saxena
 *
 */

#ifndef _MLPACK_METHOS_PERCEPTRON_RANDOMINIT
#define _MLPACK_METHOS_PERCEPTRON_RANDOMINIT

#include <mlpack/core.hpp>
/*
This class is used to initialize weights for the 
weightVectors matrix in a random manner. 
*/
namespace mlpack {
namespace perceptron {
  class RandomInitialization
  {
  public:
    RandomInitialization()
    { }

    inline static void initialize(arma::mat& W, size_t row, size_t col)
    {
      W = arma::randu<arma::mat>(row,col);
    }
  }; // class RandomInitialization
}; // namespace perceptron
}; // namespace mlpack

#endif