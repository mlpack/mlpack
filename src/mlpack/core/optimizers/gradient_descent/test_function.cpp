/**
 * @file test_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of very simple test function for gradient descent.
 */
#include "test_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

double GDTestFunction::Evaluate(const arma::mat& coordinates) const
{
  arma::vec temp = arma::trans(coordinates) * coordinates;
  return temp(0, 0);
}

void GDTestFunction::Gradient(const arma::mat& coordinates,
                              arma::mat& gradient) const
{
  gradient = 2 * coordinates;
}
