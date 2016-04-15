/**
 * @file max_pooling.hpp
 * @author Shangtong Zhang
 *
 * Definition of the MaxPooling class, which implements max pooling.
 */
#ifndef MLPACK_METHODS_ANN_POOLING_RULES_MAX_POOLING_HPP
#define MLPACK_METHODS_ANN_POOLING_RULES_MAX_POOLING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/*
 * The max pooling rule for convolution neural networks. Take the maximum value
 * within the receptive block.
 */
class MaxPooling
{
 public:
  /*
   * Return the maximum value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   */
  template<typename MatType>
  double Pooling(const MatType& input)
  {
    return input.max();
  }

  /*
   * Set the maximum value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   * @param value The unpooled value.
   * @param output The unpooled output data.
   */
  template<typename MatType>
  void Unpooling(const MatType& input, const double value, MatType& output)
  {
    output = MatType(input.n_rows, input.n_cols);
    output.fill(value / input.n_elem);
  }
};

} // namespace ann
} // namespace mlpack

#endif
