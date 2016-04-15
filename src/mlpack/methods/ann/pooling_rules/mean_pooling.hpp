/**
 * @file mean_pooling.hpp
 * @author Shangtong Zhang
 *
 * Definition of the MeanPooling class, which implements mean pooling.
 */
#ifndef MLPACK_METHODS_ANN_POOLING_RULES_MEAN_POOLING_HPP
#define MLPACK_METHODS_ANN_POOLING_RULES_MEAN_POOLING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/*
 * The mean pooling rule for convolution neural networks. Average all values
 * within the receptive block.
 */
class MLPACK_API MeanPooling
{
 public:
  /*
   * Return the average value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   */
  template<typename MatType>
  double Pooling(const MatType& input)
  {
    return arma::mean(arma::mean(input));
  }

  /*
   * Set the average value within the receptive block.
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
