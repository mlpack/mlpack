/**
 * @file methods/ann/layer/channel_shuffle.hpp
 * @author Abhinav Anand
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CHANNEL_SHUFFLE_HPP
#define MLPACK_METHODS_ANN_LAYER_CHANNEL_SHUFFLE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Definition and implementation of the Channel Shuffle Layer.
 *
 * Channel Shuffle divides the channels/units in a tensor into groups
 * and rearrange while keeping the original tensor shape.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{zhang2018shufflenet,
 *   author    = {Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun and
 *                Megvii Inc},
 *   title     = {Shufflenet: An extremely efficient convolutional neural
 *                network for mobile devices},
 *   year      = {2018},
 *   url       = {https://arxiv.org/pdf/1707.01083},
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class ChannelShuffle
{
 public:
  //! Create the Channel Shuffle object.
  ChannelShuffle();

  /**
   * The constructor for the Channel Shuffle.
   *
   * @param inRowSize Number of input rows.
   * @param inColSize Number of input columns.
   * @param depth Number of input slices.
   * @param group Number of groups for shuffling channels.
   */
  ChannelShuffle(const size_t inRowSize,
                 const size_t inColSize,
                 const size_t depth,
                 const size_t groupCount);

  /**
   * Forward pass through the layer.
   *
   * @param input The input matrix.
   * @param output The resulting interpolated output matrix.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass. Since the layer does not have any learn-able parameters,
   * we just have to down-sample the gradient to make its size compatible with
   * the input size.
   *
   * @param * (input) The input matrix.
   * @param gradient The computed backward gradient.
   * @param output The resulting down-sampled output.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /*input*/,
                const arma::Mat<eT>& gradient,
                arma::Mat<eT>& output);

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the row size of the input.
  size_t const& InRowSize() const { return inRowSize; }
  //! Modify the row size of the input.
  size_t& InRowSize() { return inRowSize; }

  //! Get the column size of the input.
  size_t const& InColSize() const { return inColSize; }
  //! Modify the column size of the input.
  size_t& InColSize() { return inColSize; }

  //! Get the depth of the input.
  size_t const& InDepth() const { return depth; }
  //! Modify the depth of the input.
  size_t& InDepth() { return depth; }

  //! Get the number of groups the channels is divided into.
  size_t const& InGroupCount() const { return groupCount; }
  //! Modify the number of groups the channels is divided into.
  size_t& InGroupCount() { return groupCount; }

  //! Get the shape of the input.
  size_t InputShape() const
  {
    return inRowSize;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored row size of the input.
  size_t inRowSize;
  //! Locally stored column size of the input.
  size_t inColSize;
  //! Locally stored depth of the input.
  size_t depth;
  //! Locally stored the number of groups the channels is divided into.
  size_t groupCount;
  //! Locally stored number of input points.
  size_t batchSize;
  //! Locally-stored delta object.
  OutputDataType delta;
  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class ChannelShuffle

} // namespace mlpack

// Include implementation.
#include "channel_shuffle_impl.hpp"

#endif
