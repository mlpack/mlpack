/**
 * @file methods/ann/reduction_rules/add_reduction.hpp
 * @author Shubham Agrawal
 *
 * Reduction of a given network with add reduction rule.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_REDUCTION_RULES_ADD_REDUCTION_HPP
#define MLPACK_METHODS_ANN_REDUCTION_RULES_ADD_REDUCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to reduce the network with the add reduction rule.
 * @tparam MatType Type of matrix used.
 */
template<typename MatType>
class AddReductionType
{
 public:
  /**
   * Reduce the specified network and store the results in the given
   * parameter.
   *
   * @param layerOutputs Network output that should be reduced.
   * @param output The network output.
   */
  void Reduce(const std::vector<MatType>& layerOutputs,
              MatType& output)
  {
    output.zeros();
		for (size_t i = 0; i < layerOutputs.size(); i++)
		{
			output += layerOutputs[i];
		}
  }

	/**
   * Reduce the specified network and store the results in the given
   * parameter.
   *
   * @param input Network layer wise delta output.
   * @param networkSize The network size.
	 * @param layerDeltas The network deltas output.
   */
	void UnReduce(
			const MatType& input,
			const size_t networkSize,
			std::vector<MatType>& layerDeltas)
	{
		layerDeltas.resize(networkSize, MatType());
		for (size_t i = 0; i < networkSize; i++)
		{
			layerDeltas[i] = input;
		}
	}

	std::vector<size_t> ReduceSize(
			const std::vector<Layer<MatType>*>& network)
	{
		if (network.size() == 0)
		{
			return std::vector<size_t>();
		}
		const std::vector<size_t> networkSize = network[0]->OutputDimensions();
		for (size_t i = 1; i < network.size(); i++)
		{
			if (networkSize != network[i]->OutputDimensions())
			{
				Log::Fatal << "Network size mismatch." << std::endl;
			}
		}
		return networkSize;
	}
}; // class AddReduction

typedef AddReductionType<arma::mat> AddReduction;

} // namespace ann
} // namespace mlpack

#endif
