#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

template<typename TargetType>
class QuantizationStrategy
{
 public:
  template<typename SourceMatType, typename TargetMatType>
  void QuantizeWeights(const SourceMatType& sourceWeights,
                       TargetMatType& targetWeights)
  {
    double scaleFactor = FindQuantizationScale(sourceWeights);
    targetWeights = arma::conv_to<TargetMatType>::from(
        arma::clamp(sourceWeights * scaleFactor, 
                    std::numeric_limits<TargetType>::min(),
                    std::numeric_limits<TargetType>::max()));
  }

 private:
  template<typename MatType>
  double FindQuantizationScale(const MatType& weights)
  {
    double maxAbs = arma::abs(weights).max();
    double targetTypeMax = std::numeric_limits<TargetType>::max();
    return targetTypeMax / maxAbs;
  }
};

} // namespace ann
} // namespace mlpack