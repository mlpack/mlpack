/**
 * @file quantization_strategy.hpp
 * @author Your Name
 *
 * Definition of the QuantizationStrategy abstract base class and a simple
 * implementation.
 */
#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_STRATEGY_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_STRATEGY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

template<typename SourceMatType, typename TargetMatType>
class QuantizationStrategy
{
 public:
  virtual ~QuantizationStrategy() = default; // Modern C++ practice

  virtual void QuantizeWeights(const SourceMatType& sourceWeights,
                               TargetMatType& targetWeights) = 0;
};

template<typename SourceMatType, typename TargetMatType>
class SimpleQuantization : public QuantizationStrategy<SourceMatType, TargetMatType>
{
 public:
  void QuantizeWeights(const SourceMatType& sourceWeights,
                       TargetMatType& targetWeights) override
  {
    // Find the minimum and maximum values in the source weights
    double minVal = sourceWeights.min();
    double maxVal = sourceWeights.max();

    if (maxVal == minVal)
    {
      // Set scaleFactor to 1 and offset to minVal, so the quantized value is zero
      scaleFactor = 1.0;
      offset = minVal;
      targetWeights.fill(static_cast<typename TargetMatType::elem_type>(0));
      return;
    }

    // Calculate the scale factor and offset
    scaleFactor = (std::numeric_limits<typename TargetMatType::elem_type>::max() - 
                   std::numeric_limits<typename TargetMatType::elem_type>::min()) / (maxVal - minVal);
    offset = minVal;

    // Perform quantization with clamping
    targetWeights = arma::conv_to<TargetMatType>::from(
        arma::clamp((sourceWeights - offset) * scaleFactor,
                    static_cast<double>(std::numeric_limits<typename TargetMatType::elem_type>::min()),
                    static_cast<double>(std::numeric_limits<typename TargetMatType::elem_type>::max())));
  }

  double GetScaleFactor() const { return scaleFactor; }
  double GetOffset() const { return offset; }

 private:
  double scaleFactor{1.0}; // Initialize with a safe default value
  double offset{0.0}; // Initialize with a safe default value
};

} // namespace ann
} // namespace mlpack

#endif
