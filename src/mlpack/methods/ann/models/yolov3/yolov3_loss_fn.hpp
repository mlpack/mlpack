/**
 * @file methods/ann/models/yolov3.hpp
 * @author Andrew Furey
 *
 * Definition of an object detection loss function, specifically for YOLOv3.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_LOSS_FN_HPP
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_LOSS_FN_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename MatType = arma::mat>
class YOLOv3Loss
{
 public:
  using ElemType = typename MatType::elem_type;
  using CubeType = typename GetCubeType<MatType>::type;

  YOLOv3Loss() { std::cout << "Default yolo loss.\n"; }
  /**
   */
  YOLOv3Loss(size_t numBoxes,
             size_t numTruths,
             size_t numAttributes) :
    numBoxes(numBoxes),
    numTruths(numTruths),
    numAttributes(numAttributes),
    keepObjectRange(repmat(regspace(0, numTruths - 1).t(), numAttributes, 1))
  {
  }

  // Expected shapes for each input:
  // prediction: (numAttributes * numBoxes, batchSize)
  // targets: (numAttributes * numTruths, batchSize)
  // besPredictionIndices: (numTruths, batchSize)
  // ignorePrediction: (numBoxes, batchSize)
  // scales: (numTruths, batchSize)
  // numTargets: (batchSize)
  /**
   */
  ElemType Forward(const MatType& predictions,
                   const MatType& targets,
                   const MatType& bestPredictionIndices,
                   const MatType& ignorePredictions,
                   const MatType& scales,
                   const MatType& numTargets);

  /**
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Serialize the EmptyLossType.
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
 private:

  size_t numBoxes;
  size_t numTruths;
  size_t numAttributes;
  MatType keepObjectRange;
};

}; // namespace mlpack

// Include implementation.
#include "yolov3_loss_fn_impl.hpp"


#endif
