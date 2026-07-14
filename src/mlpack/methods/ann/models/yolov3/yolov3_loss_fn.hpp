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
  /**
   * Create the YOLOv3Loss object, with default coefficients.
   */
  YOLOv3Loss() : YOLOv3Loss(1.0, 0.2, 1.0, 5.0) {}

  /**
   * Create the YOLOv3Loss object.
   *
   * Parameters are coefficients, which scale how much their particular loss effects
   * the total loss.
   */
  YOLOv3Loss(ElemType objectness,
             ElemType no_objectness,
             ElemType classification,
             ElemType coord);

  /**
   */
  ElemType Forward(const MatType& input, const MatType& target);

  /**
   */
  void Backward(const MatType& prediction,
                const MatType& target,
                MatType& loss);

  //! Serialize the EmptyLossType.
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
 private:

  ElemType objectness_coeff;
  ElemType no_objectness_coeff;
  ElemType classification_coeff;
  ElemType coordinate_coeff;
};

}; // namespace mlpack

// Include implementation.
#include "yolov3_loss_fn_impl.hpp"


#endif
