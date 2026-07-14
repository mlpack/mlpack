/**
 * @file methods/ann/loss_functions/yolov3_loss_fn_impl.hpp
 * @author Andrew Furey
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_MODELS_YOLOV3_LOSS_FN
#define MLPACK_METHODS_ANN_MODELS_YOLOV3_LOSS_FN

#include "yolov3_loss_fn.hpp"

namespace mlpack {

template<typename MatType>
YOLOv3Loss<MatType>::YOLOv3Loss(ElemType objectness,
                                ElemType no_objectness,
                                ElemType classification,
                                ElemType coordinate) :
    objectness_coeff(objectness),
    no_objectness_coeff(no_objectness),
    classification_coeff(classification),
    coordinate_coeff(coordinate)
{
  // Nothing to do here.
}

template<typename MatType>
typename YOLOv3Loss<MatType>::ElemType
YOLOv3Loss<MatType>::Forward(const MatType& prediction, const MatType& target)
{
  // binary cross entropy for objectness, no objectness and classification.
  // ciou for coord

  ElemType objectness_loss = 0;
  ElemType no_objectness_loss = 0;
  ElemType classification_loss = 0;
  ElemType coordinate_loss = 0;





  return objectness_loss * objectness_coeff +
         no_objectness_loss * no_objectness_coeff +
         classification_loss * classification_coeff +
         coordinate_loss * coordinate_coeff;
}

template<typename MatType>
void YOLOv3Loss<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{

}

} // namespace mlpack

#endif
