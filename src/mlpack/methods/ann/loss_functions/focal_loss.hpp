#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_FOCAL_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_FOCAL_LOSS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * @code
 * @article{Lin2017
 *   title   = {Focal Loss for Dense Object Detection},
 *   author  = {Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár},
 *   journal = {IEEE International Conference on Computer Vision (ICCV)},
 *   year    = {2017}
 * }
 * @endcode
 */

template<typename MatType = arma::mat>
class FocalLossType
{
 public:
  FocalLossType(
      const double gamma = 2.0,
      const double alpha = 0.25,
      const bool reduction = true);

  inline typename MatType::elem_type Forward(
      const MatType& prediction,
      const MatType& target);

  inline void Backward(
      const MatType& prediction,
      const MatType& target,
      MatType& loss);

  double Gamma() const { return gamma; }
  double& Gamma() { return gamma; }

  double Alpha() const { return alpha; }
  double& Alpha() { return alpha; }

  bool Reduction() const { return reduction; }
  bool& Reduction() { return reduction; }

  template<typename Archive>
  void serialize(
      Archive& ar,
      const uint32_t /* version */);

 private:
  double gamma;
  double alpha;
  bool reduction;
};

using FocalLoss = FocalLossType<arma::mat>;

} // namespace mlpack

#include "focal_loss_impl.hpp"

#endif
