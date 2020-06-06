/**
 * @file core/cv/metrics/ssim_index.hpp
 * @author Utkarsh Rai
 *
 * The SSIM (Structural Similarity) metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SSIM_HPP
#define MLPACK_CORE_CV_METRICS_SSIM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

 /**
 * The SSIM Index is a metric used for predicting the percieved quality of
 * images. It is a full reference metric, i.e. it measures the quality of a
 * processed image based on an initial distortion free image as reference.
 * Two images are exaclty x and y are exactly similar if
 * @f$\operatorname{SSIM}(x, y) = 1@f$. An SSIM index value 0 implies no
 * structural similarity.
 * The SSIM index tries to quatify the similarity a human eye would percieve
 * between two images. Since, in reality, our eye can focus on a small part
 * of the image only, in practice, we use a sliding Gaussian Window of size
 * 11X11, to calculate the value locally and then take its average. We use
 * a Gaussian Window beacause even when we look at a part of the entire
 * image, our focus is sharpest at the center.
 * The formula to calculate the SSIM index for two images is
 * @f$\operatorname{SSIM}(x, y)=\frac{\left(2 \mu_{x} \mu_{y}+c_{1}\right)
 * \left(2 \sigma_{x y}+c_{2}\right)}{\left(\mu_{x}^{2}+\mu_{y}^{2}+c_{1}
 * \right)\left(\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2}\right)}@f$
 * where @f$\mu@f$ denotes average, @f$\\sigma^{2}@f$ denotes variance and
 * @f$\sigma_{x y}@f$ deontes covariance of x and y.
*/
class SSIMIndex
{
 public:
  /**
   * @tparam DataType The type of image matrix
   * @param image The processed image
   * @param reference The distortion free reference image
   * @return The mean value of local SSIM Index
   */
  template<typename DataType>
  static double Evaluate(DataType const& image,
                         DataType const& reference);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the measurement.
   */
  static const bool NeedsMinimization = false;
}; // class SSIMIndex
} // namespace cv
} // namespace mlpack
// Include implementation.
#include "ssim_index_impl.hpp"

#endif
