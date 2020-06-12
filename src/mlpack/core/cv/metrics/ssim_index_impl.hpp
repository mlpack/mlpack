/**
 * @file core/cv/metrics/ssim_index_impl.hpp
 * @author Utkarsh Rai
 *
 * The SSIM (Structural Similarity) metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SSIMINDEX_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_SSIMINDEX_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

template<typename DataType>
double SSIMIndex::Evaluate(DataType const& image,
                           DataType const& reference)
{
  if (arma::size(reference) != arma::size(image))
  {
    std::ostringstream oss;
    oss << "SSIMIndex::Evaluate(): size of reference ("
        << reference.n_cols << "x" << reference.n_rows
        << ") does not match size of image ("
        << image.n_cols << "x" << image.n_rows << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
  if(reference.n_cols < 11 || reference.n_rows < 11)
  {
    std::ostringstream oss;
    oss << "SSIMIndex::Evaluate(): size of the images ("
        << reference.n_cols << "x" << reference.n_rows
        << ") is less than the size of the Gaussian Window (11x11)"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
  // Instantitating a Gaussian Kernel with bandwidth 1.5.
  kernel::GaussianKernel gaussianKernel = kernel::GaussianKernel(1.5);
  arma::vec point;
  // Gaussian Window of size 11x11.
  arma::mat gaussianWindow(11, 11);
  for ( size_t row = 0; row < 11; row += 1)
  {
    for ( size_t column = 0; column<11; column += 1)
    {
      gaussianWindow(row, column) = gaussianKernel.Evaluate(
      				     std::sqrt(std::pow(std::abs((int) row - 5), 2) +
      				     std::pow(std::abs((int) column - 5), 2)));
    }
  }
  gaussianWindow = gaussianWindow / arma::accu(gaussianWindow);
  
  // Calculate local mean of reference.
  arma::mat meanReference = arma::conv2(reference, gaussianWindow, "same");
  // Calculate local mean of image.
  arma::mat meanImage =  arma::conv2(image, gaussianWindow, "same");
  // Calculate local variance of reference.
  arma::mat varianceReference = arma::conv2(arma::square(reference),
      gaussianWindow, "same") - arma::square(meanReference);
  // Calculate local variance of image.
  arma::mat varianceImage = arma::conv2(arma::square(image),
      gaussianWindow, "same") - arma::square(meanImage);
  // Calulate covariance of image and reference.
  arma::mat covarianceReferenceImage = arma::conv2(reference % image,
      gaussianWindow, "same") - meanReference % meanImage;
  // Dropping the zero-padded edges.
  int nCol = meanReference.n_cols;
  int nRow = meanReference.n_rows;
  meanReference.shed_rows(nRow - 5, nRow - 1);
  meanReference.shed_rows(0, 4);
  meanReference.shed_cols(nCol - 5, nCol - 1);
  meanReference.shed_cols(0, 4);
  meanImage.shed_rows(nRow - 5, nRow - 1);
  meanImage.shed_rows(0, 4);
  meanImage.shed_cols(nCol - 5, nCol - 1);
  meanImage.shed_cols(0, 4);
  varianceReference.shed_rows(nRow - 5, nRow - 1);
  varianceReference.shed_rows(0, 4);
  varianceReference.shed_cols(nCol - 5, nCol - 1);
  varianceReference.shed_cols(0, 4);
  varianceImage.shed_rows(nRow - 5, nRow - 1);
  varianceImage.shed_rows(0, 4);
  varianceImage.shed_cols(nCol - 5, nCol - 1);
  varianceImage.shed_cols(0, 4);
  covarianceReferenceImage.shed_rows(nRow - 5, nRow - 1);
  covarianceReferenceImage.shed_rows(0, 4);
  covarianceReferenceImage.shed_cols(nCol - 5, nCol - 1);
  covarianceReferenceImage.shed_cols(0, 4);
  
  // Calculate Dynamic Range.
  const double dynamicRange = 255;

  // Calculate C1.
  const double regularisationConstant1 = 0.0001 * dynamicRange * dynamicRange;

  // Calculate C2.
  const double regularisationConstant2 = 0.0009 * dynamicRange * dynamicRange;

  // Calculate local SSIM index.
  arma::mat localSSIM = ((2 * meanReference % meanImage + regularisationConstant1) %
      (2 * covarianceReferenceImage + regularisationConstant2)) /
      ((arma::square(meanReference) + arma::square(meanImage) +
       regularisationConstant1) % (varianceReference + varianceImage +
      regularisationConstant2));

  // Calculate average of local SSIM index.
  const double meanSSIM = arma::mean(arma::mean(localSSIM));
  return meanSSIM;
}

} // namespace cv.
} // namespace mlpack.
#endif
