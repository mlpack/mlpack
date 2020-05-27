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
      <<reference.n_cols<<"X"<<reference.n_rows
      <<") does not match size of image ("
      <<image.n_cols<<"X"<<image.n_rows<<")!"
      <<std::endl;
    throw std::invalid_argument(oss.str());
  }

  kernel::GaussianKernel gaussianKernel = kernel::GaussianKernel(1.5);
  arma::mat gaussianWindow(11, 11);
  for ( double row = 0; row < 11; row+=1)
  {
    for ( double column = 0; column<11; column+=1)
    {
      gaussianWindow(row, column) = gaussianKernel.Evaluate(arma::linspace(-6, row - 5, 1),
                                                            arma::linspace(-6, column - 5, 1));
    }
  }
  arma::mat meanReference = arma::conv2(reference, gaussianWindow, "same"); 

  arma::mat meanImage =  arma::conv2(image, gaussianWindow, "same");

  arma::mat varianceReference = arma::conv2(arma::square(reference),
                                            gaussianWindow, "same") -
                                            arma::square(meanReference);

  arma::mat varianceImage = arma::conv2(arma::square(image),
                                        gaussianWindow, "same") -
                                        arma::square(meanImage);

  arma::mat covarianceReferenceImage = arma::conv2(reference%image,
                                                   gaussianWindow, "same") -
                                                   meanReference%meanImage;

  double dynamicRange = image.max() - image.min();

  double regularisationConstant1 = 0.0001 * dynamicRange * dynamicRange;

  double regularisationConstant2 = 0.0009 * dynamicRange * dynamicRange;

  arma::mat localSSIM = ((2*meanReference%meanImage + regularisationConstant1)%
                        (2*covarianceReferenceImage + regularisationConstant2))/
                        ((arma::square(meanReference)+arma::square(meanImage)+
                         regularisationConstant1)%
                        (varianceReference + varianceImage +
                         regularisationConstant2));

  double meanSSIM = arma::mean(arma::mean(localSSIM));
  return meanSSIM;

}

}// namespace cv
}// namespace mlpack
#endif
