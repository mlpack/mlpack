/**
 * @file feature_extraction.hpp
 * @author Nilay Jain 
 *
 * Feature Extraction for the edge_boxes algorithm.
 */
#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_HPP
#include <mlpack/core.hpp>
#include "feature_parameters.hpp"
namespace mlpack {
namespace structured_tree {

template <typename MatType = arma::mat, typename CubeType = arma::cube>
class StructuredForests
{
 private:
  FeatureParameters params;

 public:

  /**
   * Constructor: stores all the parameters in an object
   * of feature_parameters class.
   * @param F FeatureParameters object which stores necessary parameters.
   */ 
  StructuredForests(FeatureParameters F);

  void PrepareData(const MatType& Images, const MatType& Boundaries,\
            const MatType& Segmentations);

  /**
   * Get Dimensions of Features
   * @param FtrDim Output vector that contains the result dimensions. 
   */
  void GetFeatureDimension(arma::vec& FtrDim);
  
  /**
   * Computes distance transform of 1D vector f.
   * @param f input vector whose distance transform is to be found.
   * @param n size of the Output vector to be made.
   * @param inf a large double value.
   * @param d Output vector which stores distance transform of f.
   */
  void DistanceTransform1D(const arma::vec& f, const size_t n,\
                           const double inf, arma::vec& d);
  
  /**
   * Computes distance transform of a 2D array
   * @param Im input array whose distance transform is to be found.
   * @param inf a large double value.
   */
  void DistanceTransform2D(MatType &Im, const double inf);

  /**
   * euclidean distance transform of binary Image using squared distance
   * @param Im Input binary Image whose distance transform is to be found.
   * @param on if on == 1, 1 is taken as boundaries and vice versa.
   * @param Out Output Image.
   * This is the discription of the paper which discribes the approach 
   * for this algorithm : Distance Transforms of Sampled Functions,
   * P. Felzenszwalb, D. Huttenlocher
   * Theory of Computing, Vol. 8, No. 19, September 2012
   */  
  void DistanceTransformImage(const MatType& Im, double on, MatType& Out);

  void GetFeatures(const MatType &Image, arma::umat &loc,\
                   CubeType& RegFtr, CubeType& SSFtr,\
                   const arma::vec& table);

  /**
   * Makes a reflective border around an Image.
   * @param InImage, Image which we have to make border around.
   * @param top, border length (to be incremented) at top.
   * @param left, border length at left.
   * @param bottom, border length at bottom.
   * @param right, border length at right.
   * @param OutImage, Output Image. 
   */  
  void CopyMakeBorder(const CubeType& InImage, size_t top, 
                 size_t left, size_t bottom, size_t right,
                 CubeType& OutImage);
  
  void GetShrunkChannels(const CubeType& InImage, CubeType &reg_ch,\
                  CubeType &ss_ch, const arma::vec& table);
  
  /**
   * Converts an Image in RGB color space to LUV color space.
   * RGB must range in (0.0, 1.0).
   * @param InImage Input Image in RGB color space.
   * @param OutImage Ouptut Image in LUV color space.
   */
  void RGB2LUV(const CubeType& InImage, CubeType& OutImage,\
                   const arma::vec& table);
  
  /**
   * Resizes the Image to the given size using Bilinear Interpolation
   * @param src Input Image
   * @param height Height of Output Image.
   * @param width Width Out Output Image.
   * @param dst Output Image resized to (height, width)
   */  
  void BilinearInterpolation(const MatType& src,
                            const size_t height, 
                            const size_t width,
                            MatType& dst);
  
  /**
   * Applies a separable linear filter to an Image
   * @param InOutImage Input/Output Contains the input Image, The final filtered Image is
   *          stored in this param.
   * @param kernel Input Kernel vector to be applied on Image.
   * @param radius amount, the Image should be padded before applying filter.
   */  
  void Convolution(CubeType &InOutImage, const MatType& Filter, const size_t radius);

  /**
   * Applies a triangle filter on an Image.
   * @param InImage Input/Output Image on which filter is applied.
   * @param radius Decides the size of kernel to be applied on Image.
   */
  void ConvTriangle(CubeType &InImage, const size_t radius);

  /**
   * finds maximum of numbers on cube axis and stores maximum values
   * in MaxVal, locations of maximum values in Location
   * @param mag Input Cube for which we want to find max values and location
   * @param Location Stores the slice number at which max value occurs
   * @param MaxVal Stores the maximum value among all slices for a given (row, col).
   */
  void MaxAndLoc(CubeType &mag, arma::umat &Location, MatType& MaxVal) const;

  /**
   * Computes Magnitude & Orientation of the Edges.
   * Gradient of a function is a vector of partial derivatives in each direction.
   * In this function the edges are calculated by applying the sobel filter on Image
   * which is the same as finding the vectors of partial derivates.
   * These "vectors" have a magnitude and a direction (orientation), which we 
   * calculate in this function.
   * @param InImage Input Image for which we calculate Magnitude & Orientation.
   * @param Magnitude Magnitude of the Edges
   * @param Orientation Orientation of the Edges
   */
  void Gradient(const CubeType& InImage, 
         MatType& Magnitude,
         MatType& Orientation);

  
  void Histogram(const MatType& Magnitude,
                 const MatType& Orientation, 
                 const size_t downscale,
                 const size_t interp,
                 CubeType& HistArr);
 
  void ViewAsWindows(const CubeType& channels, const arma::umat& loc,
                     CubeType& features);

  void GetRegFtr(const CubeType& channels, const arma::umat& loc,
                 CubeType& RegFtr);

  void GetSSFtr(const CubeType& channels, const arma::umat& loc,
                 CubeType SSFtr);

  void Rearrange(const CubeType& channels, CubeType& ch);

  void PDist(const CubeType& features, const arma::uvec& grid_pos,
             CubeType& Output);

  size_t IndexMin(arma::vec& k);

  size_t Discretize(const MatType& labels, const size_t nClass,\
           const size_t nSample, arma::vec& DiscreteLabels);
};


} //namespace structured_tree
} // namespace mlpack
#include "feature_extraction_impl.hpp"
#endif



