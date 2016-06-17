/**
 * @file feature_extraction.hpp
 * @author Nilay Jain 
 *
 * Feature Extraction for the edge_boxes algorithm.
 */
#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_HPP
//#define INF 999999.9999
//#define EPS 1E-20
#include <mlpack/core.hpp>

namespace mlpack {
namespace structured_tree {

template <typename MatType = arma::mat, typename CubeType = arma::cube>
class StructuredForests
{
 private:
  FeatureParameters params;

 public:

  static constexpr double eps = 1e-20;

  std::map<std::string, size_t> options;
  
  void StructuredForests(FeatureParameters F);   
/*  MatType LoadData(MatType const &images, MatType const &boundaries,\
     MatType const &segmentations);*/

  void PrepareData(MatType const &InputData);

  void GetFeatureDimension(arma::vec &FtrDim);
  
  void DistanceTransform1D(const arma::vec& f, const size_t n,\
                           const double inf, arma::vec& d);
  
  void DistanceTransform2D(MatType &Im, const double inf);
  
  void DistanceTransformImage(const MatType& Im, double on, MatType& Out);

  void GetFeatures(const MatType &Image, arma::umat &loc, 
            CubeType& RegFtr, CubeType& SSFtr);
  
  CubeType CopyMakeBorder(CubeType const &InImage, size_t top, 
               size_t left, size_t bottom, size_t right);
  
  void GetShrunkChannels(CubeType const &InImage, CubeType &reg_ch, CubeType &ss_ch);
  
  CubeType RGB2LUV(CubeType const &InImage);
  
  MatType bilinearInterpolation(MatType const &src,
                      size_t height, size_t width);
  
  CubeType sepFilter2D(CubeType &InOutImage, arma::vec &kernel,\
                       size_t radius);

  CubeType ConvTriangle(CubeType &InImage, size_t radius);

  void Gradient(CubeType const &InImage, 
         MatType &Magnitude,
         MatType &Orientation);

  MatType MaxAndLoc(CubeType &mag, arma::umat &Location) const;

  CubeType Histogram(MatType const &Magnitude,
              MatType const &Orientation, 
              size_t downscale, size_t interp);
 
  CubeType ViewAsWindows(CubeType const &channels, arma::umat const &loc);

  CubeType GetRegFtr(CubeType const &channels, arma::umat const &loc);

  CubeType GetSSFtr(CubeType const &channels, arma::umat const &loc);

  CubeType Rearrange(CubeType const &channels);

  CubeType PDist(CubeType const &features, arma::uvec const &grid_pos);

  //void Discretize(MatType const &lbl, size_t n_class, size_t n_sample);
};


} //namespace structured_tree
} // namespace mlpack
#include "feature_extraction_impl.hpp"
#endif

