/**
 * @file feature_extraction.hpp
 * @author Nilay Jain 
 *
 * Feature Extraction for the edge_boxes algorithm.
 */
#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_HPP
#define INF 999999.9999
#define EPS 1E-20
#include <mlpack/core.hpp>

namespace mlpack {
namespace structured_tree {

template <typename MatType = arma::mat, typename CubeType = arma::cube>
class StructuredForests
{

 public:

  std::map<std::string, int> options;
  
  StructuredForests(const std::map<std::string, int>& inMap);
  
  MatType LoadData(MatType& images, MatType& boundaries,
                     MatType& segmentations);

  void PrepareData(MatType& InputData);

 private:

  arma::vec GetFeatureDimension();
  
  arma::vec dt_1d(arma::vec& f, int n);
  
  void dt_2d(MatType& im);
  
  MatType dt_image(MatType& im, double on);
  
  arma::field<CubeType> GetFeatures(MatType& img,arma::umat& loc);
  
  CubeType CopyMakeBorder(CubeType& InImage,
              int top, int left, int bottom, int right);
  
  void GetShrunkChannels(CubeType& InImage, CubeType& reg_ch, CubeType& ss_ch);
  
  CubeType RGB2LUV(CubeType& InImage);
  
  MatType bilinearInterpolation(MatType const &src,
                      size_t height, size_t width);
  
  CubeType sepFilter2D(CubeType& InImage, 
                        arma::vec& kernel, int radius);

  CubeType ConvTriangle(CubeType& InImage, int radius);

  void Gradient(CubeType& InImage, 
                MatType& Magnitude,
                MatType& Orientation);

  MatType MaxAndLoc(CubeType& mag, arma::umat& Location);

  CubeType Histogram(MatType& Magnitude,
                       MatType& Orientation, 
                       int downscale, int interp);
 
  CubeType ViewAsWindows(CubeType& channels, arma::umat& loc);

  CubeType GetRegFtr(CubeType& channels, arma::umat& loc);

  CubeType GetSSFtr(CubeType& channels, arma::umat& loc);

  CubeType Rearrange(CubeType& channels);

  CubeType PDist(CubeType& features, arma::uvec& grid_pos);

};


} //namespace structured_tree
} // namespace mlpack
#include "feature_extraction_impl.hpp"
#endif

