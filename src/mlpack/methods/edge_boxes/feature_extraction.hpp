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
#include "feature_parameters.hpp"
namespace mlpack {
namespace structured_tree {

template <typename MatType = arma::mat, typename CubeType = arma::cube>
class StructuredForests
{
 private:
  FeatureParameters params;
  static constexpr double eps = 1e-20;

 public:

 
  StructuredForests(FeatureParameters F);   
/*  MatType LoadData(MatType const &images, MatType const &boundaries,\
     MatType const &segmentations);*/

  void PrepareData(const MatType& Images, const MatType& Boundaries,\
            const MatType& Segmentations);

  void GetFeatureDimension(arma::vec& FtrDim);
  
  void DistanceTransform1D(const arma::vec& f, const size_t n,\
                           const double inf, arma::vec& d);
  
  void DistanceTransform2D(MatType &Im, const double inf);
  
  void DistanceTransformImage(const MatType& Im, double on, MatType& Out);

  void GetFeatures(const MatType &Image, arma::umat &loc,\
                   CubeType& RegFtr, CubeType& SSFtr,\
                   const arma::vec& table);
  
  void CopyMakeBorder(const CubeType& InImage, size_t top, 
                 size_t left, size_t bottom, size_t right,
                 CubeType& OutImage);
  
  void GetShrunkChannels(const CubeType& InImage, CubeType &reg_ch,\
                  CubeType &ss_ch, const arma::vec& table);
  
  void RGB2LUV(const CubeType& InImage, CubeType& OutImage,\
                   const arma::vec& table);
  
  void BilinearInterpolation(const MatType& src,
                          size_t height, size_t width,
                          MatType& dst);
  
  void SepFilter2D(CubeType &InOutImage, const arma::vec& kernel, const size_t radius);

  void ConvTriangle(CubeType &InImage, const size_t radius);

  void Gradient(const CubeType& InImage, 
         MatType& Magnitude,
         MatType& Orientation);

  void MaxAndLoc(CubeType &mag, arma::umat &Location, MatType& MaxVal) const;

  void Histogram(const MatType& Magnitude,
            const MatType& Orientation, 
            size_t downscale, size_t interp,
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



