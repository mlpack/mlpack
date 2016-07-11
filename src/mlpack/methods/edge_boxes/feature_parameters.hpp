/**
 * @file feature_parameters.hpp
 * @author Nilay Jain
 *
 * Implementation of feature parameter class.
 */

#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP

namespace mlpack {
namespace structured_tree {
 
//! This class holds all the fields for the FeatureExtraction class.
class FeatureParameters
{
 public:

  FeatureParameters(){} //default constructor

  //! getter and setter methods for all the fields in class.
  void NumImages(size_t value) { numImages = value; }
  size_t NumImages() const { return numImages; }

  void RowSize(size_t value) { rowSize = value; }
  size_t RowSize() const { return rowSize; }

  void ColSize(size_t value) { colSize = value; }
  size_t ColSize() const { return colSize; }

  void RGBD(size_t value) { rgbd = value; }
  size_t RGBD() const { return rgbd; }

  void Shrink(size_t value) { shrink = value; }
  size_t Shrink() const { return shrink; }

  void NumOrient(size_t value) { numOrient = value; }
  size_t NumOrient() const { return numOrient; }

  void GrdSmoothRad(size_t value) { grdSmoothRad = value; }
  size_t GrdSmoothRad() const { return grdSmoothRad; }

  void GrdNormRad(size_t value) { grdNormRad = value; }
  size_t GrdNormRad() const { return grdNormRad; }

  void RegSmoothRad(size_t value) { regSmoothRad = value; }
  size_t RegSmoothRad() const { return regSmoothRad; }

  void SSSmoothRad(size_t value) { ssSmoothRad = value; }
  size_t SSSmoothRad() const { return ssSmoothRad; }

  void PSize(size_t value) { pSize = value; }
  size_t PSize() const { return pSize; }

  void GSize(size_t value) { gSize = value; }
  size_t GSize() const { return gSize; }

  void NumCell(size_t value) { numCell = value; }
  size_t NumCell() const { return numCell; }

  void NumPos(size_t value) { numPos = value; }
  size_t NumPos() const { return numPos; }

  void NumNeg(size_t value) { numNeg = value; }
  size_t NumNeg() const { return numNeg; }

  void Fraction(double value) { fraction = value; }
  double Fraction() const { return fraction; }

  void NumTree(double value) { numTree = value; }
  double NumTree() const { return numTree; }
 
 private:
  //! number of images in the dataset.
  size_t numImages;

  //! row size of images.
  size_t rowSize;

  //! column size of images. 
  size_t colSize;
  
  //! 0 for RGB, 1 for RGB + depth.
  size_t rgbd;

  //! amount to shrink channels
  size_t shrink;

  //! number of orientations per gradient scale
  size_t numOrient;

  //! radius for image gradient smoothing
  size_t grdSmoothRad;

  //! radius for gradient normalization
  size_t grdNormRad;

  //! radius for regular channel smoothing
  size_t regSmoothRad;

  //! radius for similar channel smooothing
  size_t ssSmoothRad;

  //! fraction of features to use to train each tree
  double fraction;

  //! size of image patches
  size_t pSize;

  //! size of ground truth patches
  size_t gSize;

  //! number of positive patches per tree
  size_t numPos;

  //! number of negative patches per tree
  size_t numNeg;

  //! number of self similarity cells
  size_t numCell;

  //! number of trees in forest to train
  size_t numTree;
};
 
}
}
#include "feature_extraction.hpp"
#endif
