/**
 * @file feature_extraction_impl.hpp
 * @author Nilay Jain
 *
 * Implementation of feature parameter class.
 */

#ifndef MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP
#define MLPACK_METHODS_EDGE_BOXES_STRUCTURED_TREE_IMPL_HPP

namespace mlpack {
namespace structured_tree {
 
//This class holds all the fields for the FeatureExtraction class.
class FeatureParameters
{
 public:

  FeatureParameters(); //default constructor

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
 
 private:
  size_t numImages;
  size_t rowSize;
  size_t colSize;
  size_t rgbd;
  size_t shrink;
  size_t numOrient;
  size_t grdSmoothRad;
  size_t grdNormRad;
  size_t regSmoothRad;
  size_t ssSmoothRad;
  size_t pSize;
  size_t gSize;
  size_t numCell;
  size_t numPos;
  size_t numNeg;
  double numCell;
};
 
}
}
