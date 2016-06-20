/**
 * @file decision_stump.hpp
 * @author 
 *
 * Definition of decision stumps.
 */
#include <mlpack/core.hpp>
#include "feature_extraction.hpp"

using namespace mlpack;
using namespace mlpack::structured_tree;
using namespace std;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);
  /*
  :param options:
  num_images: number of images in the dataset.
  rgbd: 0 for RGB, 1 for RGB + depth
  shrink: amount to shrink channels
  n_orient: number of orientations per gradient scale
  grd_smooth_rad: radius for image gradient smoothing
  grd_norm_rad: radius for gradient normalization
  reg_smooth_rad: radius for reg channel smoothing
  ss_smooth_rad: radius for sim channel smoothing
  p_size: size of image patches
  g_size: size of ground truth patches
  n_cell: number of self similarity cells

  n_pos: number of positive patches per tree
  n_neg: number of negative patches per tree
  fraction: fraction of features to use to train each tree
  n_tree: number of trees in forest to train
  n_class: number of classes (clusters) for binary splits
  min_count: minimum number of data points to allow split
  min_child: minimum number of data points allowed at child nodes
  max_depth: maximum depth of tree
  split: options include 'gini', 'entropy' and 'twoing'
  discretize: optional function mapping structured to class labels

  stride: stride at which to compute edges
  sharpen: sharpening amount (can only decrease after training)
  n_tree_eval: number of trees to evaluate per location
  nms: if true apply non-maximum suppression to edges
  */

  FeatureParameters params = FeatureParameters();

  params.NumImages(2);
  params.RowSize(321);
  params.ColSize(481);
  params.RGBD(0);
  params.Shrink(2);
  params.NumOrient(4);
  params.GrdSmoothRad(0);
  params.GrdNormRad(4);
  params.RegSmoothRad(2);
  params.SSSmoothRad(8);
  params.Fraction(0.25);
  params.PSize(32);
  params.GSize(16);
  params.NumCell(5);
  params.NumPos(10000);
  params.NumNeg(10000);
  params.NumCell(5);
  params.NumTree(8);
  StructuredForests <arma::mat, arma::cube> SF(params);
//  arma::uvec x(2);
  //SF.GetFeatureDimension(x);
  
  arma::mat segmentations, boundaries, images;
  data::Load("/home/nilay/example/small_images.csv", images);
  data::Load("/home/nilay/example/small_boundary_1.csv", boundaries);
  data::Load("/home/nilay/example/small_segmentation_1.csv", segmentations);

  SF.PrepareData(images, boundaries, segmentations);
  cout << "PrepareData done." << endl;
  return 0;
}



