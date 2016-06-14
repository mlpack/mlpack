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

int main()
{
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

  map<string, size_t> options;
  options["num_images"] = 2;
  options["row_size"] = 321;
  options["col_size"] = 481;
  options["rgbd"] = 0;
  options["shrink"] = 2;
  options["n_orient"] = 4;
  options["grd_smooth_rad"] = 0;
  options["grd_norm_rad"] = 4;
  options["reg_smooth_rad"] = 2;
  options["ss_smooth_rad"] = 8;
  options["p_size"] = 32;
  options["g_size"] = 16;
  options["n_cell"] = 5;

  options["n_pos"] = 10000;
  options["n_neg"] = 10000;
  //options["fraction"] = 0.25;
  options["n_tree"] = 8;
  options["n_class"] = 2;
  options["min_count"] = 1;
  options["min_child"] = 8;
  options["max_depth"] = 64;
  options["split"] = 0;  // we use 0 for gini, 1 for entropy, 2 for other
  options["stride"] = 2;
  options["sharpen"] = 2;
  options["n_tree_eval"] = 4;
  options["nms"] = 1;    // 1 for true, 0 for false

  StructuredForests <arma::mat, arma::cube> SF(options);
//  arma::uvec x(2);
  //SF.GetFeatureDimension(x);
  
  arma::mat segmentations, boundaries, images;
  data::Load("/home/nilay/Desktop/GSoC/code/example/example/small_images.csv", images);
  data::Load("/home/nilay/Desktop/GSoC/code/example/example/small_boundary_1.csv", boundaries);
  data::Load("/home/nilay/Desktop/GSoC/code/example/example/small_segmentation_1.csv", segmentations);

  arma::mat input_data = SF.LoadData(images, boundaries, segmentations);
  cout << input_data.n_rows << " " << input_data.n_cols << endl;
  SF.PrepareData(input_data);
  cout << "PrepareData done." << endl;
  return 0;
}

