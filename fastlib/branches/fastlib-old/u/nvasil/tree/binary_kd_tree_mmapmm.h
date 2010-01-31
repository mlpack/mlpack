/*
 * =====================================================================================
 * 
 *       Filename:  tree_definiions.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  05/18/2007 07:08:44 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#include "fastlib/fastlib.h"
#include "u/nvasil/mmanager/memory_manager.h"
#include "u/nvasil/tree/tree_parameters_macro.h"
#include "u/nvasil/tree/euclidean_metric.h"
#include "u/nvasil/tree/null_statistics.h"
#include "u/nvasil/tree/hyper_rectangle.h"
#include "u/nvasil/tree/point_identity_discriminator.h"
#include "u/nvasil/tree/kd_pivoter1.h"
#include "u/nvasil/tree/knn_node.h"
#include "u/nvasil/tree/binary_tree.h"

struct BasicTypes2 {
  typedef float32 Precision_t;
	typedef mmapmm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
};
struct NodeParameters2 {
  typedef float32 Precision_t;
	typedef mmapmm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
	typedef HyperRectangle<BasicTypes2, false> BoundingBox_t;
	typedef NullStatistics<Loki::NullType> NodeCachedStatistics_t;
  typedef SimpleDiscriminator PointIdDiscriminator_t;
  typedef KdPivoter1<BasicTypes2, false> Pivot_t; 
};
struct TreeParameters4 {
  typedef KnnNode<NodeParameters2, false> Node_t;
  typedef KdPivoter1<BasicTypes2, false> Pivot_t; 
};

typedef BinaryTree<TreeParameters4, false> BinaryKdTreeMMAPMMKnnNode_t;


