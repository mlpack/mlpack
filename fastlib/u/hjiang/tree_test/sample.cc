#include <limits>
#include "fastlib/fastlib.h"
#include "u/nvasil/mmanager/memory_manager.h"
#include "u/nvasil/dataset/binary_dataset.h"
#include "u/nvasil/tree/euclidean_metric.h"
#include "u/nvasil/tree/null_statistics.h"
#include "u/nvasil/tree/hyper_rectangle.h"
#include "u/nvasil/tree/point_identity_discriminator.h"
#include "u/nvasil/tree/kd_pivoter1.h"
#include "u/nvasil/tree/binary_tree.h"
struct BasicTypes {
  typedef float32 Precision_t;
	typedef mmapmm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
};
struct Parameters {
  typedef float32 Precision_t;
	typedef mmapmm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
	typedef HyperRectangle<BasicTypes, false> BoundingBox_t;
	typedef NullStatistics NodeCachedStatistics_t;
  typedef SimpleDiscriminator PointIdDiscriminator_t;
  typedef KdPivoter1<BasicTypes, false> Pivot_t; 
};

typedef BinaryTree<Parameters, false> BinaryTree_t;
typedef Point<BasicTypes::Precision_t, BasicTypes::Allocator_t> Point_t;
int main(int argc, char *argv[]) {
  BasicTypes::Allocator_t::allocator_=new BasicTypes::Allocator_t();
	BasicTypes::Allocator_t::allocator_->Init();
  BinaryTree_t tree;
	int32  dimension=2;
  index_t num_of_points=1000;
  string data_file="data";
	index_t knns=40;
  string result_file="allnn";
  BinaryDataset<float32> data;
	data.Init(data_file, num_of_points, dimension);
  for(index_t i=0; i<num_of_points; i++) {
	  for(index_t j=0; j<dimension; j++) {
			 data.At(i,j)=float32(rand())/RAND_MAX - 0.48;
		}
		data.set_id(i,i);
	}		
  tree.Init(&data);
  tree.set_max_points_on_leaf(30);	
  printf("Testing kNearestNeighbor...\n");
	tree.BuildDepthFirst();
	tree.Print();
  vector<pair<BasicTypes::Precision_t, Point_t> > nearest_tree;
	for(index_t i=0; i<num_of_points; i++) {
		nearest_tree.clear();
		tree.NearestNeighbor(data.get_point(i),
                          &nearest_tree,
                          knns);
	}

	
  tree.Destruct();
	data.Destruct();
	delete BasicTypes::Allocator_t::allocator_;

}
