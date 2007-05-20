/*
 * =====================================================================================
 *
 *       Filename:  binary_tree_unit.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/27/2007 10:20:40 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include <unistd.h>
#include <sys/mman.h>
#include <limits>
#include "fastlib/fastlib.h"
#include "u/nvasil/loki/NullType.h"
#include "u/nvasil/mmanager/memory_manager.h"
#include "u/nvasil/mmanager_with_tpie/memory_manager.h"
#include "u/nvasil/test/test.h"
#include "u/nvasil/dataset/binary_dataset.h"
#include "tree_parameters_macro.h"
#include "euclidean_metric.h"
#include "null_statistics.h"
#include "hyper_rectangle.h"
#include "point_identity_discriminator.h"
#include "kd_pivoter1.h"
#include "binary_tree.h"

using namespace std;

template<typename TYPELIST, bool diagnostic>
class BinaryTreeTest {
 public:
  typedef typename TYPELIST::Precision_t   Precision_t;
	typedef typename TYPELIST::Allocator_t   Allocator_t;
  typedef typename TYPELIST::Metric_t      Metric_t;
	typedef typename TYPELIST::BoundingBox_t BoundingBox_t;
	typedef typename TYPELIST::NodeCachedStatistics_t NodeCachedStatistics_t;
	typedef typename TYPELIST::PointIdDiscriminator_t PointIdDiscriminator_t;
	typedef typename TYPELIST::Pivot_t Pivot_t;
	typedef Point<Precision_t, Allocator_t> Point_t; 
	typedef BinaryTree<TYPELIST, diagnostic> BinaryTree_t;
	typedef typename BinaryTree_t::Node_t Node_t;
  BinaryTreeTest() {
	}	
	void Init() {
    Allocator_t::allocator_ = new Allocator_t();
		Allocator_t::allocator_->Init();
		dimension_=2;
    num_of_points_=1000;
    data_file_="data";
		knns_=40;
    range_=0.2;
		result_file_="allnn";
    data_.Init(data_file_, num_of_points_, dimension_);
    for(index_t i=0; i<num_of_points_; i++) {
		  for(index_t j=0; j<dimension_; j++) {
			  data_.At(i,j)=Precision_t(rand())/RAND_MAX - 0.48;
			}
			data_.set_id(i,i);
		}		
    tree_.Init(&data_);
	}
	void Destruct() {
	  tree_.Destruct();
	  data_.Destruct();
	  unlink(data_file_.c_str());
	  unlink(data_file_.append(".ind").c_str());	
		unlink(result_file_.c_str());
		Allocator_t::allocator_->Destruct();
		delete Allocator_t::allocator_;
		unlink("temp_mem");
	}
	void BuildDepthFirst(){
    printf("Testing BuildDepthFirst...\n");
		tree_.BuildDepthFirst();
		//tree_.Print();
		printf("%s\n", tree_.Statistics().c_str());
	}
	void BuildBreadthFirst() {
		printf("Testing BuildBreadthFirst...\n");
	  tree_.BuildBreadthFirst();
		//tree_.Print();
		printf("%s\n", tree_.Statistics().c_str());
	}
	void kNearestNeighbor() {
		printf("Testing kNearestNeighbor...\n");
		tree_.BuildDepthFirst();
		//tree_.Print();
    vector<pair<Precision_t, Point_t> > nearest_tree;
		pair<Precision_t, index_t> nearest_naive[num_of_points_];
	  for(index_t i=0; i<num_of_points_; i++) {
		  nearest_tree.clear();
			tree_.NearestNeighbor(data_.get_point(i),
                            &nearest_tree,
                            knns_);
			Naive(i, nearest_naive);
			for(index_t j=0; j<knns_; j++) {
			  TEST_DOUBLE_APPROX(nearest_naive[j+1].first, 
			 	  	               nearest_tree[j].first,
					                 numeric_limits<Precision_t>::epsilon());
			  TEST_ASSERT(nearest_tree[j].second.get_id()==
				            nearest_naive[j+1].second)	;
			}
		}
	}
	void RangeNearestNeighbor() {
		printf("Testing RangeNearestNeighbor...\n");
		tree_.BuildBreadthFirst();
	  //tree_.Print();
		vector<pair<Precision_t, Point_t> > nearest_tree;
		pair<Precision_t, index_t> nearest_naive[num_of_points_];
	  for(index_t i=0; i<num_of_points_; i++) {
		  nearest_tree.clear();
			tree_.NearestNeighbor(data_.get_point(i),
                            &nearest_tree,
                            range_);
			std::sort(nearest_tree.begin(), nearest_tree.end(), 
					     typename Node_t::PairComparator());
			Naive(i, nearest_naive);
			for(index_t j=0; j<(index_t)nearest_tree.size(); j++) {
			  TEST_DOUBLE_APPROX(nearest_naive[j+1].first, 
			 	  	               nearest_tree[j].first,
					                 numeric_limits<Precision_t>::epsilon());
			  TEST_ASSERT(nearest_tree[j].second.get_id()==
				            nearest_naive[j+1].second)	;
			}
		}
	}
	void AllKNearestNeighbors() {
		printf("Testing AllKNearestNeighbors...\n");
		tree_.BuildDepthFirst();
	 //	tree_.Print();
	  tree_.InitAllKNearestNeighborOutput(result_file_, 
				                                 knns_);
		tree_.AllNearestNeighbors(tree_.parent_, knns_);
    tree_.CloseAllKNearestNeighborOutput(knns_);
    struct stat info;
		if (stat(result_file_.c_str(), &info)!=0) {
      FATAL("Error %s file %s\n",
				    strerror(errno), data_file_.c_str());
		}
		uint64 map_size = info.st_size;

    int fp=open(result_file_.c_str(), O_RDWR);
    typename Node_t::NNResult *res;
		res=(typename Node_t::NNResult *)mmap(NULL, 
				                                  map_size, 
											         				    PROT_READ | PROT_WRITE, 
															            MAP_SHARED, fp,
			                                    0);
		TEST_ASSERT(res!=MAP_FAILED);
		close(fp);
		std::sort(res, res+num_of_points_*knns_);
		pair<Precision_t, index_t> nearest_naive[num_of_points_];
    for(index_t i=0; i<num_of_points_; i++) {
		  Naive(i, nearest_naive);
		 	for(index_t j=0; j<knns_; j++) {
			 	TEST_DOUBLE_APPROX(nearest_naive[j+1].first, 
			 	  	               res[data_.get_id(i)*knns_+j].distance_,
					                 numeric_limits<Precision_t>::epsilon());
			  TEST_ASSERT(res[data_.get_id(i)*knns_+j].nearest_.get_id()==
				            nearest_naive[j+1].second);
			}			
		}		
		munmap(res, map_size);
	}
	
	void AllRangeNearestNeighbors() {
	  printf("Testing AllRangeNearestNeighbors...\n");
	 	tree_.BuildBreadthFirst();
		//tree_.Print();
	  tree_.InitAllRangeNearestNeighborOutput(result_file_);
		tree_.AllNearestNeighbors(tree_.parent_, range_);
    tree_.CloseAllRangeNearestNeighborOutput();
    struct stat info;
		if (stat(result_file_.c_str(), &info)!=0) {
      FATAL( "Error %s file %s\n",
				    strerror(errno), data_file_.c_str());
		}
		uint64 map_size = info.st_size;

    int fp=open(result_file_.c_str(), O_RDWR);
    typename Node_t::NNResult *res;
		res=(typename Node_t::NNResult *)mmap(NULL, 
				                                map_size, 
											          				PROT_READ | PROT_WRITE, 
															          MAP_SHARED, fp,
			                                  0);
		close(fp);
		TEST_ASSERT(res!=MAP_FAILED);
		std::sort(res, res+map_size/sizeof(typename Node_t::NNResult));
		pair<Precision_t, index_t> nearest_naive[num_of_points_];
		index_t i=0;
    while (i<num_of_points_) {
		  Naive(i, nearest_naive);
			index_t j=0;
			while(res[j].point_id_<(index_t)data_.get_id(i)) {
			  j++;
			}
			index_t k=1;
		 	while (nearest_naive[k].first<=range_) {
				TEST_DOUBLE_APPROX(nearest_naive[k].first, 
			 	  	               res[j].distance_,
					                 numeric_limits<Precision_t>::epsilon());
				j++;
				k++;
			}
      i++;			
		}		
		munmap(res, map_size);
	}

  void TestAll() {
	  Init();
		BuildDepthFirst();
		Destruct();
		Init();
		BuildBreadthFirst();
		Destruct();
		Init();
		kNearestNeighbor();
		Destruct();
		Init();
		RangeNearestNeighbor();
		Destruct();
		Init();
		AllKNearestNeighbors();
		Destruct();
		Init();
		AllRangeNearestNeighbors();
    Destruct();
	}	
 
 private:
  BinaryTree_t tree_; 
  BinaryDataset<Precision_t> data_;
  string data_file_;
	string result_file_;
	int32 dimension_;
	index_t num_of_points_;
	index_t knns_;
	Precision_t range_;
	
	void Naive(index_t query, 
			       pair<Precision_t, index_t> *result) {
		for(index_t i=0;  i<num_of_points_; i++) {
			Precision_t dist=Metric_t::Distance(data_.At(i), 
					                                data_.At(query), 
					                                dimension_);
		  result[i].first=dist;
		  result[i].second=data_.get_id(i);
		}
		std::sort(result, result+num_of_points_);
	}
	
	void Naive(Precision_t *query, 
			       pair<Precision_t, index_t> *result) {
		for(index_t i=0;  i<num_of_points_; i++) {
			Precision_t dist=Metric_t::Distance(data_.At(i), 
					                                query, 
					                                dimension_);
		  result[i].first=dist;
		  result[i].second=data_.get_id(i);
		}
		std::sort(result, result+num_of_points_);
	}
};

/*TREE_PARAMETERS(float32,
		            MemoryManager<false>,
		            EuclideanMetric,
	              HyperRectangle,
	              NullStatistics,
                SimpleDiscriminator,
		            KdPivoter1,
								false) 
*/								
struct BasicTypes1 {
  typedef float32 Precision_t;
	typedef mmapmm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
};
struct Parameters1 {
  typedef float32 Precision_t;
	typedef mmapmm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
	typedef HyperRectangle<BasicTypes1, false> BoundingBox_t;
	typedef NullStatistics<Loki::NullType> NodeCachedStatistics_t;
  typedef SimpleDiscriminator PointIdDiscriminator_t;
  typedef KdPivoter1<BasicTypes1, false> Pivot_t; 
};
struct BasicTypes2 {
  typedef float32 Precision_t;
	typedef tpiemm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
};
struct Parameters2 {
  typedef float32 Precision_t;
	typedef tpiemm::MemoryManager<false> Allocator_t;
	typedef EuclideanMetric<float32> Metric_t;
	typedef HyperRectangle<BasicTypes2, false> BoundingBox_t;
	typedef NullStatistics<Loki::NullType> NodeCachedStatistics_t;
  typedef SimpleDiscriminator PointIdDiscriminator_t;
  typedef KdPivoter1<BasicTypes2, false> Pivot_t; 
};


typedef BinaryTreeTest<Parameters1, false> BinaryTreeTest1_t;
typedef BinaryTreeTest<Parameters2, false> BinaryTreeTest2_t;

int main(int argc, char *argv[]) {
  BinaryTreeTest1_t test1;
  test1.TestAll();
  BinaryTreeTest2_t test2;
	test2.TestAll();

}
