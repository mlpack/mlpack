/*
 * =====================================================================================
 *
 *       Filename:  node_unit.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/23/2007 10:19:28 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "u/nvasil/loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "u/nvasil/mmanager/memory_manager.h"
#include "u/nvasil/dataset/binary_dataset.h"
#include "hyper_rectangle.h"
#include "euclidean_metric.h"
#include "null_statistics.h"
#include "point_identity_discriminator.h"
#include "computations_counter.h"
#include "node.h"

template<typename TYPELIST, bool diagnostic>
class NodeTest {
 public:
  typedef typename TYPELIST::Precision_t Precision_t;
  typedef typename TYPELIST::Allocator_t Allocator_t;
  typedef typename TYPELIST::Metric_t    Metric_t;
	typedef HyperRectangle<TYPELIST, diagnostic> HyperRectangle_t;
  struct  NodeParameters : public TYPELIST {
	 	typedef HyperRectangle_t    BoundingBox_t;
	  typedef NullStatistics      NodeCachedStatistics_t;
	  typedef SimpleDiscriminator PointIdDiscriminator_t;
	};
	typedef Node<NodeParameters, diagnostic> Node_t;
	typedef typename Allocator_t:: template ArrayPtr<Precision_t> Array_t;
	typedef Point<Precision_t, Loki::NullType> Point_t;
  NodeTest() {
	}
	~NodeTest() {
	}
	void Init() {
	  dimension_=2;
	  num_of_points_=30;
	  Allocator_t::allocator_ = new Allocator_t();
		Allocator_t::allocator_->Initialize();
	  Array_t min(dimension_);
		min[0]=-1;
		min[1]=-1;
	  Array_t max(dimension_);
		max[0]=1;
		max[1]=1;
		data_file_="data";
    dataset_.Init(data_file_, num_of_points_, dimension_);
    for(index_t i=0; i<num_of_points_; i++) {
		  dataset_.At(i)[0]=Precision_t(rand())/RAND_MAX;
			dataset_.At(i)[1]=Precision_t(-rand())/RAND_MAX;
			dataset_.set_id(i,i);
		}
		hyper_rectangle_.Init(min, max, 0, 0);
	  NullStatistics statistics;
		//	typename Node_t::NodeCachedStatistics_t statistics;
	  node_.Reset(new Node_t);
		node_->Init(hyper_rectangle_,
                statistics,
								0,
								0,
			          num_of_points_, 
			          dimension_,
							  &dataset_); 
  }
	void Destruct() {
		hyper_rectangle_.Destruct();
		delete Allocator_t::allocator_;
		dataset_.Destruct();
		unlink(data_file_.c_str());
		unlink(data_file_.append(".ind").c_str());
	}

	void FindNearest() { 
	  printf("Testing find nearest\n");
		SimpleDiscriminator discriminator;
    vector<pair<Precision_t, Point<Precision_t, Allocator_t> > > nearest;
    ComputationsCounter<diagnostic> comp;
		for(index_t i=0; i<num_of_points_; i++) {
			Point_t query_point;
      		  query_point.Alias(dataset_.At(i), dataset_.get_id(i));
			nearest.clear();
			node_->FindNearest(query_point, 
			                   nearest, 
							  				 1, 
												 dimension_,
                         discriminator,
									       comp);
			Precision_t min_dist=numeric_limits<Precision_t>::max();
			index_t min_id=0;
      for(index_t j=0; j<num_of_points_; j++) {
			  if (unlikely(dataset_.get_id(j)==dataset_.get_id(i))) {
				  continue;
				}
				Precision_t distance = HyperRectangle_t::Distance(dataset_.At(i),
						                                              dataset_.At(j),
																													dimension_);
				if (distance<min_dist) {
				  min_id=j;
					min_dist=distance;
				}
			}
			DEBUG_ASSERT_MSG(min_dist==nearest[0].first, 
					"Something wrong in the distance\n");
			DEBUG_ASSERT_MSG(min_id==nearest[0].second.get_id(), 
					             "Something wrong in the distance\n");
	  }	
	}
  void FindAllNearest() {
    printf("Testing find all nearest\n");
		typename Node_t::NNResult result[num_of_points_];
	  node_->set_kneighbors(result, 1);	  
		Precision_t max_neighbor_distance=numeric_limits<Precision_t>::max();
    SimpleDiscriminator discriminator;
    ComputationsCounter<diagnostic> comp;
	  node_->FindAllNearest(node_,
                          max_neighbor_distance,
                          1,
                          dimension_,
                          discriminator,
                          comp);
		for(index_t i=0; i<num_of_points_; i++) {
		  Precision_t min_dist=numeric_limits<Precision_t>::max();
			index_t min_id=0;
      for(index_t j=0; j<num_of_points_; j++) {
		    if (unlikely(dataset_.get_id(j)==dataset_.get_id(i))) {
			    continue;
	 	    }
		    Precision_t distance = HyperRectangle_t::Distance(dataset_.At(i),
		 	 	                                                  dataset_.At(j),
					  																						  dimension_);
		    if (distance<min_dist) {
		      min_id=j;
			    min_dist=distance;
		    }
	    }
			DEBUG_ASSERT_MSG(min_dist==result[i].distance_, 
					"Something wrong in the distance\n");
			DEBUG_ASSERT_MSG(min_id==result[i].nearest_.get_id(), 
					             "Something wrong in the distance\n");
		}
	}
	
  void TestAll(){
    Init();
    FindNearest();
    Destruct();
    Init();
    FindAllNearest();
    Destruct();	 
	}
	
 private: 
	typename Allocator_t:: template Ptr<Node_t> node_;
  string data_file_;
	HyperRectangle_t  hyper_rectangle_;
  BinaryDataset<Precision_t> dataset_;
	index_t num_of_points_;
  int32 dimension_; 
  	
};

struct BasicParameters{
	typedef float32                  Precision_t; 
	typedef MemoryManager<false>     Allocator_t; 
	typedef EuclideanMetric<float32> Metric_t;
};

int main(int argc, char *argv[]) {
	NodeTest<BasicParameters, false> node_test;
	node_test.TestAll();
}
