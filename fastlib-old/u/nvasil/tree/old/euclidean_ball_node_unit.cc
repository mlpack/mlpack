/*
 * =====================================================================================
 *
 *       Filename:  euclidean_ball_node_unit.cc
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  02/12/2007 04:11:43 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */


#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <limits>
#include "boost/scoped_ptr.hpp"
#include "base/basic_types.h"
#include "smart_memory/src/memory_manager.h" 
#include "euclidean_ball_node.h"
#include "data_file.h"
#include "data_reader.h"
#include "pivot_policy.h"

class EuclideanBallPivotTest : public CppUnit::TestFixture  {
  CPPUNIT_TEST_SUITE(EuclideanBallPivotTest);
  CPPUNIT_TEST(UpdateHyperBallTest);
	CPPUNIT_TEST(FindPivotPointsTest);
	//CPPUNIT_TEST(PivotParentTest);
  CPPUNIT_TEST(PivotTest);
  CPPUNIT_TEST_SUITE_END();
 public:
  typedef  EuclideanBallNode<float32, uint64, 
					        MemoryManager<false>, false> NodeHyperBall_t;
	typedef NodeHyperBall_t::Pivot_t  Pivot_t;
  typedef PivotPolicy<float32, uint64, MemoryManager<false>, 
					            NodeHyperBall_t, false> PivotPolicy_t;
	const char *kInputFile_;
 void setUp() {
	  kInputFile_ = "data1_float32_uint64";
		MemoryManager<false>::allocator = new MemoryManager<false>();
		MemoryManager<false>::allocator->Initialize();
		dimension_=2;
		box_pivot_.dimension_ = dimension_;
		box_pivot_.radious_ = 0.5;
		box_pivot_.center_.Reset(MemoryManager<false>::malloc<float32>(dimension_));
		box_pivot_.left_.Reset(MemoryManager<false>::malloc<float32>(dimension_));
 		box_pivot_.right_.Reset(MemoryManager<false>::malloc<float32>(dimension_));
		
		for(int32 i=0;  i< dimension_; i++) {
      box_pivot_.center_[i]=0;
		}
		void *temp;
  	OpenDataFile(string(kInputFile_), &dimension_, 
                 &num_of_points_, 
                 &temp,
                 &map_size_);
    data_ = new DataReader<float32, uint64>(temp, dimension_);

	}
	void tearDown() {
    if(remove("temp_mem")<0) {
	    fprintf(stderr,"Unable to delete file, error %s\n", strerror(errno));
	  }
		delete MemoryManager<false>::allocator;
		CloseDataFile(data_->get_source() , map_size_);
		delete data_;
	}
 protected:
  void UpdateHyperBallTest() {
	  float32 *point1 = new float32[dimension_];
    float32 *point2 = new float32[dimension_];
  
	  for(int32 i=0; i<dimension_; i++) {
	    point1[i] =  1;
		  point2[i] = -1;
	  }
		PivotPolicy_t::UpdateHyperBall(point1, box_pivot_);
		PivotPolicy_t::UpdateHyperBall(point2, box_pivot_);
    PivotPolicy_t::NormalizeHyperBall(box_pivot_, 2);
    for(int32 i=0; i<dimension_; i++) {
			CPPUNIT_ASSERT(box_pivot_.center_[i] == 0);
	  } 
	}
  void FindPivotPointsTest() {
    float32 *point1 = new float32[dimension_];
    float32 *point2 = new float32[dimension_];
  
	  for(int32 i=0; i<dimension_; i++) {
	    point1[i] =  i;
		  point2[i] = -i;
	  }
		PivotPolicy_t::UpdateHyperBall(point1, box_pivot_);
		PivotPolicy_t::UpdateHyperBall(point2, box_pivot_);
		PivotPolicy_t::NormalizeHyperBall(box_pivot_, 2);
    for(int32 i=0; i<dimension_; i++) {
			CPPUNIT_ASSERT(box_pivot_.center_[i] == 0);
	  }
		for(uint64 i=0; i<num_of_points_; i++) {
		 	PivotPolicy_t::UpdateHyperBall(data_->At(i), box_pivot_);
		}
		PivotPolicy_t::NormalizeHyperBall(box_pivot_, num_of_points_);

		PivotPolicy_t::FindPivotPoints(data_, box_pivot_, 0, num_of_points_);
		float32 dist1 = PivotPolicy_t::Metric_t::Distance(box_pivot_.left_, 
				                                              box_pivot_.center_,
																											dimension_);
    float32 dist2 = PivotPolicy_t::Metric_t::Distance(box_pivot_.right_, 
			                                                box_pivot_.center_,
																											dimension_);
         
    float32 dist3 = PivotPolicy_t::Metric_t::Distance(box_pivot_.right_, 
			                                                box_pivot_.left_,
																											dimension_);
     
		CPPUNIT_ASSERT(dist3>dist1);
		CPPUNIT_ASSERT(dist3>dist2);
	}
 void PivotTest() {
   node_pivot_.box_pivot_data_ = box_pivot_;
	 node_pivot_.start_ = 0;
	 node_pivot_.num_of_points_ = num_of_points_;
	 pair<Pivot_t*, Pivot_t*> pivot_pair;
   for(uint64 i=0; i<num_of_points_; i++) {
		 PivotPolicy_t::UpdateHyperBall(data_->At(i), box_pivot_);
	 }
	 PivotPolicy_t::NormalizeHyperBall(box_pivot_, num_of_points_);

	 PivotPolicy_t::FindPivotPoints(data_, box_pivot_, 0, num_of_points_);

	 pivot_pair = PivotPolicy_t::Pivot(data_, &node_pivot_);
	 Pivot_t *temp_pv = pivot_pair.first;
	 for(uint64 i=0; i<temp_pv->num_of_points_; i++) {
		 float32 dist_left = 
			 PivotPolicy_t::Metric_t::Distance(node_pivot_.box_pivot_data_.left_,
					                               data_->At(temp_pv->start_+i),
																				 dimension_);
		 float32 dist_right=
			 PivotPolicy_t::Metric_t::Distance(node_pivot_.box_pivot_data_.right_,
					                               data_->At(temp_pv->start_+i),
																				 dimension_);

	   CPPUNIT_ASSERT(dist_left<=dist_right);
	 }
   temp_pv = pivot_pair.second;
   for(uint64 i=0; i<temp_pv->num_of_points_; i++) {
		 float32 dist_left = 
			 PivotPolicy_t::Metric_t::Distance(node_pivot_.box_pivot_data_.left_,
					                               data_->At(temp_pv->start_+i),
																				 dimension_);
		 float32 dist_right=
			 PivotPolicy_t::Metric_t::Distance(node_pivot_.box_pivot_data_.right_,
					                               data_->At(temp_pv->start_+i),
																				 dimension_);

	   CPPUNIT_ASSERT(dist_left>dist_right);
	 }
   
 } 
 private:
	NodeHyperBall_t *node_;
  NodeHyperBall_t::BoundingBox_t::PivotData box_pivot_;
  Pivot_t node_pivot_;	
	int32 dimension_;
	uint64 num_of_points_;
	DataReader<float32, uint64> *data_;
	uint64 map_size_;
};

class EuclideanBallNNTest : public CppUnit::TestFixture  {
  CPPUNIT_TEST_SUITE(EuclideanBallNNTest);
  CPPUNIT_TEST(TestFindNearest);
	CPPUNIT_TEST(TestFindAllNearest);
  CPPUNIT_TEST_SUITE_END();
 public:
  typedef  EuclideanBallNode<float32, uint32, 
					        MemoryManager<false>, false> NodeHyperBall_t;
	typedef NodeHyperBall_t::Pivot_t  Pivot_t;
  static const int32 kDimension_ = 2;
  static const int32 kDataPoints_ = 5;
  static const int32 kTestPoints_ = 2;
  static const float32 kMinHrValues_[kDimension_];
  static const float32 kMaxHrValues_[kDimension_];
  // we add another extra float per point as a point id
  // we will just set it to zero. we only need this
  // to fool the DataReader class
  static  float32 kData_[kDataPoints_][kDimension_+1];
  static const float32 kRange_ = 0.102;
  static const float32 kNearestDistances_[kDataPoints_][2];
  static const int32   kNearestPoints_[kDataPoints_][2];
 
	static const char *kInputFile_;
  void setUp() {
		// This is sort of a hack but we can't really have float32 idprecision
		// but we can't have different types stored in an array
		// So we are doing this hack
		for(uint32 i=0; i<(uint32)kDataPoints_; i++) {
		  memcpy(&kData_[i][kDimension_], &i, sizeof(uint32));
		};
		MemoryManager<false>::allocator = new MemoryManager<false>();
		MemoryManager<false>::allocator->Initialize();
    data_ = new DataReader<float32, uint32>((void *)kData_, kDimension_);
		NodeHyperBall_t::BoundingBox_t::PivotData box_pivot;
    Pivot_t pivot;
		pivot.box_pivot_data_.dimension_ = kDimension_;
		pivot.box_pivot_data_.radious_ = 0;
		pivot.box_pivot_data_.center_.Reset(MemoryManager<false>::malloc<float32>(kDimension_));
   	pivot.box_pivot_data_.left_.Reset(MemoryManager<false>::malloc<float32>(kDimension_));
   	pivot.box_pivot_data_.right_.Reset(MemoryManager<false>::malloc<float32>(kDimension_));

		pivot.start_ = 0;
		pivot.num_of_points_ = 5;
		node_.Reset(new NodeHyperBall_t(&pivot, 0, data_));
		CPPUNIT_ASSERT(node_->IsLeaf()==true);
	}
  void tearDown() {
    if(remove("temp_mem")<0) {
	    fprintf(stderr,"Unable to delete file, error %s\n", strerror(errno));
	  }
		delete MemoryManager<false>::allocator;
		delete data_;

  }
 protected:
  void TestFindNearest() {
    
    ComputationsCounter<false> comp;
		// Nearest neighbor
    for(int32 i=0; i<kDataPoints_; i++) {
      Point<float32, uint32, MemoryManager<false> > nearest;
      float32 distance = numeric_limits<float32>::max();
			node_->FindNearest<float32*, 
				Point<float32, unsigned int, MemoryManager<false> >, int> 
				((float32*)kData_[i], nearest, distance, 1, kDimension_, comp);
      CPPUNIT_ASSERT(fabs(distance-kNearestDistances_[i][0])<= 
                     numeric_limits<float32>::epsilon());
      CPPUNIT_ASSERT(nearest==node_->get_points()[kNearestPoints_[i][0]]);                          
    }
    // 2 nearest neighbors
    for(int32 i=0; i<kDataPoints_; i++) {
      vector<pair<float32, Point<float32, uint32, MemoryManager<false> > > > 
				nearest;
      float32 distance=numeric_limits<float32>::max();
      node_->FindNearest(kData_[i], nearest, distance, 2, kDimension_,
                         comp);
      CPPUNIT_ASSERT(fabs(nearest[0].first - kNearestDistances_[i][0])
                     - numeric_limits<float32>::epsilon());
      CPPUNIT_ASSERT(nearest[0].second == 
          node_->get_points()[kNearestPoints_[i][0]]); 
      CPPUNIT_ASSERT(nearest[1].second == 
          node_->get_points()[kNearestPoints_[i][1]]);                                 
    }
     
    // range  nearest neighbors
    for(int32 i=0; i<kDataPoints_-1; i++) {
      vector<pair<float32, Point<float32, uint32, MemoryManager<false> > > > 
				nearest;
      float32 distance=numeric_limits<float32>::max();
      node_->FindNearest(kData_[i], nearest, distance, kRange_, kDimension_,
                         comp);
      CPPUNIT_ASSERT(fabs(nearest[0].first - kNearestDistances_[i][0])
                     - numeric_limits<float32>::epsilon());
      CPPUNIT_ASSERT(nearest[0].second == 
          node_->get_points()[kNearestPoints_[i][0]]); 
      CPPUNIT_ASSERT(nearest[1].second == 
          node_->get_points()[kNearestPoints_[i][1]]);                                  
    }
  }
  void TestFindAllNearest() {
    printf("TestFindAllNearest()\n");
   	float32 max_neighbor_distance;
   	ComputationsCounter<false> comp;
		// All one nearest neaighbors
   	node_->FindAllNearest(node_,
                          max_neighbor_distance,
                          0,
                          1,
                          kDimension_,
                          comp);
    CPPUNIT_ASSERT(node_->get_neighbors() != NULL);
    for(uint32 i=0; i<node_->get_num_of_points(); i++) {
      CPPUNIT_ASSERT(node_->get_neighbors()->at(i)->at(0).second == 
          node_->get_points()[kNearestPoints_[i][0]]);                          
    }
	}
 private:	
	MemoryManager<false>::Ptr<NodeHyperBall_t> node_;
  NodeHyperBall_t::BoundingBox_t::PivotData box_pivot_;
	DataReader<float32, uint32> *data_;
	
};

const float32 EuclideanBallNNTest::kMinHrValues_[EuclideanBallNNTest::kDimension_] 
    = {-1, -1};
const float32 EuclideanBallNNTest::kMaxHrValues_[EuclideanBallNNTest::kDimension_] = 
{1, 1};
float32 EuclideanBallNNTest::kData_[EuclideanBallNNTest::kDataPoints_]
                                   [EuclideanBallNNTest::kDimension_+1] = {
	{0, 0.00, 0}, 
	{0, 0.10, 1},  
	{0, 0.18, 2}, 
  {0, 0.25, 3},
  {0, 0.31, 4}};
const float32 EuclideanBallNNTest::kNearestDistances_[EuclideanBallNNTest::kDataPoints_][2] = {
	{float32(0.1 * 0.1), float32(0.18 * 0.18)},
	{float32(0.08 * 0.08), float32(0.1 * 0.1)},
	{float32(0.07 * 0.07), float32(0.08 * 0.08)},
	{float32(0.06 * 0.06), float32(0.07 * 0.07)},
	{float32(0.06 * 0.06), float32(0.13 * 0.13)}};
const int32 EuclideanBallNNTest::kNearestPoints_[EuclideanBallNNTest::kDataPoints_][2] = {
	{1, 2},
	{2, 0},
	{3, 1},
	{4, 2},
	{3, 2}};	

CPPUNIT_TEST_SUITE_REGISTRATION(EuclideanBallPivotTest);
CPPUNIT_TEST_SUITE_REGISTRATION(EuclideanBallNNTest);
int main( int argc, char **argv)
{
  // Create the event manager and test controller
  CPPUNIT_NS::TestResult controller;

  // Add a listener that colllects test result
  CPPUNIT_NS::TestResultCollector result;
  controller.addListener( &result );        

  // Add a listener that print dots as test run.
  CPPUNIT_NS::BriefTestProgressListener progress;
  controller.addListener( &progress );      

  // Add the top suite to the test runner
  CPPUNIT_NS::TestRunner runner;
  runner.addTest( CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest() );
  runner.run( controller );

  // Print test in a compiler compatible format.
  CPPUNIT_NS::CompilerOutputter outputter( &result, CPPUNIT_NS::stdCOut() );
  outputter.write(); 

  return result.wasSuccessful() ? 0 : 1;
}


