/*
 * =====================================================================================
 *
 *       Filename:  sparse_vector_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/04/2007 08:20:52 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */
#include <limits>
#include <vector>
#include <map>
#include "fastlib/fastlib.h"
#include "u/nvasil/test/test.h"
#include "u/nvasil/sparse_matrix/sparse_vector.h"

class SparseVectorTest {
 public:
	void Init() {
    dim_  = 100;
	  ind1_	= new index_t[10];
		val1_ = new double[10];
		val2_.Init(5);
		dim_  = 40;
		for(index_t i=0; i<10; i++) {
		  ind1_[i] = 2*i+1;
			val1_[i] = 3*i+1;
		}
		
		for(index_t i=0; i<5; i++) {
		  ind2_.push_back(2*i+1);
			val2_[i] = 4*i+1;
		}
    v1_.Init(ind1_, val1_, 10, dim_);
    v2_.Init(ind2_, val2_, dim_);
	}
  
	void Destruct() {
	  v1_.Destruct();
		v2_.Destruct();
		delete []ind1_;
		delete []val1_;
		ind2_.clear();
		val2_.Destruct();
	}

	void TestInit1() {
	  Init();
		for(index_t i=0; i<10; i++) {
		  TEST_DOUBLE_APPROX(v1_.get(i), 
			  	               2*i+1,
				                 std::numeric_limits<double>::epsilon());
		}
		TEST_DOUBLE_APPROX(v1_.get(0), 0,  std::numeric_limits<double>::epsilon());
	  TEST_DOUBLE_APPROX(v1_.get(4), 0,  std::numeric_limits<double>::epsilon());
		TEST_DOUBLE_APPROX(v1_.get(24), 0,  std::numeric_limits<double>::epsilon());
     Destruct();
	}

	void TestInit2() {
	  std::map<index_t, double> mp;
    for(index_t i=0; i<5; i++) {
		  mp[i]=3*i+1;
		}
    SparseVector v;
    v.Init(mp, dim_);
		for(index_t i=0; i<dim_; i++) {
		  TEST_DOUBLE_APPROX(v.get(i), v2_.get(i), std::numeric_limits<double>::epsilon());
		}
	}
  void TestCopyConstructor() {
	  SparseVector v(v1_);
    for(index_t i=0; i<dim_; i++) {
		  TEST_DOUBLE_APPROX(v.get(i), v1_.get(i), std::numeric_limits<double>::epsilon());
	  }
	}
	void TestSet() {
    for(index_t i=0; i<dim_; i++) {
		  v1_.set(i, i);
		}
		for(index_t i=0; i<dim_; i++) {
		  TEST_DOUBLE_APPROX(v1_.get(i), i, std::numeric_limits<double>::epsilon());
	  }
	}
	
	void TestAdd() {
    SparseVector v;
    sparse::AddVectors(v1_, v2_, &v);
		double expected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]+= 4*i+1;
		}
    for(index_t i=0; i<dim_; i++) {
 		  TEST_DOUBLE_APPROX(v.get(i), expected_result[i], 
					               std::numeric_limits<double>::epsilon());
		}
	}
	
	void TestSubtract() {
	  SparseVector v;
		sparse::SubtractVectors(v1_, v2_, &v);
    double expected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]-= 4*i+1;
		}
    for(index_t i=0; i<dim_; i++) {
 		  TEST_DOUBLE_APPROX(v.get(i), expected_result[i], 
					               std::numeric_limits<double>::epsilon());
		}
	}
	
	void TestPointProduct() {
	  SparseVector v;
		sparse::PointProductVectors(v1_, v2_, &v);
    double expected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]*= 4*i+1;
		}
    for(index_t i=0; i<dim_; i++) {
 		  TEST_DOUBLE_APPROX(v.get(i), expected_result[i], 
					               std::numeric_limits<double>::epsilon());
		}
	}
	
	void TestDotProduct() {
	  double dot_prod;
		sparse::DotProductVectors(v1_, v2_, &dot_prod);
    double expected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]*= 4*i+1;
		}
		double expected_dot_prod=0;
    for(index_t i=0; i<dim_; i++) {
			expected_dot_prod+=expected_result[i];
 		}
    TEST_DOUBLE_APPROX(dot_prod, 
				               expected_dot_prod,
					             std::numeric_limits<double>::epsilon());

	}
	
	void TestDistance() {
	  double dist;
		sparse::DistanceSqEuclideanVector(v1_, v2_, &dist);
    double expected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]-= 4*i+1;
		}
		double distance=0;
    for(index_t i=0; i<dim_; i++) {
			distance = expected_result[i] * expected_result[i];
 		}
    TEST_DOUBLE_APPROX(distance, 
				               dist,
					             std::numeric_limits<double>::epsilon());
	}
	
	void TestAll() {
		TestInit1(); 
		TestInit2();
    TestCopyConstructor();
	  TestSet();
	  TestAdd();
	  TestSubtract();
	  TestPointProduct();
	  TestDotProduct();
	  TestDistance();
	}

 private:
  SparseVector            v1_;
	SparseVector            v2_;
	SparseVector            v3_;
	index_t              *ind1_;
	std::vector<index_t>  ind2_;
  double                *val1_;	
	Vector                 val2_;
	index_t                dim_;
};

int main() {
  SparseVectorTest test;
	test.TestAll();
}

