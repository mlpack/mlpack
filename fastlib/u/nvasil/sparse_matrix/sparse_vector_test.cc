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
#include "u/nvasil/sparse_matrix/sparse_vector.h"
/*
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
		val2.Destruct();
	}

	void TestInit1() {
	  Init();
		for(index_t i=0; i<10; i++) {
		  TEST_DOUBLE_APPROX(v1_.get(i), 
			  	               2*i+1,
				                 numeric_limits<Precision_t>::epsilon());
		}
		TEST_DOUBLE_APPROX(v1_.get(0), 0,  numeric_limits<Precision_t>::epsilon());
	  TEST_DOUBLE_APPROX(v1_.get(4), 0,  numeric_limits<Precision_t>::epsilon());
		TEST_DOUBLE_APPROX(v1_.get(24), 0,  numeric_limits<Precision_t>::epsilon());
     Destruct();
	}

	void TestInit2() {
	  std::map mp;
    for(index_t i=0; i<5; i++) {
		  mp[i]=3*i+1;
		}
    SparseVector v;
    v.Init(mp, dim_);
		for(index_t i=0; i<dim_; i++) {
		  TEST_DOUBLE_ASSERT(v.get(i), v2.get(i), numeric_limits<Precision_t>::epsilon());
		}
	}
  void TestCopyConstructor() {
	  SparseVector v(v1);
    for(index_t i=0; i<dim_; i++) {
		  TEST_DOUBLE_ASSERT(v.get(i), v1.get(i), numeric_limits<Precision_t>::epsilon());
	  }
	}
	void TestSet() {
    for(index_t i=0; i<dim_; i++) {
		  v1_.set(i, i);
		}
		for(index_t i=0; i<dim_; i++) {
		  TEST_DOUBLE_ASSERT(v1_.get(i), i, numeric_limits<Precision_t>::epsilon());
	  }
	}
	
	void TestAdd() {
    SparseVector v;
    sparse::Add(v1, v2, &v);
		double epected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]+= 4*i+1;
		}
    for(index_t i=0; i<dim_; i++) {
 		  TEST_DOUBLE_ASSERT(v.get(i), expected_result[i], 
					               numeric_limits<Precision_t>::epsilon());
		}
	}
	
	void TestSubtract() {
	  SparseVector v;
		sparse::Subtract(v1, v2, &v);
    double epected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]-= 4*i+1;
		}
    for(index_t i=0; i<dim_; i++) {
 		  TEST_DOUBLE_ASSERT(v.get(i), expected_result[i], 
					               numeric_limits<Precision_t>::epsilon());
		}
	}
	
	void TestPointProduct() {
	  SparseVector v;
		sparse::PointProduct(v1, v2, &v);
    double epected_result[dim_];
		memset(expected_result, 0, dim_*sizeof(double));
    for(index_t i=0; i<10; i++) {
		  expected_result[2*i+1]+= 3*i+1;
		}
    for(index_t i=0; i<5; i++) {
		  expected_result[2*i+1]*= 4*i+1;
		}
    for(index_t i=0; i<dim_; i++) {
 		  TEST_DOUBLE_ASSERT(v.get(i), expected_result[i], 
					               numeric_limits<Precision_t>::epsilon());
		}
	}
	
	void TestDotProduct() {
	  double dot_prod;
		sparse::DotProduct(v1, v2, &dot_prod);
    double epected_result[dim_];
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
    TEST_DOUBLE_ASSERT(dot_prod, 
				               expected_dot_prod,
					             numeric_limits<Precision_t>::epsilon());

	}
	
	void TestDistance() {
	  double dist;
		sparse::Distance(v1, v2, &dist);
    double epected_result[dim_];
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
    TEST_DOUBLE_ASSERT(distance, 
				               dist,
					             numeric_limits<Precision_t>::epsilon());
	}
	
	void TestAll() {
		TestInit1(); 
		TestInit2();
    TestCopyConstructor();
    TestGet();
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
  double                *val1;	
	Vector                 val2;
	index_t                dim_;
};

int main() {
  SparseVectorTest test;
	test.TestAll();
}
*/
