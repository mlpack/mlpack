/*
 * =====================================================================================
 *
 *       Filename:  kernel_pca_test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/09/2008 11:26:48 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#include "u/nvasil/kernel_pca/kernel_pca.h"
#include "fastlib/fastlib.h"
#include "base/test.h"
#include <vector>

class KernelPCATest {
 public:		
	void Init() {
	  engine_ = new KernelPCA();
		engine_->Init("test_data_3_1000.csv", 5, 20);
	}
	void Destruct() {
	  delete engine_;
	}
	void TestGeneralKernelPCA() {
   NONFATAL("Testing KernelPCA...\n");
	 Matrix eigen_vectors;
   std::vector<double> eigen_values;
	 Init();
   engine_->ComputeNeighborhoods();
	 double bandwidth;
	 engine_->EstimateBandwidth(&bandwidth);
	 NONFATAL("Estimated bandwidth %lg ...\n", bandwidth);
   kernel_.set(bandwidth); 
   engine_->LoadAffinityMatrix();
	 engine_->ComputeGeneralKernelPCA(kernel_, 5, 
			                              &eigen_vectors,
																		&eigen_values);

	 engine_->SaveToTextFile("results", eigen_vectors, eigen_values);
	 Destruct();
   NONFATAL("Test ComputeGeneralKernelPCA passed...!\n");
	}
	void TestLLE() {
		NONFATAL("Testing Compute LLE\n");
    Matrix eigen_vectors;
    std::vector<double> eigen_values;
	  Init();
    engine_->ComputeNeighborhoods();
    engine_->LoadAffinityMatrix();
	 	engine_->ComputeLLE(5,
		 	                  &eigen_vectors,
									      &eigen_values);
	  engine_->SaveToTextFile("results", eigen_vectors, eigen_values);
	  Destruct();
	  NONFATAL("Test ComputeLLE passed...!\n");
  }
	void TestAll() {
	   TestGeneralKernelPCA();
		 TestLLE();
	}
 private:
	KernelPCA *engine_;
	KernelPCA::GaussianKernel kernel_;
};

int main() {
  KernelPCATest test;
	test.TestAll();
}
