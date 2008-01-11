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
		engine_->Init("test_data_3_1000", "");
	}
	void Destruct() {
	  delete engine_;
	}
	void TestGeneralKernelPCA() {
   Matrix eigen_vectors;
   std::vector<double> eigen_values;
	 Init();
   engine_->ComputeNeighborhoods(10);
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
	}
	void TestAll() {
	  TestGeneralKernelPCA();
	}
 private:
	KernelPCA *engine_;
	KernelPCA::GaussianKernel kernel_;
};

int main() {
  KernelPCATest test;
	test.TestAll();
}
