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

#include "u/nvasil/kernel_pca.h"
#include "fastlib/fastlib.h"
#include "base/test.h"
#include <vector>

class KernelPCATest {
 public:		
	void Init() {
	  engine_ = new KernelPCA();
		engine_->Init("test_data", "");
		kernel.set(1.0);
	}
	void Destruct() {
	  delete engine_;
	}
	void TestGeneralKernelPCA() {
   Matrix eigen_vectors;
   std::vector<double> eigen_values;
	 Init();
   engine_->ComputeNeighborhoods(10);
   engine_->LoadAffinityMatrix();
	 engine_->ComputeGeneralKernelPCA(kernel, 3, 
			                              &eigen_vectors,
																		&eigen_values);
	 engine_->SaveToTextFile("kernel_pca_results");
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
	test.All();
}
