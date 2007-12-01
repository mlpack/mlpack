/*
 * =====================================================================================
 * 
 *       Filename:  kernel_pca.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  11/30/2007 08:34:19 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef KERNEL_PCA_H_
#define KERNEL_PCA_H_
#ifndef HAVING_CONFIG_H
#define HAVE_CONFIG_H
#endif
#include <string>
#include "fastlib/fastlib.h"
#include "fastlib/la/matrix.h"
#include "u/nvasil/binary_dataset.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Version.h"

template<typename TREE, bool diagnostic>
class KernelPCA {
	typedef typename TREE::Precision_t   Precision_t;
	typedef TREE                         Tree_t;
	typdef enum OutputMode_t {Matrix_E=0, TexFile_E, BinaryFile_E};
 public:
	void Init();
  void Destruct();
	void ComputeAffinity(index_t knn);
  template<typename DISTANCEKERNEL>		
	void ComputeGeneralKernelPCA(DISTANCEKERNEL kernel, index_t num_of_eigenvalues);
	void ComputeIsomap(index_t num_of_eigenvalues);
	void ComputeLLE(index_t num_of_eigenvalues);
	template<KERNEL>
	void ComputeDiffusionMaps(KERNEL kernel, index_t num_of_eigenvalues);
	void ComputeLaplacialnEigenmaps(index_t);
	void ComputeSpectralRegression(std::string label_file);
  Vector get_eigenvalues();
	Matrix get_eigenvectors();
	void SaveToTextFile(std::string file);
	void SaveToBinaryFile(std::string file);
  void set_output_mode(OutputMode_t mode) {
	  output_mode_ = mode;
	}
	
 private:
  Tree_t tree_;
  Epetra_SerialComm comm_;
  Epetra_CrsMatrix kernel_matrix_;	
  BinaryDataset<Precision_t> data_; 
  OutputMode_t output_mode_;

} ;

#include "u/nvasil/kernel_pca/kernel_pca_impl.h"
#endif
