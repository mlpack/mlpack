#include "fastlib/fastlib_int.h"
#include "ifgt_kde.h"
#include "ifgt_choose_parameters.h"
#include "kde.h"
#include "ifgt_choose_truncation_number.h"
#include "kcenter_clustering.h"

void concatenate_vectors(Matrix &source, Vector &dest) {

  for(index_t i = 0; i < source.n_cols(); i++) {
    for(index_t j = 0; j < source.n_rows(); j++) {
      dest[i * source.n_rows() + j] = source.get(j, i);
    }
  }
}

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  // read the datasets and do k-center clustering
  Dataset ref_dataset;
  Matrix qset_;
  Matrix rset_;
  Vector pWeights;

  // read the datasets
  const char *rfname = fx_param_str_req(NULL, "data");
  const char *qfname = fx_param_str(NULL, "query", rfname);

  // read reference dataset
  ref_dataset.InitFromFile(rfname);
  rset_.Own(&(ref_dataset.matrix()));

  // read the reference weights
  char *rwfname = NULL;
  if(fx_param_exists(NULL, "dwgts")) {
    rwfname = (char *)fx_param_str(NULL, "dwgts", NULL);
  }

  if(rwfname != NULL) {
    Dataset ref_weights;
    ref_weights.InitFromFile(rwfname);
    pWeights.Copy(ref_weights.matrix().GetColumnPtr(0),
		  ref_weights.matrix().n_rows());
  }
  else {
    pWeights.Init(rset_.n_cols());
    pWeights.SetAll(1);
  }

  if(!strcmp(qfname, rfname)) {
    qset_.Alias(rset_);
  }
  else {
    Dataset query_dataset;
    query_dataset.InitFromFile(qfname);
    qset_.Own(&(query_dataset.matrix()));
  }

  Vector pSources;
  pSources.Init(rset_.n_rows() * rset_.n_cols());
  concatenate_vectors(rset_, pSources);
  Vector pTargets;
  pTargets.Init(qset_.n_rows() * qset_.n_cols());
  concatenate_vectors(qset_, pTargets);

  double Bandwidth = fx_param_double_req(NULL, "bandwidth");
  Vector pGaussTransform;
  pGaussTransform.Init(qset_.n_cols());
  double epsilon = fx_param_double(NULL, "tau", 0.1);


  // choose parameters
  ImprovedFastGaussTransformChooseParameters cp(rset_.n_rows(), Bandwidth,
						epsilon, 
						(int) ceil(0.2 * 100 / 
							   sqrt
							   (2 * Bandwidth * 
							    Bandwidth)));
  
  printf("Number of clusters chosen: %d\n", cp.K);
  printf("Maximum truncation number: %d\n", cp.p_max);
  printf("Maximum cutoff radius: %g\n", cp.r);
  
  // run k-center clustering
  ArrayList<int> pClusterIndex;
  pClusterIndex.Init(rset_.n_cols());
  KCenterClustering kc(rset_.n_rows(), rset_.n_cols(), pSources.ptr(),
		       pClusterIndex.begin(), cp.K);
  
  printf("Clustering %d...\n", cp.K);
  kc.Cluster();

  for(index_t i = 0;i < pClusterIndex.size(); i++) {
    printf("%d ", pClusterIndex[i]);
  }
  printf("\n");

  ArrayList<int> pNumPoints;
  pNumPoints.Init(cp.K);
  Vector pClusterCenter;
  pClusterCenter.Init(rset_.n_rows() * cp.K);
  Vector pClusterRadii;
  pClusterRadii.Init(cp.K);
  kc.ComputeClusterCenters(cp.K, pClusterCenter.ptr(), pNumPoints.begin(),
			   pClusterRadii.ptr());

  // update truncation number
  ImprovedFastGaussTransformChooseTruncationNumber ct(rset_.n_rows(), 
						      Bandwidth, epsilon,
						      kc.MaxClusterRadius);

  // initialize IFGT instance
  ArrayList<int> pTruncNumber;
  pTruncNumber.Init(rset_.n_cols());
  for(index_t i = 0; i < rset_.n_cols(); i++) {
    pTruncNumber[i] = 0;
  }
  ImprovedFastGaussTransform* pIFGT = new 
    ImprovedFastGaussTransform(rset_.n_rows(), rset_.n_cols(), qset_.n_cols(), 
			       pSources.ptr(), Bandwidth, pWeights.ptr(),
			       pTargets.ptr(), ct.p_max,
			       cp.K, pClusterIndex.begin(),
			       pClusterCenter.ptr(), pClusterRadii.ptr(),
			       cp.r, epsilon, pGaussTransform.ptr(),
			       pTruncNumber.begin());

  // run IFGT
  pIFGT->Evaluate();
  FILE *stream = stdout;
  const char *fname = NULL;
  
  GaussianKernel kernel;
  kernel.Init(Bandwidth);
  double norm_const = kernel.CalcNormConstant(qset_.n_rows()) *
    rset_.n_cols();


  for(index_t q = 0; q < qset_.n_cols(); q++) {
    pGaussTransform[q] /= norm_const;
  }
  if((fname = fx_param_str(NULL, "ifgt_kde_output", NULL)) != NULL) {
    stream = fopen(fname, "w+");
  }
  for(index_t q = 0; q < qset_.n_cols(); q++) {
    fprintf(stream, "%g\n", pGaussTransform[q]);
  }

  if(stream != stdout) {
    fclose(stream);
  }


  // check answer with naive
  NaiveKde<GaussianKernel> naive_kde;
  naive_kde.Init(qset_, rset_);
  naive_kde.Compute();
  naive_kde.ComputeMaximumRelativeError(pGaussTransform);
  naive_kde.PrintDebug();

  delete pIFGT;

  fx_done();
  return 0;
}
