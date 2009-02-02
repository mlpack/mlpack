
#ifndef FASTLIB_MIXTURE_GAUSSIAN_DISTRIBUTION_H
#define FASTLIB_MIXTURE_GAUSSIAN_DISTRIBUTION_H
#include <fastlib/fastlib.h>

class MixtureGaussianDistribution {
  ArrayList<double> pMixtures;
  ArrayList<GaussianDistribution> gDistributions;
 public:
  MixtureGaussianDistribution(index_t n_mixs, index_t dim);
  MixtureGaussianDistribution(const MixtureGaussianDistribution& mgd);
	
  double logP(const Vector& x);	
  static void createFromCols(const Matrix& src,
			     index_t col, GaussianDistribution* tmp);
  void Generate(Vector* x);
  void StartAccumulate();
  void EndAccumulate();
  void Accumulate(const Vector& x, double weight);
			
  void Save(FILE* f);	
  const GaussianDistribution& getCluster(index_t i) 
    { return gDistributions[i]; }
  const Matrix& getCov() { return covariance; }
  index_t n_dim() { return mean.length(); }
  index_t n_mixs() { return pMixtures.size(); }
};

#endif
