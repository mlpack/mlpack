#include "mixtureGaussianDistribution.h"
#include "gaussianDistribution.h"
#include "support.h"

using namespace supportHMM;

MixtureGaussianDistribution::GaussianDistribution(index_t n_mixs, index_t dim) {
  pMixtures.Init();
  gDistributions.Init();

  double s = 0.0;
  for (index_t i = 0; i < n_mixs; i++) {
    double r = math::Random(0, 1);
    s += r;
    pMixtures.PushBackCopy(r);
  }
  for (index_t i = 0; i < n_mixs; i++)
    pMixtures[i] /= s;

  for (index_t i = 0; i < n_mixs; i++) {
    GaussianDistribution gd(dim);
    gDistributions.PushBackCopy(gd);
  }
}

MixtureGaussianDistribution::MixtureGaussianDistribution(const MixtureGaussianDistribution& mgd) {
  pMixtures.Copy(mgd.pMixtures);
  gDistributions.Copy(mgd.gDistributions);
}

double MixtureGaussianDistribution::logP(const Vector& x) {
  double s = 0;
  for (index_t i = 0; i < n_mixs(); i++) 
    s += pMixtures[i] * exp(gDistributions[i].logP(x));
  return log(s);
}

void MixtureGaussianDistribution::createFromFile(
    TextLineReader& f, MixtureGaussianDistribution* tmp) {
  index_t dim = src.n_rows();
  for (index_t i = 0; i < n_mixs; i++) {
  }
}

void GaussianDistribution::Generate(Vector* x) {
  RandomNormal(mean, sqrCov, x);
}

void GaussianDistribution::StartAccumulate() {
  accMean.SetZero();
  accCov.SetZero();
  accDenom = 0.0;
}

void GaussianDistribution::EndAccumulate() {
  la::ScaleInit(1.0/accDenom, accMean, &mean);
  Matrix meanMat;
  meanMat.AliasColVector(mean);
  la::ScaleOverwrite(1.0/accDenom, accCov, &covariance);
  la::MulExpert(-1.0, false, meanMat, true, meanMat, 1.0, &covariance);
}

void GaussianDistribution::Accumulate(const Vector& x, double weight) {
  la::AddExpert(weight, x, &accMean);
  Matrix xMat;
  xMat.AliasColVector(x);
  la::MulExpert(weight, false, xMat, true, xMat, 1.0, &accCov);
  accDenom += weight;
}

void GaussianDistribution::Save(FILE* f) {
  printVector(f, mean);
  printMatrix(f, covariance);
}

void GaussianDistribution::InitMeanCov(index_t dim) {
  mean.Init(dim);
  covariance.Init(dim, dim);
	
  invCov.Init(dim, dim);
  sqrCov.Init(dim, dim);
	
  accMean.Init(dim);
  accCov.Init(dim, dim);
}

void GaussianDistribution::setMeanCov(const Vector& m, const Matrix& cov) {
  this->mean.CopyValues(m);
  this->covariance.CopyValues(cov);
  la::InverseOverwrite(cov, &invCov);
  Matrix tmp;
  la::CholeskyInit(cov, &tmp);
  sqrCov.CopyValues(tmp);
  gConst = -0.5*n_dim()*log(2*math::PI)-0.5*la::DeterminantLog(cov, NULL);
}

