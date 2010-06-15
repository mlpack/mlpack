/**
 * @file mixtureDST.cc
 *
 * This file contains implementation of functions in mixtureDST.h
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "mixtureDST.h"

using namespace hmm_support;

void MixtureGauss::Init(int K, int N) {
  for (int i = 0; i < K; i++) {
    Vector v;
    RAND_NORMAL_01_INIT(N, &v);
    means.push_back(v);
  }
  
  for (int i = 0; i < means.size(); i++) {
    Matrix m;
    m.Init(N, N); m.SetZero();
    for (int j = 0; j < N; j++) m.ref(j, j) = 1.0;
    covs.push_back(m);
  }

  prior.Init(means.size());
  for (int i = 0; i < prior.length(); i++) prior[i] = 1.0/K;

  ACC_means.assign(means.begin(), means.end());
  ACC_covs.assign(covs.begin(), covs.end());
  ACC_prior.Init(K);
  inv_covs.assign(covs.begin(), covs.end());
  det_covs.Init(covs.size());
  for (int i = 0; i < K; i++) {
    double det = la::Determinant(covs[i]);
    la::InverseOverwrite(covs[i], &inv_covs[i]);
    det_covs[i] = pow(2.0*math::PI, -N/2.0) * pow(det, -0.5);
  }
}

void MixtureGauss::Init(int K, const Matrix& data, const std::vector<int>& labels) {
  int N = data.n_rows();
  for (int i = 0; i < K; i++) {
    Vector v;
    v.Init(N);
    means.push_back(v);
  }
  
  for (int i = 0; i < means.size(); i++) {
    Matrix m;
    m.Init(N, N);
    covs.push_back(m);
  }

  prior.Init(means.size());

  ACC_means.assign(means.begin(), means.end());
  ACC_covs.assign(covs.begin(), covs.end());
  ACC_prior.Init(K);
  inv_covs.assign(covs.begin(), covs.end());
  det_covs.Init(covs.size());
  start_accumulate();
  //printf("cols = %d rows = %d\n", data.n_cols(), data.n_rows()); 
  for (int i = 0; i < data.n_cols(); i++) {
    Vector v;
    data.MakeColumnVector(i, &v);
    //printf("%d\n", i);
    accumulate_cluster(labels[i], v);
  }
  end_accumulate_cluster();
}

void MixtureGauss::InitFromFile(const char* mean_fn, const char* covs_fn, const char* prior_fn) {
  Matrix meansmat;
  data::Load(mean_fn, &meansmat);
  mat2arrlst(meansmat, &means);
  int N = means[0].length();
  int K = means.size();
  if (covs_fn != NULL) {
    Matrix covsmat;
    data::Load(covs_fn, &covsmat);
    mat2arrlstmat(N, covsmat, &covs);
    DEBUG_ASSERT_MSG(K==covs.size(), "InitFromFile: sizes do not match !");
  }
  else {
    for (int i = 0; i < means.size(); i++) {
      Matrix m;
      m.Init(N, N); m.SetZero();
      for (int j = 0; j < N; j++) m.ref(j, j) = 1.0;
      covs.push_back(m);
    }
  }
  if (prior_fn != NULL) {
    Matrix priormat;
    data::Load(prior_fn, &priormat);
    DEBUG_ASSERT_MSG(K==priormat.n_cols(), "InitFromFile: sizes do not match !!");
    prior.Init(K);
    for (int i = 0; i < K; i++) prior[i] = priormat.get(0, i);
  }
  else {
    prior.Init(means.size());
    for (int i = 0; i < prior.length(); i++) prior[i] = 1.0/K;
  }
  
  ACC_means.assign(means.begin(), means.end());
  ACC_covs.assign(covs.begin(), covs.end());
  ACC_prior.Init(K);
  inv_covs.assign(covs.begin(), covs.end());
  det_covs.Init(covs.size());
  for (int i = 0; i < K; i++) {
    double det = la::Determinant(covs[i]);
    la::InverseOverwrite(covs[i], &inv_covs[i]);
    det_covs[i] = pow(2.0*math::PI, -N/2.0) * pow(det, -0.5);
  }
}

void MixtureGauss::InitFromProfile(const std::vector<Matrix>& matlst, int start, int N) {
  DEBUG_ASSERT(matlst[start].n_cols()==1);
  Vector tmp;
  matlst[start].MakeColumnVector(0, &tmp);
  prior.Copy(tmp);

  // DEBUG: print_vector(prior, "  prior = ");

  int K = prior.length();
  for (int i = start+1; i < start+2*K+1; i+=2) {
    DEBUG_ASSERT(matlst[i].n_rows()==N && matlst[i].n_cols()==1);
    DEBUG_ASSERT(matlst[i+1].n_rows()==N && matlst[i+1].n_cols()==N);
    Vector m;
    matlst[i].MakeColumnVector(0, &m);
    means.push_back(m);
    covs.push_back(matlst[i+1]);    
  }
  ACC_means.assign(means.begin(),means.end());
  ACC_covs.assign(covs.begin(),covs.end());
  ACC_prior.Init(K);
  inv_covs.assign(covs.begin(),covs.end());
  det_covs.Init(covs.size());
  for (int i = 0; i < K; i++) {
    double det = la::Determinant(covs[i]);
    la::InverseOverwrite(covs[i], &inv_covs[i]);
    det_covs[i] = pow(2.0*math::PI, -N/2.0) * pow(det, -0.5);
  }  
}

void MixtureGauss::print_mixture(const char* s) const {
  int K = means.size();
  printf("%s - Mixture (%d)\n", s, K);
  print_vector(prior, "  PRIOR");
  for (int i = 0; i < K; i++) {
    printf("  CLUSTER %d:\n", i);
    print_vector(means[i], "  MEANS");
    print_matrix(covs[i], "  COVS");
  }
}

void MixtureGauss::generate(Vector* v) const {
  int K = means.size();
  double r = RAND_UNIFORM_01();
  int cluster = K-1;
  double s = 0;
  for (int i = 0; i < K; i++) {
    s += prior[i];
    if (s >= r) {
      cluster = i;
      break;
    }
  }
  RAND_NORMAL_INIT(means[cluster], covs[cluster], v);
}

double MixtureGauss::getPDF(const Vector& v) const {
  int K = means.size();
  double s = 0;
  for (int i = 0; i < K; i++)
    s += getPDF(i, v);
  return s;
}

double MixtureGauss::getPDF(int cluster, const Vector& v) const {
  return prior[cluster]*NORMAL_DENSITY(v, means[cluster], inv_covs[cluster], det_covs[cluster]);
}
