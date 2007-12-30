
#include "fastlib/fastlib.h"
#include "support.h"
#include "mixtureDST.h"

void MixtureGauss::Init(int K, int N) {
  means.Init();
  for (int i = 0; i < K; i++) {
    Vector v;
    RAND_NORMAL_01_INIT(N, &v);
    means.AddBackItem(v);
  }
  
  covs.Init();
  for (int i = 0; i < means.size(); i++) {
    Matrix m;
    m.Init(N, N); m.SetZero();
    for (int j = 0; j < N; j++) m.ref(j, j) = 1.0;
    covs.AddBackItem(m);
  }

  prior.Init(means.size());
  for (int i = 0; i < prior.length(); i++) prior[i] = 1.0/K;

  ACC_means.Copy(means);
  ACC_covs.Copy(covs);
  ACC_prior.Init(K);
  inv_covs.Copy(covs);
  det_covs.Init(covs.size());
  for (int i = 0; i < K; i++) {
    double det = la::Determinant(covs[i]);
    la::InverseOverwrite(covs[i], &inv_covs[i]);
    det_covs[i] = pow(2.0*math::PI, -N/2.0) * pow(det, -0.5);
  }
}

void MixtureGauss::Init(int K, const Matrix& data, const ArrayList<int>& labels) {
  means.Init();
  int N = data.n_rows();
  for (int i = 0; i < K; i++) {
    Vector v;
    v.Init(N);
    means.AddBackItem(v);
  }
  
  covs.Init();
  for (int i = 0; i < means.size(); i++) {
    Matrix m;
    m.Init(N, N);
    covs.AddBackItem(m);
  }

  prior.Init(means.size());

  ACC_means.Copy(means);
  ACC_covs.Copy(covs);
  ACC_prior.Init(K);
  inv_covs.Copy(covs);
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
    covs.Init();
    for (int i = 0; i < means.size(); i++) {
      Matrix m;
      m.Init(N, N); m.SetZero();
      for (int j = 0; j < N; j++) m.ref(j, j) = 1.0;
      covs.AddBackItem(m);
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
  
  ACC_means.Copy(means);
  ACC_covs.Copy(covs);
  ACC_prior.Init(K);
  inv_covs.Copy(covs);
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
  double r = RAND_UNIFORM_01;
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
