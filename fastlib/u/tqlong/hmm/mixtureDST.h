#ifndef FASTLIB_MIXTURE_GAUSSIAN_H
#define FASTLIB_MIXTURE_GAUSSIAN_H
#include "fastlib/fastlib.h"
#include "support.h"
class MixtureGauss {
  ArrayList<Vector> means;
  ArrayList<Matrix> covs;
  Vector prior;

  ArrayList<Matrix> inv_covs;
  Vector det_covs;

  ArrayList<Vector> ACC_means;
  ArrayList<Matrix> ACC_covs;
  Vector ACC_prior;
  double total;
 public:
  void InitFromFile(const char* mean_fn, const char* covs_fn = NULL, const char* prior_fn = NULL);
  void InitFromProfile(const ArrayList<Matrix>& matlst, int start, int N);
  void Init(int K, int N);
  void Init(int K, const Matrix& data, const ArrayList<int>& labels);
  void print_mixture(const char* s) const;
  void generate(Vector* v) const;
  double MixtureGauss::getPDF(const Vector& v) const;
  double MixtureGauss::getPDF(int cluster, const Vector& v) const;
  const Vector& get_prior() const { return prior; }
  const Vector& get_mean(int k) const { return means[k]; }
  const Matrix& get_cov(int k) const { return covs[k]; }
  int n_clusters() const { return means.size(); }
  int v_length() const { return means[0].length(); }

  void start_accumulate() {
    total = 0;
    for (int i = 0; i < means.size(); i++) {
      ACC_means[i].SetZero();
      ACC_covs[i].SetZero();
      ACC_prior.SetZero();
    }
  }
  void accumulate(const Vector& v) {
    double s = getPDF(v);
    for (int i = 0; i < means.size(); i++) {
      double p = getPDF(i, v) / s;
      ACC_prior[i] += p;
      la::AddExpert(p, v, &ACC_means[i]);
      Vector d;
      la::SubInit(v, means[i], &d);
      Matrix D;
      D.AliasColVector(d);
      la::MulExpert(p, false, D, true, D, 1.0, &ACC_covs[i]);
    }
    total ++;
  }
  void accumulate_cluster(int i, const Vector& v) {
    la::AddTo(v, &ACC_means[i]);
    Matrix V;
    V.AliasColVector(v);
    la::MulExpert(1.0, false, V, true, V, 1.0, &ACC_covs[i]);
    ACC_prior[i]++;
    total++;
  }
  void accumulate(double p, int i, const Vector& v) {
    ACC_prior[i] += p;
    la::AddExpert(p, v, &ACC_means[i]);
    Vector d;
    la::SubInit(v, means[i], &d);
    Matrix D;
    D.AliasColVector(d);
    la::MulExpert(p, false, D, true, D, 1.0, &ACC_covs[i]);
    total += p;
  }
  void end_accumulate_cluster() {
    for (int i = 0; i < means.size(); i++) 
      if (ACC_prior[i] != 0) {
	la::ScaleOverwrite(1.0/ACC_prior[i], ACC_means[i], &means[i]);
	Matrix M;
	M.AliasColVector(means[i]);
	la::MulExpert(-1.0, false, M, true, M, 1.0/ACC_prior[i], &ACC_covs[i]);
	covs[i].CopyValues(ACC_covs[i]);
	prior[i] = ACC_prior[i]/total;

	double det = la::Determinant(covs[i]);
	la::InverseOverwrite(covs[i], &inv_covs[i]);
	det_covs[i] = pow(2.0*math::PI, -means[i].length()/2.0) * pow(det, -0.5);
      }
  }
  void end_accumulate() {
    for (int i = 0; i < means.size(); i++) {
      if (ACC_prior[i] != 0) {
	la::ScaleOverwrite(1.0/ACC_prior[i], ACC_means[i], &means[i]);
	la::ScaleOverwrite(1.0/ACC_prior[i], ACC_covs[i], &covs[i]);
	prior[i] = ACC_prior[i]/total;

	double det = la::Determinant(covs[i]);
	la::InverseOverwrite(covs[i], &inv_covs[i]);
	det_covs[i] = pow(2.0*math::PI, -means[i].length()/2.0) * pow(det, -0.5);
      }
    }
  }
};
#endif
