
#include <fastlib/fastlib.h>
#include <map>
#include "support.h"
#include "discreteHMM.h"

using namespace hmm_support;

class TriValues {
public:
  int v1, v2, v3;
  TriValues(int v1, int v2, int v3) {
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
  }
  TriValues(double v1, double v2, double v3) {
    this->v1 = (int)v1;
    this->v2 = (int)v2;
    this->v3 = (int)v3;
  }
  friend bool operator < (const TriValues &a, const TriValues &b);
};

bool operator < (const TriValues &a, const TriValues &b) {
  return (a.v1 < b.v1) || (a.v1 == b.v1 && a.v2 < b.v2) ||
    (a.v1 == b.v1 && a.v2 == b.v2 && a.v3 < b.v3);
}

typedef std::map<TriValues, double> TriValMap;

void CalObsFreq(const ArrayList<Vector>& list, TriValMap* obsmap) {
  double total = 0;
  for (int i = 0; i < list.size(); i++) {
    for (int j = 0; j < list[i].length()-3; j++) {
      total += 1;
      TriValues obs(list[i][j], list[i][j+1], list[i][j+2]);
      TriValMap::iterator it = obsmap->find(obs);
      if (it != obsmap->end()) it->second += 1;
      else (*obsmap)[obs] = 1;
    }
  }
  for (TriValMap::iterator it = obsmap->begin(); it != obsmap->end(); it++)
    it->second /= total;
}

double*** Create3DArray(int m, int n, int p) {
  double*** arr = new double** [m];
  for (int i = 0; i < m; i++) {
    arr[i] = new double*[n];
    for (int j = 0; j < n; j++)
      arr[i][j] = new double[p];
  }
  return arr;
}

double** Create2DArray(int m, int n) {
  double** arr = new double* [m];
  for (int i = 0; i < m; i++) arr[i] = new double[n];
  return arr;
}

double* Create1DArray(int m) {
  double* arr = new double [m];
  return arr;
}

void SetZero3D(double*** arr, int m, int n, int p) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
	arr[i][j][k] = 0;
}

void Copy3D(double*** src, double*** dst, int m, int n, int p) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++) dst[i][j][k] = src[i][j][k];
}

void Copy2D(double** src, double** dst, int m, int n) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) dst[i][j] = src[i][j];
}

void SetZero2D(double** arr, int m, int n) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
	arr[i][j] = 0;
}

void SetZero1D(double* arr, int m) {
  for (int i = 0; i < m; i++) arr[i] = 0;
}

void SetRandom3D(double*** arr, int m, int n, int p) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
	arr[i][j][k] = (double) rand() / RAND_MAX + 1e-10;
}

void SetRandom2D(double** arr, int m, int n) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
	arr[i][j] = (double) rand() / RAND_MAX + 1e-10;
}

void SetRandom1D(double* arr, int m) {
  for (int i = 0; i < m; i++)
    arr[i] = (double) rand() / RAND_MAX + 1e-10;
}

void SetSumUnity3D(double*** arr, int m, int n, int p) {
  double sum = 0;
  for (int s1s = 0; s1s < m; s1s++)
    for (int s2s = 0; s2s < n; s2s++)
      for (int s3s = 0; s3s < p; s3s++) 
	sum += arr[s1s][s2s][s3s];
  for (int s1s = 0; s1s < m; s1s++)
    for (int s2s = 0; s2s < n; s2s++)
      for (int s3s = 0; s3s < p; s3s++) 
	arr[s1s][s2s][s3s] *= (1/sum);
}

void SetSumUnity2D(double** arr, int m, int n) {
  double sum = 0;
  for (int s1s = 0; s1s < m; s1s++)
    for (int s2s = 0; s2s < n; s2s++)
      sum += arr[s1s][s2s];
  for (int s1s = 0; s1s < m; s1s++)
    for (int s2s = 0; s2s < n; s2s++)
      arr[s1s][s2s] *= (1/sum);
}

void SetSumUnity1D(double* arr, int m) {
  double sum = 0;
  for (int s1s = 0; s1s < m; s1s++)
    sum += arr[s1s];
  for (int s1s = 0; s1s < m; s1s++)
    arr[s1s] *= (1/sum);
}

void MMFTriState(int M, double*** a_i, Matrix* tran, 
		 double tol, int max_iteration) {
  double** tr, **af_dtr, **dtr, **tmp_tr;
  double* p, *af_dp, *dp, *tmp_p;
  double old_error = 1;

  double*** f_i;

  SetSumUnity3D(a_i, M, M, M);
  f_i = Create3DArray(M, M, M);
 
  tr = Create2DArray(M, M); tmp_tr = Create2DArray(M, M);
  p = Create1DArray(M); tmp_p = Create1DArray(M);
  dtr = Create2DArray(M, M); af_dtr = Create2DArray(M, M);
  dp = Create1DArray(M); af_dp = Create1DArray(M);


  SetRandom2D(tr, M, M); 
  for (int s = 0; s < M; s++)
    SetSumUnity1D(tr[s], M);
  SetRandom1D(p, M); SetSumUnity1D(p, M);

  for (int iter = 0; iter < max_iteration; iter ++) {
    double error = 0;
    for (int s1 = 0; s1 < M; s1++)
      for (int s2 = 0; s2 < M; s2++)
	for (int s3 = 0; s3 < M; s3++) {
	  f_i[s1][s2][s3] = p[s1]*tr[s1][s2]*tr[s2][s3];
	  error += a_i[s1][s2][s3] * log (a_i[s1][s2][s3]/f_i[s1][s2][s3]) 
	    - a_i[s1][s2][s3] + f_i[s1][s2][s3];
	}
    //printf("iter=%d error=%f\n",iter,error);
    if (fabs(error-old_error) < tol || fabs((error-old_error)/old_error) < tol)
      break;
    old_error = error;

    SetZero2D(af_dtr, M, M); SetZero2D(dtr, M, M); 
    SetZero1D(af_dp, M); SetZero1D(dp, M); 
    for (int s1 = 0; s1 < M; s1++)
      for (int s2 = 0; s2 < M; s2++)
	for (int s3 = 0; s3 < M; s3++) {
	  double tmp = a_i[s1][s2][s3]/f_i[s1][s2][s3];
	  af_dp[s1] += tmp*tr[s1][s2]*tr[s2][s3];
	  dp[s1] += tr[s1][s2]*tr[s2][s3];

	  af_dtr[s1][s2] += tmp*p[s1]*tr[s2][s3];
	  dtr[s1][s2] += p[s1]*tr[s2][s3];
	  af_dtr[s2][s3] += tmp*p[s1]*tr[s1][s2];
	  dtr[s2][s3] += p[s1]*tr[s1][s2];
	}

    if (iter % 2) {
      for (int s1 = 0; s1 < M; s1++)
	for (int s2 = 0; s2 < M; s2++) {
	  double nume = af_dtr[s1][s2], deno = dtr[s1][s2];
	  for (int s = 0; s < M; s++) {
	    nume += tr[s1][s]*dtr[s1][s];
	    deno += tr[s1][s]*af_dtr[s1][s];
	  }
	  tmp_tr[s1][s2] = tr[s1][s2]*nume/deno;
	}
      for (int s1 = 0; s1 < M; s1++)
	for (int s2 = 0; s2 < M; s2++) tr[s1][s2] = tmp_tr[s1][s2];
    }
    else {
      for (int s = 0; s < M; s++) {
	double nume = af_dp[s], deno = dp[s];
	for (int ss = 0; ss < M; ss++) {
	  nume += p[ss]*dp[ss];
	  deno += p[ss]*af_dp[ss];
	}
	tmp_p[s] = p[s]*nume/deno;
      }
      for (int s = 0; s < M; s++) p[s] = tmp_p[s];
    }
  }
  
  for (int s1 = 0; s1 < M; s1++)
    for (int s2 = 0; s2 < M; s2++) {
      tran->ref(s1,s2) = tr[s1][s2];
      printf("%f\n", tr[s1][s2]);
    }
  for (int s = 0; s < M; s++)
    printf("p[%d] = %f\n", s, p[s]);
}

void DiscreteHMM::TrainMMF(const ArrayList<Vector>& list_data_seq, 
			   double rho, int max_iteration, double tolerance) {
  TriValMap a_i;
  double*** tri, ***af_dtri, ***dtri, ***tmp_tri, ***min_tri;
  double** so, **af_dso, **dso, **tmp_so, **min_so;
  
  CalObsFreq(list_data_seq, &a_i);
  TriValMap f_i(a_i);

  int M = transmission_.n_rows();
  int N = emission_.n_cols();

  tri = Create3DArray(M,M,M);
  af_dtri = Create3DArray(M,M,M);
  dtri = Create3DArray(M,M,M);
  tmp_tri = Create3DArray(M,M,M);
  min_tri = Create3DArray(M,M,M);
  
  so = Create2DArray(M, N);
  af_dso = Create2DArray(M, N);
  dso = Create2DArray(M, N);
  tmp_so = Create2DArray(M, N);
  min_so = Create2DArray(M, N);
  
  double min_error = 1e10;
  double old_min = 1;
  for (int rand_iter = 0; rand_iter < 500; rand_iter++) {
    SetRandom3D(tri, M, M, M); SetSumUnity3D(tri, M, M, M);
    SetRandom2D(so, M, N);
    for (int s = 0; s < M; s++) SetSumUnity1D(so[s], N);

    double old_error = 1;
    for (int iter = 0; iter < max_iteration; iter ++) {
      TriValMap::iterator it, ait;
      for (it = f_i.begin(); it != f_i.end(); it++) {
	TriValues obs = it->first;
	it->second = 0;
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) 
	      it->second += so[s1][obs.v1]*so[s2][obs.v2]*so[s3][obs.v3]
		*tri[s1][s2][s3];
      }

      SetZero3D(af_dtri, M, M, M); SetZero3D(dtri, M, M, M);
      SetZero2D(af_dso, M, N); SetZero2D(dso, M, N);
      double error = 0;
      for (it = f_i.begin(), ait = a_i.begin(); it != f_i.end(); it++, ait++) {
	TriValues obs = it->first;
	double af = ait->second / it->second;
	error += ait->second*log(ait->second/it->second)-ait->second+it->second;
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) {
	      af_dtri[s1][s2][s3] += af*so[s1][obs.v1]*so[s2][obs.v2]
		*so[s3][obs.v3];
	      dtri[s1][s2][s3] += so[s1][obs.v1]*so[s2][obs.v2]*so[s3][obs.v3];

	      af_dso[s1][obs.v1] += af*so[s2][obs.v2]*so[s3][obs.v3]
		*tri[s1][s2][s3];
	      dso[s1][obs.v1] += so[s2][obs.v2]*so[s3][obs.v3]
		*tri[s1][s2][s3];

	      af_dso[s2][obs.v2] += af*so[s1][obs.v1]*so[s3][obs.v3]
		*tri[s1][s2][s3];
	      dso[s2][obs.v2] += so[s1][obs.v1]*so[s3][obs.v3]
		*tri[s1][s2][s3];

	      af_dso[s3][obs.v3] += af*so[s2][obs.v2]*so[s1][obs.v1]
		*tri[s1][s2][s3];
	      dso[s3][obs.v3] += so[s2][obs.v2]*so[s1][obs.v1]
		*tri[s1][s2][s3];
	    }
      }
      //printf("iter = %5d error = %f\n", iter, error);
      if (fabs(error-old_error) < tolerance 
	  || fabs((error-old_error)/old_error) < tolerance) {
	old_error = error;
	break;
      }
      old_error = error;

      if (iter % 2) {
	double nume_cm = 0, deno_cm = 0;
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) {
	      nume_cm += tri[s1][s2][s3]*dtri[s1][s2][s3];
	      deno_cm += tri[s1][s2][s3]*af_dtri[s1][s3][s3];
	    }
	    
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) {
	      double nume = af_dtri[s1][s2][s3]+nume_cm;
	      double deno = dtri[s1][s2][s3]+deno_cm;
	      tmp_tri[s1][s2][s3] = tri[s1][s2][s3]*nume/deno;
	    }
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) tri[s1][s2][s3]=tmp_tri[s1][s2][s3];

      }
      else {
	for (int s = 0; s < M; s++) {
	  double nume_cm = 0, deno_cm = 0;
	  for (int v = 0; v < N; v++) {
	    nume_cm += so[s][v]*dso[s][v];
	    deno_cm += so[s][v]*af_dso[s][v];
	  }
	  for (int v = 0; v < N; v++) {
	    double nume = af_dso[s][v]+nume_cm;
	    double deno = dso[s][v]+deno_cm;
	    tmp_so[s][v] = so[s][v]*nume/deno;
	  }	
	  for (int v = 0; v < N; v++) so[s][v] = tmp_so[s][v];
	}
      }

    }
    if (min_error > old_error) {
      min_error = old_error;
      Copy3D(tri, min_tri, M, M, M);
      Copy2D(so, min_so, M, N);

      //if (fabs(min_error-old_min) < tolerance 
      //  || fabs((min_error-old_min)/old_min) < tolerance) break;
      old_min = min_error;
    }
    printf("rand_iter=%d min_error=%f\n", rand_iter, min_error);

  }

  for (int s = 0; s < M; s++)
    for (int v = 0; v < N; v++) emission_.ref(s, v) = min_so[s][v];

  MMFTriState(M, min_tri, &transmission_, tolerance, max_iteration);
}

const fx_entry_doc mmf_main_entries[] = {
  {"seqfile", FX_REQUIRED, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  //  {"guess", FX_PARAM, FX_STR, NULL,
  //   "  File containing guessing HMM model profile.\n"},
  {"numstate", FX_REQUIRED, FX_INT, NULL,
   "  If no guessing profile specified, at least provide the"
   " number of states.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  Output file containing trained HMM profile.\n"},
  {"rho", FX_PARAM, FX_DOUBLE, NULL,
   "  Regularization parameter, default = 1.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance on log-likelihood as a stopping criteria.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc mmf_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc mmf_main_doc = {
  mmf_main_entries, mmf_main_submodules,
  "This is a program training HMM models from data sequences using MMF. \n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &mmf_main_doc);
  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str_req(NULL, "profile");

  srand(time(NULL));
  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  int numstate = fx_param_int_req(NULL, "numstate");
  printf("Randomly generate parameters: NUMSTATE = %d\n", numstate);

  double rho = fx_param_double(NULL, "rho", 1);
  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  DiscreteHMM hmm;
  hmm.InitFromData(seqs, numstate);

  hmm.TrainMMF(seqs, rho, maxiter, tol);

  hmm.SaveProfile(proout);
  fx_done(NULL);
}
