
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

void Print2D(double** arr, int m, int n) {
  printf("MATRIX :\n");
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) printf("%10.4f", arr[i][j]);
    printf("\n");
  }
}

void Print1D(double* arr, int m) {
  printf("VECTOR :\n");
  for (int i = 0; i < m; i++) {
    printf("%10.4f", arr[i]);
  }
  printf("\n");
}

void Copy1D(double* src, double* dst, int m) {
  for (int i = 0; i < m; i++) dst[i] = src[i];
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

void DiscreteHMM::TrainMMF(const ArrayList<Vector>& list_data_seq, 
			   int maxrand, int max_iteration, double tolerance) {
  TriValMap a_i;
  //double*** tri, ***af_dtri, ***dtri, ***tmp_tri, ***min_tri;
  double** so, **af_dso, **dso, **tmp_so, **min_so;
  double** tr, **af_dtr, **dtr, **tmp_tr, **min_tr;
  double *p, *af_dp, *dp, *tmp_p, *min_p;
  
  CalObsFreq(list_data_seq, &a_i);
  TriValMap f_i(a_i);

  int M = transmission_.n_rows();
  int N = emission_.n_cols();

  //printf("M = %d N = %d\n", M, N); //commented out by Nishant for speed

  //tri = Create3DArray(M,M,M);
  //af_dtri = Create3DArray(M,M,M);
  //dtri = Create3DArray(M,M,M);
  //tmp_tri = Create3DArray(M,M,M);
  //min_tri = Create3DArray(M,M,M);
  
  so = Create2DArray(M, N);
  af_dso = Create2DArray(M, N);
  dso = Create2DArray(M, N);
  tmp_so = Create2DArray(M, N);
  min_so = Create2DArray(M, N);

  tr = Create2DArray(M, M);
  af_dtr = Create2DArray(M, M);
  dtr = Create2DArray(M, M);
  tmp_tr = Create2DArray(M, M);
  min_tr = Create2DArray(M, M);

  p = Create1DArray(M);
  af_dp = Create1DArray(M);
  dp = Create1DArray(M);
  tmp_p = Create1DArray(M);
  min_p = Create1DArray(M);
  
  double min_error = 1e10;
  double old_min = 1;
  for (int rand_iter = 0; rand_iter < maxrand; rand_iter++) {
    //SetRandom3D(tri, M, M, M); SetSumUnity3D(tri, M, M, M);
    SetRandom2D(so, M, N);
    for (int s = 0; s < M; s++) SetSumUnity1D(so[s], N);
    SetRandom2D(tr, M, M);
    for (int s = 0; s < M; s++) SetSumUnity1D(tr[s], M);
    SetRandom1D(p, M); SetSumUnity1D(p, M);

    //Print2D(tr, M, M);
    //Print2D(so, M, N);
    //Print1D(p, M);

    double old_error = 1;
    for (int iter = 0; iter < max_iteration; iter ++) {
      TriValMap::iterator it, ait;
      for (it = f_i.begin(); it != f_i.end(); it++) {
	TriValues obs = it->first;
	it->second = 0;
	//if (obs.v1 >= N || obs.v2 >= N || obs.v3 >= N)
	//  printf("--- WARNING ---\n");
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) 
	      it->second += so[s1][obs.v1]*so[s2][obs.v2]*so[s3][obs.v3]
		*p[s1]*tr[s1][s2]*tr[s2][s3];//tri[s1][s2][s3];
	//if (it->second < 1e-10)
	//printf("\n----- PRE WARNING ----- %d %d %d\n", obs.v1, obs.v2, obs.v3);
      }
      //printf("TEST1\n");
      // SetZero3D(af_dtri, M, M, M); SetZero3D(dtri, M, M, M);
      SetZero2D(af_dso, M, N); SetZero2D(dso, M, N);
      SetZero2D(af_dtr, M, M); SetZero2D(dtr, M, M);
      SetZero1D(af_dp, M); SetZero1D(dp, M);
      //printf("TEST2\n");
      double error = 0;
      for (it = f_i.begin(), ait = a_i.begin(); it != f_i.end(); it++, ait++) {
	//printf("---- error = %f", error);
	TriValues obs = it->first;
	//if (it->second < 1e-10)
	  //printf("\n----- WARNING ----- %d %d %d\n", obs.v1, obs.v2, obs.v3);
	//printf("---- error = %f", error);
	double af = ait->second / it->second;
	error += ait->second*log(af)-ait->second+it->second;
	//printf("---- error = %f af = %f", error, af);
	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++) {
	      //af_dtri[s1][s2][s3] += af*so[s1][obs.v1]*so[s2][obs.v2]
	      //*so[s3][obs.v3];
	      //dtri[s1][s2][s3] += so[s1][obs.v1]*so[s2][obs.v2]*so[s3][obs.v3];
	      
	      double con_so = so[s1][obs.v1]*so[s2][obs.v2]*so[s3][obs.v3];
	      double con_tr = tr[s1][s2]*tr[s2][s3];
	      double con_p  = p[s1];

	      af_dso[s1][obs.v1] += af*so[s2][obs.v2]*so[s3][obs.v3]
		*con_p*con_tr;
	      dso[s1][obs.v1] += so[s2][obs.v2]*so[s3][obs.v3]
		*con_p*con_tr;

	      af_dso[s2][obs.v2] += af*so[s1][obs.v1]*so[s3][obs.v3]
		*con_p*con_tr;
	      dso[s2][obs.v2] += so[s1][obs.v1]*so[s3][obs.v3]
		*con_p*con_tr;

	      af_dso[s3][obs.v3] += af*so[s2][obs.v2]*so[s1][obs.v1]
		*con_p*con_tr;
	      dso[s3][obs.v3] += so[s2][obs.v2]*so[s1][obs.v1]
		*con_p*con_tr;

	      af_dp[s1] += af*con_so*con_tr;
	      dp[s1] += con_so*con_tr;

	      af_dtr[s1][s2] += af*tr[s2][s3]*con_so*con_p;
	      dtr[s1][s2] += tr[s2][s3]*con_so*con_p;

	      af_dtr[s2][s3] += af*tr[s1][s2]*con_so*con_p;
	      dtr[s2][s3] += tr[s1][s2]*con_so*con_p;
	    }
      }
      //printf("--iter = %5d error = %f\n", iter, error);
      if (fabs(error-old_error) < tolerance 
	  || fabs((error-old_error)/old_error) < tolerance) {
	old_error = error;
	break;
      }
      old_error = error;

      if (iter % 3==0) {
	//double nume_cm = 0, deno_cm = 0;
	//for (int s1 = 0; s1 < M; s1++)
	//  for (int s2 = 0; s2 < M; s2++)
	//    for (int s3 = 0; s3 < M; s3++) {
	//      nume_cm += tri[s1][s2][s3]*dtri[s1][s2][s3];
	//      deno_cm += tri[s1][s2][s3]*af_dtri[s1][s3][s3];
	//    }
	    
	//for (int s1 = 0; s1 < M; s1++)
	//  for (int s2 = 0; s2 < M; s2++)
	//    for (int s3 = 0; s3 < M; s3++) {
	//      double nume = af_dtri[s1][s2][s3]+nume_cm;
	//      double deno = dtri[s1][s2][s3]+deno_cm;
	//      tmp_tri[s1][s2][s3] = tri[s1][s2][s3]*nume/deno;
	//    }
	//for (int s1 = 0; s1 < M; s1++)
	//  for (int s2 = 0; s2 < M; s2++)
	//    for (int s3 = 0; s3 < M; s3++) tri[s1][s2][s3]=tmp_tri[s1][s2][s3];
	for (int s1 = 0; s1 < M; s1++) {
	  double nume_cm = 0, deno_cm = 0;
	  for (int s2 = 0; s2 < M; s2++) {
	    nume_cm += tr[s1][s2]*dtr[s1][s2];
	    deno_cm += tr[s1][s2]*af_dtr[s1][s2];
	  }
	  for (int s2 = 0; s2 < M; s2++)
	    tr[s1][s2] *= (af_dtr[s1][s2]+nume_cm)/(dtr[s1][s2]+deno_cm);
	}
      }
      else if (iter % 3 == 1) {
	for (int s = 0; s < M; s++) {
	  double nume_cm = 0, deno_cm = 0;
	  for (int v = 0; v < N; v++) {
	    nume_cm += so[s][v]*dso[s][v];
	    deno_cm += so[s][v]*af_dso[s][v];
	  }
	  for (int v = 0; v < N; v++) 
	    so[s][v] *= (af_dso[s][v]+nume_cm)/(dso[s][v]+deno_cm);
	}
      }
      else {
	double nume_cm = 0, deno_cm = 0;
	for (int s = 0; s < M; s++) {
	    nume_cm += p[s]*dp[s];
	    deno_cm += p[s]*af_dp[s];
	}
	for (int s = 0; s < M; s++)
	  p[s] *= (af_dp[s]+nume_cm)/(dp[s]+deno_cm);
      }

    }
    if (min_error > old_error) {
      min_error = old_error;
      //Copy3D(tri, min_tri, M, M, M);
      Copy2D(so, min_so, M, N);
      Copy2D(tr, min_tr, M, M);
      Copy1D(p, min_p, M);

      //if (fabs(min_error-old_min) < tolerance 
      //  || fabs((min_error-old_min)/old_min) < tolerance) break;
      old_min = min_error;
    }
    //printf("rand_iter=%d min_error=%f\n", rand_iter, min_error); //commented out by Nishant for speed

  }

  for (int s = 0; s < M; s++)
    for (int v = 0; v < N; v++) emission_.ref(s, v) = min_so[s][v];
  for (int s1 = 0; s1 < M; s1++)
    for (int s2 = 0; s2 < M; s2++)
      transmission_.ref(s1, s2) = min_tr[s1][s2];
  
  //for (int s = 0; s < M; s++) //commented out by Nishant for speed
  //printf("p[%d] = %f\n", s, min_p[s]); //commented out by Nishant for speed

  

  // below delete statements added by Nishant to fix memory leaks
  for(int i = 0; i < M; i++) {
    delete[] so[i];
    delete[] af_dso[i];
    delete[] dso[i];
    delete[] tmp_so[i];
    delete[] min_so[i];
    delete[] tr[i];
    delete[] af_dtr[i];
    delete[] dtr[i];
    delete[] tmp_tr[i];
    delete[] min_tr[i];
  }
    
  delete[] so;
  delete[] af_dso;
  delete[] dso;
  delete[] tmp_so;
  delete[] min_so;
  delete[] tr;
  delete[] af_dtr;
  delete[] dtr;
  delete[] tmp_tr;
  delete[] min_tr;
  
  delete[] p;
  delete[] af_dp;
  delete[] dp;
  delete[] tmp_p;
  delete[] min_p;




  //MMFTriState(M, min_tri, &transmission_, tolerance, max_iteration);
}

const fx_entry_doc mmf_main_entries[] = {
  {"seqfile", FX_REQUIRED, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  //  {"guess", FX_PARAM, FX_STR, NULL,
  //   "  File containing guessing HMM model profile.\n"},
  {"numstate", FX_REQUIRED, FX_INT, NULL,
   "  If no guessing profile specified, at least provide the"
   " number of states.\n"},
  {"numsymbol", FX_REQUIRED, FX_INT, NULL,
   " Number of symbols.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  Output file containing trained HMM profile.\n"},
  {"rho", FX_PARAM, FX_DOUBLE, NULL,
   "  Regularization parameter, default = 1.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance on log-likelihood as a stopping criteria.\n"},
  {"maxrand", FX_PARAM, FX_INT, NULL,
   "  Maximum number of random iterations, default = 60.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc mmf_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc mmf_main_doc = {
  mmf_main_entries, mmf_main_submodules,
  "This is a program training HMM models from data sequences using MMF. \n"
};
/*
int main(int argc, char* argv[]) {
  fx_init(argc, argv, &mmf_main_doc);
  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str_req(NULL, "profile");

  srand(time(NULL));
  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  int numstate = fx_param_int_req(NULL, "numstate");
  int numsymbol = fx_param_int_req(NULL, "numsymbol");
  printf("Randomly generate parameters: NUMSTATE = %d\n", numstate);

  //double rho = fx_param_double(NULL, "rho", 1);
  int maxiter = fx_param_int(NULL, "maxiter", 500);
  int maxrand = fx_param_int(NULL, "maxrand", 60);
  double tol = fx_param_double(NULL, "tolerance", 1e-4);

  DiscreteHMM hmm;
  printf("numstate = %d numsymbol = %d\n", numstate, numsymbol);
  hmm.Init(numstate, numsymbol);

  hmm.TrainMMF(seqs, maxrand, maxiter, tol);

  hmm.SaveProfile(proout);
  fx_done(NULL);
}
*/
