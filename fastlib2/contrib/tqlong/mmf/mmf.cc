
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
    printf("iter=%d error=%f\n",iter,error);
    if (fabs(error-old_error) < tol || fabs((error-old_error)/old_error) < tol)
      break;
    old_error = error;
    /*
      if (error < min_error) {
      for (int s1 = 0; s1 < M; s1++)
      for (int s2 = 0; s2 < M; s2++)
      min_tr[s1][s2] = tr[s1][s2];
      min_error = error;
      }
    */

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

void MMFTriState1(int M, double*** tri, Matrix* tran, 
		 double rho, int max_iteration) {
  double** tr;
  double* p;

  double sum = 0;
  for (int s1s = 0; s1s < M; s1s++)
    for (int s2s = 0; s2s < M; s2s++)
      for (int s3s = 0; s3s < M; s3s++) 
	sum += tri[s1s][s2s][s3s];
  for (int s1s = 0; s1s < M; s1s++)
    for (int s2s = 0; s2s < M; s2s++)
      for (int s3s = 0; s3s < M; s3s++) 
	tri[s1s][s2s][s3s] *= (1/sum);
 

  tr = Create2DArray(M, M);
  p = Create1DArray(M);
  SetRandom2D(tr, M, M);
  SetRandom1D(p, M);

  for (int  iter = 0; iter < max_iteration; iter ++) {
    for (int s1 = 0; s1 < M; s1++)
      for (int s2 = 0; s2 < M; s2++) {
	double deno = 0, nume = 0;

	for (int s1s = 0; s1s < M; s1s++)
	  for (int s2s = 0; s2s < M; s2s++)
	    for (int s3s = 0; s3s < M; s3s++) {
	      double a_i = tri[s1s][s2s][s3s];
	      double f_i = p[s1s]*tr[s1s][s2s]*tr[s2s][s3s];
	      if (s1s==s1 && s2s==s2) {
		deno += p[s1s]*tr[s2s][s3s];
		nume += a_i/f_i*p[s1s]*tr[s2s][s3s];
	      }
	      if (s2s==s1 && s3s==s2) {
		deno += p[s1s]*tr[s1s][s2s];
		nume += a_i/f_i*p[s1s]*tr[s1s][s2s];
	      }
	    }
	
	double sum = 0;
	for (int s2s = 0; s2s < M; s2s++) sum += tr[s1][s2s];
	tr[s1][s2] *= (nume+rho) / (deno + rho*sum);
      }
    
    for (int s = 0; s < M; s++) {
      double deno = 0, nume = 0;
      for (int s2s = 0; s2s < M; s2s++)
	for (int s3s = 0; s3s < M; s3s++) {
	  double a_i = tri[s][s2s][s3s];
	  double f_i = p[s]*tr[s][s2s]*tr[s2s][s3s];
	  deno += tr[s][s2s]*tr[s2s][s3s];
	  nume += a_i/f_i*tr[s][s2s]*tr[s2s][s3s];
	}
      double sum = 0;
      for (int ss = 0; ss < M; ss++) sum += p[ss];
      p[s] *= (nume+rho) / (deno + rho*sum);
    }

    double error = 0;
    for (int s1s = 0; s1s < M; s1s++)
      for (int s2s = 0; s2s < M; s2s++)
	for (int s3s = 0; s3s < M; s3s++) {
	  double a_i = tri[s1s][s2s][s3s];
	  double f_i = p[s1s]*tr[s1s][s2s]*tr[s2s][s3s];
	  error += a_i*log(a_i/f_i)-a_i+f_i;
	}
    printf("error = %f\n", error);
  }

  for (int s1s = 0; s1s < M; s1s++)
    for (int s2s = 0; s2s < M; s2s++)
      for (int s3s = 0; s3s < M; s3s++) {
	double a_i = tri[s1s][s2s][s3s];
	double f_i = p[s1s]*tr[s1s][s2s]*tr[s2s][s3s];
	printf("a_i = %f f_i = %f\n", a_i, f_i);
      }

  for (int s1 = 0; s1 < M; s1++)
    for (int s2 = 0; s2 < M; s2++)
      tran->ref(s1,s2) = tr[s1][s2];
  for (int s = 0; s < M; s++)
    printf("p[%d] = %f\n", s, p[s]);
}

void DiscreteHMM::TrainMMF(const ArrayList<Vector>& list_data_seq, 
			   double rho, int max_iteration, double tolerance) {
  TriValMap obsmap;
  double*** triState;
  double*** triStateDeno;
  double*** triStateNume;
  double** stateObs;
  double** stateObsDeno;
  double** stateObsNume;
  
  CalObsFreq(list_data_seq, &obsmap);

  int M = transmission_.n_rows();
  int N = emission_.n_cols();
  triState = Create3DArray(M,M,M);
  triStateDeno = Create3DArray(M, M, M);
  triStateNume = Create3DArray(M, M, M);
  stateObs = Create2DArray(M, N);
  stateObsDeno = Create2DArray(M, N);
  stateObsNume = Create2DArray(M, N);

  SetRandom3D(triState, M, M, M);
  SetRandom2D(stateObs, M, N);
  
  for (int iter = 0; iter < max_iteration; iter ++) {
    /*
    SetZero3D(triStateDeno, M, M, M);
    SetZero3D(triStateNume, M, M, M);
    SetZero2D(stateObsDeno, M, M);
    SetZero2D(stateObsNume, M, M);
    double error = 0;
    for (TriValMap::iterator it = obsmap.begin(); it != obsmap.end(); it++) {
      TriValues obs = it->first;
      double a_i = it->second;
      double f_i = 0;

      for (int s1 = 0; s1 < M; s1++)
	for (int s2 = 0; s2 < M; s2++)
	  for (int s3 = 0; s3 < M; s3++)
	    f_i += stateObs[s1][obs.v1]*stateObs[s2][obs.v2]
	      *stateObs[s3][obs.v3]*triState[s1][s2][s3];
      //printf("f_i = %f a_i = %f  ", f_i, a_i);

      error += a_i * log(a_i/f_i) - a_i + f_i;

      for (int s1 = 0; s1 < M; s1++)
	for (int s2 = 0; s2 < M; s2++)
	  for (int s3 = 0; s3 < M; s3++) {
	    triStateNume[s1][s2][s3] += (a_i/f_i*stateObs[s1][obs.v1]
	      *stateObs[s2][obs.v2]*stateObs[s3][obs.v3]);
	    triStateDeno[s1][s2][s3] += (stateObs[s1][obs.v1]
	      *stateObs[s2][obs.v2]*stateObs[s3][obs.v3]);

	    stateObsNume[s1][obs.v1] += (a_i/f_i*stateObs[s2][obs.v2]
	      *stateObs[s3][obs.v3]*triState[s1][s2][s3]);
	    stateObsDeno[s1][obs.v1] += (stateObs[s2][obs.v2]
	      *stateObs[s3][obs.v3]*triState[s1][s2][s3]);

	    stateObsNume[s2][obs.v2] += (a_i/f_i*stateObs[s1][obs.v1]
	      *stateObs[s3][obs.v3]*triState[s1][s2][s3]);
	    stateObsDeno[s2][obs.v2] += (stateObs[s1][obs.v1]
	      *stateObs[s3][obs.v3]*triState[s1][s2][s3]);

	    stateObsNume[s3][obs.v3] += (a_i/f_i*stateObs[s2][obs.v2]
	      *stateObs[s1][obs.v1]*triState[s1][s2][s3]);
	    stateObsDeno[s3][obs.v3] += (stateObs[s2][obs.v2]
	      *stateObs[s1][obs.v1]*triState[s1][s2][s3]);
	  }
    }
    printf("\niter = %d error = %f\n", iter, error);

    if (iter % 2 == 0)
      for (int s1 = 0; s1 < M; s1++)
	for (int s2 = 0; s2 < M; s2++)
	  for (int s3 = 0; s3 < M; s3++) {
	    triState[s1][s2][s3] *= (triStateNume[s1][s2][s3]
				     /triStateDeno[s1][s2][s3]);
	    //printf("triState = %f\n", triState[s1][s2][s3]);
	  }
    else
      for (int s = 0; s < M; s++)
	for (int v = 0; v < N; v++)
	  stateObs[s][v] *= (stateObsNume[s][v]/stateObsDeno[s][v]);
    */
      for (int s1 = 0; s1 < M; s1++)
	for (int s2 = 0; s2 < M; s2++)
	  for (int s3 = 0; s3 < M; s3++) {
	    double deno = 0, nume = 0;
	    for (TriValMap::iterator it = obsmap.begin(); 
		 it != obsmap.end(); it++) {
	      TriValues obs = it->first;
	      double a_i = it->second;
	      double f_i = 0;
	      for (int ss1 = 0; ss1 < M; ss1++)
		for (int ss2 = 0; ss2 < M; ss2++)
		  for (int ss3 = 0; ss3 < M; ss3++)
		    f_i += stateObs[ss1][obs.v1]*stateObs[ss2][obs.v2]
		      *stateObs[ss3][obs.v3]*triState[ss1][ss2][ss3];
	      double tmp = stateObs[s1][obs.v1]
		*stateObs[s2][obs.v2]*stateObs[s3][obs.v3];
	      nume += a_i/f_i*tmp;
	      deno += tmp;
	    }
	    double sum = 0;
	    for (int ss1 = 0; ss1 < M; ss1++)
	      for (int ss2 = 0; ss2 < M; ss2++)
		for (int ss3 = 0; ss3 < M; ss3++)
		  sum += triState[ss1][ss2][ss3];
	    
	    triState[s1][s2][s3] *= (nume+rho) / (deno + rho*sum);
	  }
      for (int s = 0; s < M; s++)
	for (int v = 0; v < N; v++) {
	  double deno = 0, nume = 0;
	  for (TriValMap::iterator it = obsmap.begin(); 
	       it != obsmap.end(); it++) {
	    TriValues obs = it->first;
	    if (obs.v1 != v && obs.v2 != v && obs.v3 != v) continue;
	    double a_i = it->second;
	    double f_i = 0;
	    for (int s1 = 0; s1 < M; s1++)
	      for (int s2 = 0; s2 < M; s2++)
		for (int s3 = 0; s3 < M; s3++)
		  f_i += stateObs[s1][obs.v1]*stateObs[s2][obs.v2]
		    *stateObs[s3][obs.v3]*triState[s1][s2][s3];
	    if (obs.v1 == v) 
	      for (int s2 = 0; s2 < M; s2++)
		for (int s3 = 0; s3 < M; s3++) {
		  double tmp = stateObs[s2][obs.v2]
		    *stateObs[s3][obs.v3]*triState[s][s2][s3];
		  nume += a_i/f_i*tmp;
		  deno += tmp;
		}

	    if (obs.v2 == v) 
	      for (int s1 = 0; s1 < M; s1++)
		for (int s3 = 0; s3 < M; s3++) {
		  double tmp = stateObs[s1][obs.v1]
		    *stateObs[s3][obs.v3]*triState[s1][s][s3];
		  nume += a_i/f_i*tmp;
		  deno += tmp;
		}

	    if (obs.v3 == v) 
	      for (int s1 = 0; s1 < M; s1++)
		for (int s2 = 0; s2 < M; s2++) {
		  double tmp = stateObs[s1][obs.v1]
		    *stateObs[s2][obs.v2]*triState[s1][s2][s];
		  nume += a_i/f_i*tmp;
		  deno += tmp;
		}
	  }
	  double sum = 0;
	  for (int vv = 0; vv < N; vv++)
	    sum += stateObs[s][vv];
	  stateObs[s][v] *= (nume+rho) / (deno+rho*sum);
	}
      /*
      double error = 0;
      for (TriValMap::iterator it = obsmap.begin(); it != obsmap.end(); it++) {
	TriValues obs = it->first;
	double a_i = it->second;
	double f_i = 0;

	for (int s1 = 0; s1 < M; s1++)
	  for (int s2 = 0; s2 < M; s2++)
	    for (int s3 = 0; s3 < M; s3++)
	      f_i += stateObs[s1][obs.v1]*stateObs[s2][obs.v2]
		*stateObs[s3][obs.v3]*triState[s1][s2][s3];
	//printf("f_i = %f a_i = %f  ", f_i, a_i);

	error += a_i * log(a_i/f_i) - a_i + f_i;
      }
      
      printf("iter = %5d error = %f\n", iter, error);
      */
  }

  /*
  for (TriValMap::iterator it = obsmap.begin(); it != obsmap.end(); it++) {
    TriValues obs = it->first;
    double a_i = it->second;
    double f_i = 0;
    for (int s1 = 0; s1 < M; s1++)
      for (int s2 = 0; s2 < M; s2++)
	for (int s3 = 0; s3 < M; s3++)
	  f_i += stateObs[s1][obs.v1]*stateObs[s2][obs.v2]
	    *stateObs[s3][obs.v3]*triState[s1][s2][s3];

    printf("a_i = %f f_i = %f\n", a_i, f_i);
  }
  */

  for (int s = 0; s < M; s++)
    for (int v = 0; v < N; v++) 
      emission_.ref(s, v) = stateObs[s][v];
  /*  
  for (int s1 = 0; s1 < M; s1++) {
    double sums1 = 0;
    for (int s2 = 0; s2 < M; s2++)
      for (int s3 = 0; s3 < M; s3++)
	sums1 += triState[s1][s2][s3];
    for (int s2 = 0; s2 < M; s2++) {
      double sums2 = 0;
      for (int s3 = 0; s3 < M; s3++)
	sums2 += triState[s1][s2][s3];
      transmission_.ref(s1, s2) = sums2 / sums1;
    }
  }
  */

  MMFTriState(M, triState, &transmission_, tolerance, max_iteration);
  /*
  Vector sums1;
  sums1.Init(M); 
  sums1.SetZero();
  transmission_.SetZero();
  for (int s1 = 0; s1 < M; s1++)
    for (int s2 = 0; s2 < M; s2++)
      for (int s3 = 0; s3 < M; s3++) {
	double tmp = triState[s1][s2][s3];
	printf("s1=%d s2=%d s3=%d triState=%f\n", 
	       s1,s2,s3,tmp);
	transmission_.ref(s1, s2) += tmp;
	if (s1 != s2 || s2 != s3)
	  transmission_.ref(s2, s3) += tmp;
	sums1[s1] += tmp; 
	if (s1 != s2)
	  sums1[s2] += tmp; 
      }
  for (int s1 = 0; s1 < M; s1++) {
    printf("-- s1=%d sum=%f\n", s1, sums1[s1]);
    for (int s2 = 0; s2 < M; s2++) {
      printf("-- s1=%d s2=%d tran=%f\n", s1,s2,transmission_.get(s1,s2));
      transmission_.ref(s1, s2) /= sums1[s1];
    }
  }
  */
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
