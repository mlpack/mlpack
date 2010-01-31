
#include <fastlib/fastlib.h>
#include <list>
#include "svm_projection.h"

namespace la {
};

namespace SVM_Projection {
  void MulOverwrite(const Matrix& A, const Vector& x, Vector* y, 
		    const SV_index& SVs) {
    DEBUG_ASSERT(A.n_rows() == x.length() &&
		 A.n_rows() == y->length());
    for (index_t i = 0; i < A.n_rows(); i++) {
      (*y)[i] = 0.0;
      for (SV_index::const_iterator it = SVs.begin(); it != SVs.end(); it++)
	(*y)[i] += A.get(i, *it) * x[*it];
    }
  }

  void OptimizeQuadraticOnSimplex(const Matrix& Q, Vector* x, SV_index* SVs) {
    DEBUG_ASSERT(Q.n_rows() == Q.n_cols());
    index_t n = Q.n_rows();
    Vector& _x = *x;
    Vector a, w, dx;
    SV_index& _SVs = *SVs;

    _x.Init(n); //_x.SetAll(1.0/n); 
    _x.SetZero(); _x[0] = 1;
    a.Init(n); w.Init(n); dx.Init(n);
    _SVs.clear(); _SVs.push_back(0);
    //for (index_t i = 0; i < n; i++) _SVs.push_back(i);
    
    double L = 1;
    double change = 1;
    index_t iter = 0;
    double tol = 1e-6;
    //printf("^^^^^\n");
    //ot::Print(_x);
    while (/*iter < 2000 && */change > tol) {
      iter++;

      MulOverwrite(Q, _x, &a, _SVs);
      
      double L1 = la::LengthEuclidean(a) / la::LengthEuclidean(_x);
      if (L < L1) L = L1;
      printf("L=%f\n",L);

      dx.CopyValues(_x);
      la::AddExpert(-1.0/L, a, &_x);
      
      //printf("*****\n");
      //ot::Print(_x);
      ProjectOnSimplex(_x, &w, &_SVs);
      _x.CopyValues(w);

      la::SubFrom(_x, &dx);
      change = la::LengthEuclidean(dx);

      if (fx_param_exists(NULL, "verbose"))
	printf("iter = %d change = %g SVs = %ld\n", iter, change, _SVs.size());
      //printf("^^^^^\n");
      //ot::Print(_x);
    }
    printf("Summary: iter = %d change = %g SVs = %ld L = %f\n", iter, change, _SVs.size(), L);    
  }

  void OptimQuadratic(const Matrix& Q, Vector* x, SV_index* SVs) {
    DEBUG_ASSERT(Q.n_rows() == Q.n_cols());
    index_t n = Q.n_rows();
    Vector& _x = *x;
    //Vector a, w, dx;
    SV_index& _SVs = *SVs;
    ArrayList<char> bSV;
    
    bSV.Init(n);
    for (int i = 0; i < n; i++) bSV[i] = 0;

    _x.Init(n); //_x.SetAll(1.0/n); 
    _x.SetZero(); _x[0] = 1;
    //a.Init(n); w.Init(n); dx.Init(n);
    _SVs.clear(); _SVs.push_back(0);
    bSV[0] = 1;

    int o_idx = 0; int o_inc = 1;
    index_t n_s = fx_param_int(NULL, "ns", 2);
    Vector c, y;
    c.Init(n_s); y.Init(n_s);
    SV_index SVs_y;
    //double gamma = 1.0;
    double s_change = 0;
    double tol = 1e-6;
    int iter = 0;
    int round = 0;
    double gamma = 1;
    while (true) {
      iter++;
      if (o_inc*(n_s-1)+1 > n) { 
	o_idx = 0; o_inc = 1; 
	round++;
	if (s_change < tol) break;
	s_change = 0;
      }
      // choose a subset of variables
      ArrayList<index_t> idx; idx.Init();
      bool process = false;
      for (index_t i = 0; i < n_s; i++) {
	index_t j = (o_idx+i*o_inc)%n;
	idx.PushBackCopy(j);
	if (bSV[i] != 0) process = true;
      }
      o_idx += 1;
      if (o_idx >= n) { 
	o_idx = 0; o_inc += 1; 
      }
      if (!process) continue;
      
      //printf("("); for (int i = 0; i < n_s; i++) printf("%d,", idx[i]); printf(")=");
      //for (int i = 0; i < n_s; i++) printf("%f,",_x[idx[i]]); printf("\n");
      // compute the sum of variables (need to be fixed)
      double alpha = 0;
      for (index_t i = 0; i < n_s; i++) alpha += _x[idx[i]];
      if (alpha <= 1e-12) continue;
      
      // compute gamma
      double g = 0;
      double s = 0;
      for (index_t i = 0; i < n_s; i++) {
	double t = 0;
	for (index_t j = 0; j < n_s; j++)
	  t += Q.get(idx[i], idx[j])* _x[idx[i]];
	g += t*t;
	s += _x[idx[i]]*_x[idx[i]];
      }
      //printf("gamma=%f, s = %f\n", gamma, s);
      if (sqrt(s) > 1e-12) g = sqrt(g)/sqrt(s);
      else g = 1;
      if (gamma < g) gamma  = g;
      if (gamma < 1) gamma = 1;
      //printf("alpha = %f, gamma=%f\n", alpha,gamma);

      // compute c: point to be projected
      for (index_t i = 0; i < n_s; i++) {
	index_t ii = idx[i];
	c[i] = 0;
	for (SV_index::iterator it = _SVs.begin(); it != _SVs.end(); it++)
	  c[i] += Q.get(ii, (*it)) * _x[*it];
	c[i] = _x[ii] - c[i]/gamma;
      }

      // project
      la::Scale(1.0/alpha, &c);
      //printf("c="); for (int i = 0; i < n_s; i++) printf("%f,",c[i]); printf("\n");
      ProjectOnSimplex(c, &y, &SVs_y);

      // collect result
      double change = 0;
      for (index_t i = 0; i < n_s; i++) {
	index_t ii = idx[i];
	change += (_x[ii] - alpha * y[i]) * (_x[ii] - alpha * y[i]);
	_x[ii] = alpha * y[i];
	bSV[ii] = 0;
      }
      s_change += sqrt(change);

      for (SV_index::iterator it = SVs_y.begin(); it != SVs_y.end(); it++)
	bSV[idx[*it]] = 2;

      // sync SVs
      SV_index::iterator it = _SVs.begin();
      do {
	index_t i = *it;
	if (bSV[i] == 0) // not a SV anymore
	  it = _SVs.erase(it); 
	else {
	  if (bSV[i] == 2) bSV[i] = 1; // old SV remains SV
	  it++;
	}
      } while (it != _SVs.end());
      
      for (index_t i = 0; i < n_s; i++) {
	index_t ii = idx[i];
	if (bSV[ii] == 2) { // add new SV
	  _SVs.push_back(ii);
	  bSV[ii] = 1;
	}
      }

      //char ch = getc(stdin);
      //ot::Print(_x);
    }
    printf("Summary: iter = %d round = %d change = %f SVs = %ld L=%f \n", 
	   iter, round, s_change, _SVs.size(), gamma);
  }
};
