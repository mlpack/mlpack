
#include <fastlib/fastlib.h>
#include "imregister.h"

/* Using Levenberg-Marquardt algorithm */
int projective_register(const Matrix& P, const Matrix& Q, Vector* m_) {
  Vector& m = *m_;
  m.Init(8); 
  m[0] = 1; m[1] = 0; m[2] = 0;
  m[3] = 0; m[4] = 1; m[5] = 0;
  m[6] = 0; m[7] = 0;

  double sigma = 1;
  double lambda = 1;
  index_t MAX_ITER = 200;

  Matrix A; Vector b, de_dm;
  A.Init(8,8); b.Init(8); de_dm.Init(8);
  double old_error = INFINITY, error;
  for (index_t iter = 0; iter < MAX_ITER; iter++) {
    A.SetZero();
    b.SetZero();
    error = 0;
    // for every point, compute point contribution to A and b
    for (index_t r = 0; r < P.n_rows(); r++) {
      for (index_t c = 0; c < P.n_cols(); c++) {
	double P_rc = smooth_intensity(P, (double) r, (double) c, sigma);
	double r_map, c_map, d_map; // r & c after mapping, d_map: homogeneous
	projective_map(m, r, c, &r_map, &c_map, &d_map);
	double Q_rc_map, Q_dr, Q_dc;
	Q_rc_map = smooth_intensity(Q, r_map, c_map, sigma, &Q_dr, &Q_dc);
	double e = Q_rc_map-P_rc;
	de_dm[0] = r*Q_dr/d_map;
	de_dm[1] = c*Q_dr/d_map;
	de_dm[2] = Q_dr/d_map;
	de_dm[3] = r*Q_dc/d_map;
	de_dm[4] = c*Q_dc/d_map;
	de_dm[5] = Q_dc/d_map;
	de_dm[6] = -r*(r_map*Q_dr+c_map*Q_dc)/d_map;
	de_dm[7] = -c*(r_map*Q_dr+c_map*Q_dc)/d_map;
	// add point contribution to A and b
	for (index_t i = 0; i < 8; i++) {
	  b[i] += -2*e*de_dm[i];
	  for (index_t j = 0; j < 8; j++) 
	    A.ref(i, j) += de_dm[i]*de_dm[j];
	}
	error += e*e;
      }
    }
    //printf("iter = %d error = %f\n", iter, error);
    if (old_error > error) // good step
      lambda *= 0.8;
    else                   // bad step
      lambda *= 2.0;
    old_error = error;
    // A & b in place, compute direction to update m
    // dm = (A+\lambda I) \ b
    for (index_t i = 0; i < 8; i++) A.ref(i, i) += lambda;
    Vector dm;
    la::SolveInit(A, b, &dm);
    la::AddTo(dm, &m);
  }
  printf("error = %f\n", error);
  return 0;
}

#define max(a,b) ((a)>(b) ? (a):(b))
#define min(a,b) ((a)<(b) ? (a):(b))

double smooth_intensity(const Matrix& I, double r, double c, double sigma) {
  double width = sigma*3;
  index_t rmin, rmax, cmin, cmax;
  rmin = (index_t) max(0, ceil(r-width));
  cmin = (index_t) max(0, ceil(c-width));
  rmax = (index_t) min(I.n_rows()-1, floor(r+width));
  cmax = (index_t) min(I.n_cols()-1, floor(c+width));

  double intensity = 0;
  for (index_t r_u = rmin; r_u <= rmax; r_u++)
    for (index_t c_v = cmin; c_v <= cmax; c_v++) {
      double u = r-r_u;
      double v = c-c_v;
      double g = exp(-0.5/sigma/sigma*(u*u+v*v));
      intensity += I.get(r_u, c_v)*g;
    }
  return intensity;
}

double smooth_intensity(const Matrix& I, double r, double c, double sigma,
			double* Ir, double* Ic) {
  double width = sigma*3;
  index_t rmin, rmax, cmin, cmax;
  rmin = (index_t) max(0, ceil(r-width));
  cmin = (index_t) max(0, ceil(c-width));
  rmax = (index_t) min(I.n_rows()-1, floor(r+width));
  cmax = (index_t) min(I.n_cols()-1, floor(c+width));

  double intensity = 0;
  *Ir = 0; *Ic = 0;
  for (index_t r_u = rmin; r_u <= rmax; r_u++)
    for (index_t c_v = cmin; c_v <= cmax; c_v++) {
      double u = r-r_u;
      double v = c-c_v;
      double g = exp(-0.5/sigma/sigma*(u*u+v*v));
      double gr = -u/sigma/sigma*g;
      double gc = -v/sigma/sigma*g;
      intensity += I.get(r_u, c_v)*g;
      *Ir += I.get(r_u, c_v)*gr;
      *Ic += I.get(r_u, c_v)*gc;
    }
  return intensity;
}

void projective_map(const Vector& m, double r, double c, 
		    double* r_map, double* c_map, double* d_map) {
  *d_map = m[6]*r+m[7]*c+1;
  *r_map = (m[0]*r+m[1]*c+m[2])/(*d_map);
  *c_map = (m[3]*r+m[4]*c+m[5])/(*d_map);
}
