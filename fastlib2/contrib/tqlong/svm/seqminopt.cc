#include "svm.h"

namespace SVMLib {

bool targetpm1(const Vector& y);
bool vecgeq(const Vector& x, double tol);
void SelectWorkingSet(const Vector& y, const Vector& grad, 
		      const IndexSet& Iup, const IndexSet& Idown,
		      index_t& ii, index_t& jj,
		      double& val1, double& val2);
void CalABVec(const Vector& y, const Vector& box,
	      Vector& Avec, Vector& Bvec);
void InitIupdown(const Vector& y, const Vector& alpha, 
		 const Vector& AVec, const Vector& BVec, 
		 IndexSet& Iup, IndexSet& Idown);
void maxIup(const Vector& y, const Vector& grad, const IndexSet& Iup, 
	    index_t& ii, double& val1);
void minIdown(const Vector& y, const Vector& grad, const IndexSet& Idown, 
	    index_t& jj, double& val2);
index_t CheckKKT(const Vector& y, const Vector& box, const Vector& grad, 
		 const Vector& alpha, double offset, double tolKKT);
void UpdateAlphas(index_t i, index_t j, Kernel& kernel,
		  const Vector& y, const Vector& box, 		  
		  const Vector& Avec, const Vector& Bvec,
		  Vector& grad, Vector& alpha, IndexSet& Iup, IndexSet& Idown);
double eps = 1e-12;
double svTol = 1e-6;
  //#define leq(a, b) ((a) <= (b)+eps)
  //#define geq(a, b) ((a) >= (b)-eps)
#define leq(a, b, tol) ((a) <= (b)+(tol))
#define geq(a, b, tol) ((a) >= (b)-(tol))
#define lt(a, b, tol) ((a) <= (b)-(tol))
#define gt(a, b, tol) ((a) >= (b)+(tol))
#define min(a, b) ((a)<(b)?(a):(b))
#define max(a, b) ((a)>(b)?(a):(b))

int seqminopt(const Matrix& X, const Vector& y, const Vector& box,
	      KernelFunc kfunc, SMOOptions options,
	      Vector& alpha, IndexSet& SVs, double& offset) {
  // Check consistency
  index_t n_points = X.n_cols();
  DEBUG_ASSERT(y.length() == n_points); // check number of taget labels
  DEBUG_ASSERT(targetpm1(y)); // check if targets contain both +1 and -1
  DEBUG_ASSERT(box.length() == n_points); // check number of box constraints 
  DEBUG_ASSERT(vecgeq(box, options.tolKKT)); // check box >= KKT tolerance

  // Initialization
  //svTol = sqrt(eps);
  double KKTViolationTol = options.KKTViolationLevel * n_points;
  Kernel kernel(kfunc, X);
  Vector kernel_diag; kernel.get_diag(&kernel_diag);
  
  alpha.Init(n_points); alpha.SetZero();
  SVs.InitEmpty();
  Vector grad; grad.Init(n_points); grad.SetAll(1.0);
  offset = NAN;

  Vector Avec, Bvec; 
  CalABVec(y, box, Avec, Bvec);
  IndexSet Iup(n_points), Idown(n_points);
  InitIupdown(y, alpha, Avec, Bvec, Iup, Idown);

  // Main loop
  for (index_t iter = 0; iter < 1000; iter++) {
    index_t i, j;
    double val1, val2;
    SelectWorkingSet(y, grad, Iup, Idown, i, j, val1, val2);
    offset = (val1+val2)/2;
    // Check exit condition
    if (val1-val2 < options.tolKKT) {
      break;
    }
    UpdateAlphas(i, j, kernel, y, box, grad, Avec, Bvec, alpha, Iup, Idown);
    // Check number of KKT violations every 500 iterations
    if (iter % 500 == 0) {
      maxIup(y, grad, Iup, i, val1);
      minIdown(y, grad, Idown, j, val2);
      offset = (val1+val2)/2;
      index_t count = CheckKKT(y, box, grad, alpha, offset, options.tolKKT);
      if (KKTViolationTol > 0 && count <= KKTViolationTol) break;
    }    
  }

  for (index_t i = 0; i < n_points; i++)
    SVs.addremove(i, lt(alpha[i], 0, eps));

  // Check box contraints
  index_t box_violate = 0;
  for (index_t i = 0; i < n_points; i++)
    if (!geq(alpha[i], 0,eps) || !leq(alpha[i], box[i],eps)) box_violate++;
  printf("Total box constraint violations = %d", box_violate);
  // Check linear constraints
  double linEq = la::Dot(y, alpha);
  if (fabs(linEq) >= options.tolKKT) 
    printf("Linear constraint is violated");
  return 1;
}

void UpdateAlphas(index_t i, index_t j, Kernel& kernel,
		  const Vector& y, const Vector& box, 
		  const Vector& Avec, const Vector& Bvec,
		  Vector& grad,Vector& alpha, IndexSet& Iup, IndexSet& Idown) {
  double Kii = 0, Kjj = 0, Kij = 0;
  double eta;
  double low, high;
  double alpha_i, alpha_j;
  kernel.get_element(i, j, Kii, Kjj, Kij);
  eta = Kii+Kjj-2*Kij;

  // clip limits
  if (y[i] == y[j]) {
    low = max(0, alpha[j] + alpha[i] - box[i]);
    high = min(box[j], alpha[j] + alpha[i]);
  }
  else {
    low = max(0, alpha[j] - alpha[i]);
    high = min(box[j], box[i]+alpha[j]-alpha[i]);
  }

  // analytic solution of alpha_i and alpha_j
  if (geq(eta, 0, eps)) { // eta > 0
    double lambda = -y[i]*grad[i]+y[j]*grad[j];
    alpha_j = alpha[j] + y[j]/eta*lambda;
    // alpha_j = min(high, max(low, alpha_j));
    if (leq(alpha_j, low, eps)) alpha_j = low;
    else if (geq(alpha_j, high, eps)) alpha_j = high;
  }
  else {
    //evalPsiAtEnd(i, j, low, high);
    double s = y[i] * y[j];
    double fi = -grad[i]-alpha[i]*Kii-s*alpha[i]*Kij;
    double fj = -grad[j]-alpha[j]*Kjj-s*alpha[i]*Kij;
    double Li = alpha[i]+s*(alpha[j]-low);
    double Hi = alpha[i]+s*(alpha[j]-high);
    double psi_l = Li*fi+low*fj+Li*Li*Kii/2+low*low*Kjj/2+s*low*Li*Kij;
    double psi_h = Hi*fi+high*fj+Hi*Hi*Kii/2+high*high*Kjj/2+s*high*Hi*Kij;
    if (lt(psi_l, psi_h, eps))
      alpha_j = low;
    else if (gt(psi_l, psi_h, eps))
      alpha_j = high;
    else alpha_j = alpha[j];
  }
  alpha_i = alpha[i]+y[j]*y[i]*(alpha[j]-alpha_j);
  if (leq(alpha_i, 0, eps)) alpha_i = 0;
  else if (geq(alpha_i, box[i], eps)) alpha_i = box[i];

  // update grad
  Vector col_i, col_j;
  kernel.get_column(i, &col_i);
  kernel.get_column(j, &col_j);
  for (index_t k = 0; k < grad.length(); k++) {
    grad[k] = grad[k] - col_i[k]*y[k]*(alpha_i-alpha[i])*y[i]
      - col_j[k]*y[k]*(alpha_j-alpha[j])*y[j];
  }
  Iup.addremove(i, lt(y[i]*alpha_i, Bvec[i], svTol));
  Idown.addremove(i, gt(y[i]*alpha_i, Avec[i], svTol));
  Iup.addremove(j, lt(y[j]*alpha_j, Bvec[j], svTol));
  Idown.addremove(j, gt(y[j]*alpha_j, Avec[j], svTol));

  alpha[i] = alpha_i;
  alpha[j] = alpha_j;
}

index_t CheckKKT(const Vector& y, const Vector& box, const Vector& grad, 
		 const Vector& alpha, double offset, double tolKKT) {
  index_t n_points = y.length();
  index_t count = 0;
  for (index_t i = 0; i < n_points; i++) {
    double amount = -grad[i] + y[i]*offset;
    if (leq(alpha[i], 0, svTol)) { // alpha[i] == 0
      if (geq(amount, 0, tolKKT)) count++;
    }
    else if (geq(alpha[i], box[i], svTol)) { // alpha[i] == box[i]
      if (leq(amount, 0, tolKKT)) count++;
    }
    else { // 0 < alpha[i] < box[i]
      if (leq(amount, 0, tolKKT) && geq(amount, 0, tolKKT)) count++;
    }
  }
  return count;
}

void SelectWorkingSet(const Vector& y, const Vector& grad, 
		      const IndexSet& Iup, const IndexSet& Idown,
		      index_t& ii, index_t& jj,
		      double& val1, double &val2) {
  // select i
  maxIup(y, grad, Iup, ii, val1);
  // select j
  minIdown(y, grad, Idown, jj, val2);
}

void maxIup(const Vector& y, const Vector& grad, const IndexSet& Iup, 
	    index_t& ii, double& val1) {
  double max = -INFINITY;
  for (index_t i = 0; i < Iup.get_n(); i++) {
    double val1 = y[Iup[i]]*grad[Iup[i]];
    if (val1 > max) {
      max = val1;
      ii = Iup[i];
    }
  }
  val1 = max;
}

void minIdown(const Vector& y, const Vector& grad, const IndexSet& Idown, 
	    index_t& jj, double& val2) {
  double min = INFINITY;
  for (index_t i = 0; i < Idown.get_n(); i++) {
    double val2 = y[Idown[i]]*grad[Idown[i]];
    if (val2 < min) {
      min = val2;
      jj = Idown[i];
    }
  }
  val2 = min;
}

void InitIupdown(const Vector& y, const Vector& alpha, 
		 const Vector& Avec, const Vector& Bvec, 
		 IndexSet& Iup, IndexSet& Idown) {
  index_t n_points = y.length();
  for (index_t i = 0; i < n_points; i++) {
    Iup.addremove(i, lt(y[i]*alpha[i], Bvec[i], svTol));
    Idown.addremove(i, gt(y[i]*alpha[i], Avec[i], svTol));
  }
}

void CalABVec(const Vector& y, const Vector& box,
	      Vector& Avec, Vector& Bvec) {
  index_t n_points = y.length();
  Avec.Init(n_points); Bvec.Init(n_points);
  for (index_t i = 0; i < n_points; i++) {
    if (y[i] == -1) {
      Avec[i] = -box[i];
      Bvec[i] = 0;
    }
    else {
      Avec[i] = 0;
      Bvec[i] = box[i];
    }
  }
}

bool targetpm1(const Vector& y) {
  bool p1 = false;
  bool m1 = false;
  for (index_t i = 0; i < y.length(); i++) {
    if (y[i] == +1.0) p1 = true;
    else if (y[i] == -1.0) m1 = true;
    else {
      p1 = false;
      m1 = false;
      break;
    }
  }
  return p1 && m1;
}

bool vecgeq(const Vector& x, double tol) {
  for (index_t i = 0; i < x.length(); i++)
    if (x[i] < tol) return false;
  return true;
}

};
