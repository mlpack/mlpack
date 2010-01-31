
#pragma once

#include <fastlib/fastlib.h>

typedef Vector Argument;

namespace optim {

  /** Back tracking line search on direction dir
   *  using Armijo condition, starting step size 1.0
   *  Assume that <gx0, dir> < 0 (descent direction)
   *  Need to implement
   *    F.AddExpert(double alpha, const T& x, T* y) // y = y+alpha*x
   *    F.Dot(const T& x, const T& y) // return dot product of x and y
   */
  template <class F, class FArg>
  double BackTrackLineSearch(F& fun,
			     const FArg& x0, double fx0, 
			     const FArg& gx0, const FArg& dir, 
			     FArg* xnew, double fTol) {
    double desR = fun.Dot(gx0, dir); // descent rate
    double sign = (desR<0)?1.0:-1.0;
    //printf("desR = %f\n", desR);
    //ot::Print(gx0); ot::Print(dir);
    //DEBUG_ASSERT(desR <= 0);
    // if (fabs(desR)/(fx0>1.0?fx0:1.0) < fTol) { // Objective change too small
    //	(*xnew) = x0;
    //  return fx0;      
    //}
    DEBUG_ASSERT(~isinf(fx0)); // infeasible ?

    double step = 1.0;
    double tau = 0.5;
    double c1 = 1e-4;
    double c2 = 0.9;
    (*xnew) = x0;
    fun.AddExpert(step*sign, dir, xnew);
    FArg grad;
    grad.Init(fun.n_dim());
    while (1) {
      double fval = fun.fValue(*xnew);
      if (fval <= fx0 + c1*step*sign*desR) { // Armijo condition
	fun.fGradient(*xnew, &grad);
	if (fun.Dot(grad, dir) >= c2*desR) // Culvature condition
	  return fval; 
      }
      
      if (fabs(step*desR)/(fx0>1.0?fx0:1.0) < fTol) { 
	// Objective changes too small
	(*xnew) = x0;
	return fx0;
      }
      step *= tau;
      fun.AddExpert(step*sign*(1-1/tau), dir, xnew);
    }
  }
  
  template <class F, class FArg>
  double GradientDescent(F& fun,
			 const FArg& x0, FArg* x,
			 int maxIter, double fTol, double gTol,
			 FArg* grad = NULL) {
    Vector g;
    Vector xnew;
    Vector dir;

    xnew.Init(fun.n_dim());
    g.Init(fun.n_dim());
    dir.Init(fun.n_dim());

    (*x) = x0;
    double f = fun.fValue(*x);
    for (int i = 0; i < maxIter; i++) {
      fun.fGradient(*x, &g);

      double gNorm = sqrt(fun.Dot(g, g));
      if (gNorm < gTol) {
	//printf("Gradient too small.\n");
	if (grad != NULL) (*grad) = g;
	break;
      }

      double old_f = f;
      fun.ScaleOverwrite(-1.0, g, &dir);      
      f = BackTrackLineSearch(fun, *x, f, g, dir, &xnew, fTol);      
      (*x) = xnew;

      //printf("iter %d f = %f\n", i, f);

      if (fabs(f-old_f)/(f>1.0?f:1.0) < fTol) {
	//printf("Objective changes too small.\n");
	if (grad != NULL) fun.fGradient(*x, grad);
	break;
      }
    }
    
    return f;
  }

  template <class F, class FArg>
  double BFGSDescent(F& fun,
		     const FArg& x0, FArg* x,
		     int maxIter, double fTol, double gTol) {
    index_t dim = fun.n_dim();
    DEBUG_ASSERT(dim == x0.length());
    DEBUG_ASSERT(dim == x->length());

    FArg xnew;
    FArg g1, g2;
    FArg *xk = x, *xkp1 = &xnew;
    FArg *gk = &g1, *gkp1 = &g2, dir;
    FArg sk, yk, Hv;
    double rhok;
    
    (*xk) = x0; xkp1->Init(dim);
    g1.Init(dim); g2.Init(dim);
    dir.Init(dim);
    sk.Init(dim); yk.Init(dim);
    Hv.Init(dim); 
    Matrix H;
    H.Init(dim, dim);
    H.SetZero();
    for (index_t i = 0; i < dim; i++) H.ref(i, i) = 1.0;

    double f = fun.fValue(*xk);
    fun.fGradient(*xk, gk);
    for (index_t k = 0; k < maxIter; k++) {
      double gNorm = sqrt(fun.Dot(*gk, *gk));
      if (gNorm < gTol) {
	printf("Gradient too small.\n");
	break;
      }

      // Compute search direction
      fun.MulOverwrite(H, *gk, &dir);
      fun.Scale(-1.0, &dir);

      //ot::Print(*xk);
      //ot::Print(dir);

      // Back track line search
      double old_f = f;
      f = BackTrackLineSearch(fun, *xk, f, *gk, dir, xkp1, fTol);

      fun.fGradient(*xkp1, gkp1);

      fun.SubOverwrite(*xk, *xkp1, &sk);
      fun.SubOverwrite(*gk, *gkp1, &yk);

      rhok = 1.0/fun.Dot(yk, sk);

      // BFGS_Update_H_InPlace(&H, rhok, sk, yk); //O(n^2)
      fun.MulOverwrite(H, yk, &Hv);
      for (index_t i = 0; i < dim; i++)
	for (index_t j = 0; j < dim; j++)
	  H.ref(i, j) += -rhok*Hv[i]*sk[j];
      fun.MulOverwrite(yk, H, &Hv);
      for (index_t i = 0; i < dim; i++)
	for (index_t j = 0; j < dim; j++)
	  H.ref(i, j) += -rhok*sk[i]*Hv[j]+rhok*sk[i]*sk[j];
            
      FArg *tmp; // swap the pointers to avoid copying
      tmp = xk; xk = xkp1; xkp1 = tmp;
      tmp = gk; gk = gkp1; gkp1 = tmp;

      printf("iter %d f = %f\n", k, f);
      //ot::Print(*xk);

      if (fabs(f-old_f)/(f>1.0?f:1.0) < fTol) {
	printf("Objective changes too small.\n");
	break;
      }
    }
    if (xk != x) (*x) = (*xk);
    return f;
  }

  template <class F, class FArg>
  double L_BFGSDescent(F& fun,
		       const FArg& x0, FArg* x,
		       int maxIter, double fTol, double gTol, index_t mem_size,
		       int* nIter = NULL) {
    index_t dim = fun.n_dim();
    DEBUG_ASSERT(dim == x0.length());
    DEBUG_ASSERT(dim == x->length());

    ArrayList<FArg> s, y;
    Vector rho, alpha;
    double gamma = 1.0;
    s.Init(); y.Init();
    rho.Init(mem_size); alpha.Init(mem_size);

    FArg xnew;
    FArg g1, g2;
    FArg *xk = x, *xkp1 = &xnew;
    FArg *gk = &g1, *gkp1 = &g2, dir;
    FArg sk, yk;
    
    (*xk) = x0; xkp1->Init(dim);
    g1.Init(dim); g2.Init(dim);
    dir.Init(dim);
    sk.Init(dim); yk.Init(dim);

    double f = fun.fValue(*xk);
    fun.fGradient(*xk, gk);
    index_t k;
    for (k = 0; k < maxIter; k++) {
      double gNorm = sqrt(fun.Dot(*gk, *gk));
      if (gNorm < gTol) {
	printf("Gradient too small.\n");
	break;
      }

      // Compute search direction (L-BFGS two loop recursion)
      fun.ScaleOverwrite(-1.0, *gk, &dir);
      for (index_t i = k-1; i >= 0; i--) {
	index_t id = i%mem_size;
	alpha[id] = rho[id]*fun.Dot(s[id], dir);
	fun.AddExpert(-alpha[id], y[id], &dir);
      }
      fun.Scale(gamma, &dir);
      for (index_t i = (k-mem_size>=0?k-mem_size:0); i < k; i++) {
	index_t id = i%mem_size;
	double beta = rho[id]*fun.Dot(y[id], dir);
	fun.AddExpert(alpha[id]-beta, s[id], &dir);
      }
      //ot::Print(*xk);
      //ot::Print(dir);

      // Back track line search
      double old_f = f;
      f = BackTrackLineSearch(fun, *xk, f, *gk, dir, xkp1, fTol);

      fun.fGradient(*xkp1, gkp1);

      fun.SubOverwrite(*xk, *xkp1, &sk);
      fun.SubOverwrite(*gk, *gkp1, &yk);

      if (k >= mem_size) {
	s[k%mem_size] = sk;
	y[k%mem_size] = yk;
      }
      else {
	s.PushBackCopy(sk);
	y.PushBackCopy(yk);
      }

      gamma = fun.Dot(sk, yk);
      rho[k%mem_size] = 1.0/gamma;
      gamma /= fun.Dot(yk, yk);
      if (isnan(gamma) || isinf(gamma)) gamma = 1.0; // reset H0
      //printf("gamma = %f\n", gamma);

      FArg *tmp; // swap the pointers to avoid copying
      tmp = xk; xk = xkp1; xkp1 = tmp;
      tmp = gk; gk = gkp1; gkp1 = tmp;

      //printf("iter %d f = %f\n", k, f);
      //ot::Print(*xk);

      if (fabs(f-old_f)/(f>1.0?f:1.0) < fTol) {
	//printf("Objective changes too small.\n");
	break;
      }
    }
    if (xk != x) (*x) = (*xk);
    if (nIter != NULL) *nIter += k;
    return f;
  }

  template <class F, class FArg>
  class Barrier {
    F* pf;
    double t;
  public:
    Barrier(F& f) { this->pf = &f; }
    void setT(double t) { this->t = t; }

    index_t n_dim() { return pf->n_dim(); }
    double fValue(const FArg& x) {
      double fval = t*pf->fValue(x);
      for (index_t i = 0; i < pf->n_con(); i++) {
	double c = pf->cValue(i, x);
	if (c >= 0) return INFINITY;
	fval += -log(-c);
      }
      return fval;
    }
   
    void fGradient(const FArg& x, FArg* g) {
      pf->fGradient(x, g);
      pf->Scale(t, g);
      FArg cGrad;
      cGrad.Init(n_dim());
      for (index_t i = 0; i < pf->n_con(); i++) {
	pf->cGradient(i, x, &cGrad);
	pf->AddExpert(-1.0/pf->cValue(i, x), cGrad, g);
      }
    }
    
    double Dot(const FArg& x, const FArg& y) { return pf->Dot(x, y); }
    void AddExpert(double alpha, const FArg& x, FArg* y) {
      pf->AddExpert(alpha, x, y);
    }
    void Scale(double alpha, FArg* x) { pf->Scale(alpha, x); }
    void ScaleOverwrite(double alpha, const FArg& x, FArg* y) {
      pf->ScaleOverwrite(alpha, x, y);
    }
    void SubOverwrite(const FArg &x, const FArg& y, FArg* z) {
      pf->SubOverwrite(x, y, z);
    }
    void MulOverwrite(const Matrix &A, const FArg& x, FArg* y) {
      pf->MulOverwrite(A, x, y);
    }
    void MulOverwrite(const FArg& x, const Matrix &A, FArg* y) {
      pf->MulOverwrite(x, A, y);
    }
  };

  template <class F, class FArg>
  double BarrierMethod(F& fun,
		       const FArg& x0, FArg* x,
		       int maxIter, double fTol, double gTol) {
    double mu = 5.0;
    double t = 1.0;

    FArg xnew;
    xnew.Init(fun.n_dim());

    (*x) = x0;
    double f;
    Barrier<F, FArg> barrier(fun);

    int nIter = 0;
    while (fun.n_con()/t > fTol) {
      barrier.setT(t);
      //f = GradientDescent(barrier, *x, &xnew, maxIter, fTol*2, gTol);
      //f = BFGSDescent(barrier, *x, &xnew, maxIter, fTol*10, gTol);
      f = L_BFGSDescent(barrier, *x, &xnew, maxIter-nIter, fTol*10, gTol, 10,
			&nIter);
      (*x) = xnew;
      t = t*mu;
      printf("iter %d f = %f\n", nIter, fun.fValue(*x));
    }
    return fun.fValue(*x);
  }			   
}
