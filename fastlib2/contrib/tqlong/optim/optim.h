
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
    DEBUG_ASSERT(desR < 0);
    DEBUG_ASSERT(~isinf(fx0));

    double step = 1.0;
    double tau = 0.5;
    double c1 = 1e-3;
    (*xnew) = x0;
    fun.AddExpert(step, dir, xnew);
    while (1) {
      double fval = fun.fValue(*xnew);
      if (fval <= fx0 + c1*step*desR) 
	return fval; // Armijo condition
      if ((-step*desR)/(fx0>1.0?fx0:1.0) < fTol) { 
	// Objective changes too small
	(*xnew) = x0;
	return fx0;
      }
      step *= tau;
      fun.AddExpert(step*(1-1/tau), dir, xnew);
    }
  }
  
  template <class F, class FArg>
  double GradientDescent(F& fun,
			 const FArg& x0, FArg* x,
			 int maxIter, double fTol, double gTol) {
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
	break;
      }

      double old_f = f;
      fun.ScaleOverwrite(-1.0, g, &dir);      
      f = BackTrackLineSearch(fun, *x, f, g, dir, &xnew, fTol);      
      x->CopyValues(xnew);

      //printf("iter %d f = %f\n", i, f);

      if (fabs(f-old_f)/(f>1.0?f:1.0) < fTol) {
	//printf("Objective changes too small.\n");
	break;
      }
    }
    
    return f;
  }

  template <class F, class FArg>
  class Barrier {
    F f;
    double t;
  public:
    Barrier(const F& f) { this->f = f; }

    void setT(double t) { this->t = t; }

    index_t n_dim() { return f.n_dim(); }

    double fValue(const FArg& x) {
      double fval = t*f.fValue(x);
      for (index_t i = 0; i < f.n_con(); i++) {
	double c = f.cValue(i, x);
	if (c >= 0) return INFINITY;
	fval += -log(-c);
      }
      return fval;
    }
   
    void fGradient(const FArg& x, FArg* g) {
      f.fGradient(x, g);
      f.Scale(t, g);
      FArg cGrad;
      cGrad.Init(n_dim());
      for (index_t i = 0; i < f.n_con(); i++) {
	f.cGradient(i, x, &cGrad);
	f.AddExpert(-1.0/f.cValue(i, x), cGrad, g);
      }
    }

    double Dot(const FArg& x, const FArg& y) { return f.Dot(x, y); }
    void ScaleOverwrite(double alpha, const FArg& x, FArg* y) {
      f.ScaleOverwrite(alpha, x, y);
    }
    void AddExpert(double alpha, const FArg& x, FArg* y) {
      f.AddExpert(alpha, x, y);
    }
    void Scale(double alpha, FArg* x) { f.Scale(alpha, x); }
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
    while (fun.n_con()/t > fTol) {
      barrier.setT(t);
      f = GradientDescent(barrier, *x, &xnew, maxIter, fTol*2, gTol);
      (*x) = xnew;
      t = t*mu;
    }
    return fun.fValue(*x);
  }			   
}
