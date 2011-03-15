template <class L, class R, class S>
struct Shoot {
	Int nvar;
	Doub x1,x2;
	L &load;
	R &d;
	S &score;
	Doub atol,rtol;
	Doub h1,hmin;
	VecDoub y;
	Shoot(Int nvarr, Doub xx1, Doub xx2, L &loadd, R &dd, S &scoree) :
		nvar(nvarr), x1(xx1), x2(xx2), load(loadd), d(dd),
		score(scoree), atol(1.0e-14), rtol(atol), hmin(0.0), y(nvar) {}
	VecDoub operator() (VecDoub_I &v) {
		h1=(x2-x1)/100.0;
		y=load(x1,v);
		Output out;
		Odeint<StepperDopr853<R> > integ(y,x1,x2,atol,rtol,h1,hmin,out,d);
		integ.integrate();
		return score(x2,y);
	}
};
