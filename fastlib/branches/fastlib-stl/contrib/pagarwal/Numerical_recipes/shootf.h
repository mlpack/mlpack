template <class L1, class L2, class R, class S>
struct Shootf {
	Int nvar,n2;
	Doub x1,x2,xf;
	L1 &load1;
	L2 &load2;
	R &d;
	S &score;
	Doub atol,rtol;
	Doub h1,hmin;
	VecDoub y,f1,f2;
	Shootf(Int nvarr, Int nn2,  Doub xx1, Doub xx2, Doub xxf, L1 &loadd1,
		L2 &loadd2, R &dd, S &scoree) : nvar(nvarr), n2(nn2), x1(xx1),
		x2(xx2), xf(xxf), load1(loadd1), load2(loadd2), d(dd),
		score(scoree), atol(1.0e-14), rtol(atol), hmin(0.0), y(nvar),
		f1(nvar), f2(nvar) {}
	VecDoub operator() (VecDoub_I &v) {
		VecDoub v2(nvar-n2,&v[n2]);
		h1=(x2-x1)/100.0;
		y=load1(x1,v);
		Output out;
		Odeint<StepperDopr853<R> > integ1(y,x1,xf,atol,rtol,h1,hmin,out,d);
		integ1.integrate();
		f1=score(xf,y);
		y=load2(x2,v2);
		Odeint<StepperDopr853<R> > integ2(y,x2,xf,atol,rtol,h1,hmin,out,d);
		integ2.integrate();
		f2=score(xf,y);
		for (Int i=0;i<nvar;i++) f1[i] -= f2[i];
		return f1;
	}
};
