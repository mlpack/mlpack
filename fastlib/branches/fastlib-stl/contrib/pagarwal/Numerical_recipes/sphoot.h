struct Rhs {
	Int m;
	Doub c2;
	Rhs(Int mm, Doub cc2) : m(mm), c2(cc2) {}
	void operator() (const Doub x, VecDoub_I &y, VecDoub_O &dydx)
	{
		dydx[0]=y[1];
		dydx[1]=(2.0*x*(m+1.0)*y[1]-(y[2]-c2*x*x)*y[0])/(1.0-x*x);
		dydx[2]=0.0;
	}
};

struct Load {
	Int n,m;
	Doub gmma,c2,dx;
	VecDoub y;
	Load(Int nn, Int mm, Doub gmmaa, Doub cc2, Doub dxx) : n(nn), m(mm),
		gmma(gmmaa), c2(cc2), dx(dxx), y(3) {}
	VecDoub operator() (const Doub x1, VecDoub_I &v)
	{
		Doub y1 = ((n-m & 1) != 0 ? -gmma : gmma);
		y[2]=v[0];
		y[1] = -(y[2]-c2)*y1/(2*(m+1));
		y[0]=y1+y[1]*dx;
		return y;
	}
};

struct Score {
	Int n,m;
	VecDoub f;
	Score(Int nn, Int mm) : n(nn), m(mm), f(1) {}
	VecDoub operator() (const Doub xf, VecDoub_I &y)
	{
		f[0]=((n-m & 1) != 0 ? y[0] : y[1]);
		return f;
	}
};

Int main_sphoot(void) {
	const Int N2=1,MM=3;
	Bool check;
	VecDoub v(N2);
	Int j,m=3,n=5;
	Doub c2[]={1.5,-1.5,0.0};
	Int nvar=3;
	Doub dx=1.0e-8;
	for (j=0;j<MM;j++) {
		Doub gmma=1.0;
		Doub q1=n;
		for (Int i=1;i<=m;i++) gmma *= -0.5*(n+i)*(q1--/i);
		v[0]=n*(n+1)-m*(m+1)+c2[j]/2.0;
		Doub x1= -1.0+dx;
		Doub x2=0.0;
		Load load(n,m,gmma,c2[j],dx);
		Rhs d(m,c2[j]);
		Score score(n,m);
		Shoot<Load,Rhs,Score> shoot(nvar,x1,x2,load,d,score);
		newt(v,check,shoot);
		if (check) {
			cout << "shoot failed; bad initial guess" << endl;
		} else {
			cout << "    " << "mu(m,n)" << endl;
			cout << fixed << setprecision(6);
			cout << setw(12) << v[0] << endl;
		}
	}
	return 0;
}
