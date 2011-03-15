struct Rhsfpt {
	Int m;
	Doub c2;
	Rhsfpt(Int mm, Doub cc2) : m(mm), c2(cc2) {}
	void operator() (const Doub x, VecDoub_I &y, VecDoub_O &dydx)
	{
		dydx[0]=y[1];
		dydx[1]=(2.0*x*(m+1.0)*y[1]-(y[2]-c2*x*x)*y[0])/(1.0-x*x);
		dydx[2]=0.0;
	}
};

struct Load1 {
	Int n,m;
	Doub gmma,c2,dx;
	VecDoub y;
	Load1(Int nn, Int mm, Doub gmmaa, Doub cc2, Doub dxx) : n(nn), m(mm),
		gmma(gmmaa), c2(cc2), dx(dxx), y(3) {}
	VecDoub operator() (const Doub x1, VecDoub_I &v1)
	{
		Doub y1 = ((n-m & 1) != 0 ? -gmma : gmma);
		y[2]=v1[0];
		y[1] = -(y[2]-c2)*y1/(2*(m+1));
		y[0]=y1+y[1]*dx;
		return y;
	}
};

struct Load2 {
	Int m;
	Doub c2;
	VecDoub y;
	Load2(Int mm, Doub cc2) : m(mm), c2(cc2), y(3) {}
	VecDoub operator() (const Doub x2, VecDoub_I &v2)
	{
		y[2]=v2[1];
		y[0]=v2[0];
		y[1]=(y[2]-c2)*y[0]/(2*(m+1));
		return y;
	}
};

struct Score {
	VecDoub f;
	Score() : f(3) {}
	VecDoub operator() (const Doub xf, VecDoub_I &y)
	{
		for (Int i=0;i<3;i++) f[i]=y[i];
		return f;
	}
};

Int main_sphfpt(void) {
	const Int N1=2,N2=1,NTOT=N1+N2,MM=3;
	Bool check;
	VecDoub v(NTOT);
	Int j,m=3,n=5,n2=N2;
	Doub c2[]={1.5,-1.5,0.0};
	Int nvar=NTOT;
	Doub dx=1.0e-8;
	for (j=0;j<MM;j++) {
		Doub gmma=1.0;
		Doub q1=n;
		for (Int i=1;i<=m;i++) gmma *= -0.5*(n+i)*(q1--/i);
		v[0]=n*(n+1)-m*(m+1)+c2[j]/2.0;
		v[2]=v[0];
		v[1]=gmma*(1.0-(v[2]-c2[j])*dx/(2*(m+1)));
		Doub x1= -1.0+dx;
		Doub x2=1.0-dx;
		Doub xf=0.0;
		Load1 load1(n,m,gmma,c2[j],dx);
		Load2 load2(m,c2[j]);
		Rhsfpt d(m,c2[j]);
		Score score;
		Shootf<Load1,Load2,Rhsfpt,Score> shootf(nvar,n2,x1,x2,xf,load1,
			load2,d,score);
		newt(v,check,shootf);
		if (check) {
			cout << "shootf failed; bad initial guess" << endl;
		} else {
			cout << "    " << "mu(m,n)" << endl;
			cout << fixed << setprecision(6);
			cout << setw(12) << v[0] << endl;
		}
	}
	return 0;
}
