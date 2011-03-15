void hypser(const Complex &a, const Complex &b, const Complex &c,
	const Complex &z, Complex &series, Complex &deriv)
{
	deriv=0.0;
	Complex fac=1.0;
	Complex temp=fac;
	Complex aa=a;
	Complex bb=b;
	Complex cc=c;
	for (Int n=1;n<=1000;n++) {
		fac *= ((aa*bb)/cc);
		deriv += fac;
		fac *= ((1.0/n)*z);
		series=temp+fac;
		if (series == temp) return;
		temp=series;
		aa += 1.0;
		bb += 1.0;
		cc += 1.0;
	}
	throw("convergence failure in hypser");
}
struct Hypderiv {
	Complex a,b,c,z0,dz;
	Hypderiv(const Complex &aa, const Complex &bb,
		const Complex &cc, const Complex &z00,
		const Complex &dzz) : a(aa),b(bb),c(cc),z0(z00),dz(dzz) {}
	void operator() (const Doub s, VecDoub_I &yy, VecDoub_O &dyyds) {
		Complex z,y[2],dyds[2];
		y[0]=Complex(yy[0],yy[1]);
		y[1]=Complex(yy[2],yy[3]);
		z=z0+s*dz;
		dyds[0]=y[1]*dz;
		dyds[1]=(a*b*y[0]-(c-(a+b+1.0)*z)*y[1])*dz/(z*(1.0-z));
		dyyds[0]=real(dyds[0]);
		dyyds[1]=imag(dyds[0]);
		dyyds[2]=real(dyds[1]);
		dyyds[3]=imag(dyds[1]);
	}
};
Complex hypgeo(const Complex &a, const Complex &b,const Complex &c,
	const Complex &z)
{
	const Doub atol=1.0e-14,rtol=1.0e-14;
	Complex ans,dz,z0,y[2];
	VecDoub yy(4);
	if (norm(z) <= 0.25) {
		hypser(a,b,c,z,ans,y[1]);
		return ans;
	}
	else if (real(z) < 0.0) z0=Complex(-0.5,0.0);
	else if (real(z) <= 1.0) z0=Complex(0.5,0.0);
	else z0=Complex(0.0,imag(z) >= 0.0 ? 0.5 : -0.5);
	dz=z-z0;
	hypser(a,b,c,z0,y[0],y[1]);
	yy[0]=real(y[0]);
	yy[1]=imag(y[0]);
	yy[2]=real(y[1]);
	yy[3]=imag(y[1]);
	Hypderiv d(a,b,c,z0,dz);
	Output out;
	Odeint<StepperBS<Hypderiv> > ode(yy,0.0,1.0,atol,rtol,0.1,0.0,out,d);
	ode.integrate();
	y[0]=Complex(yy[0],yy[1]);
	return y[0];
}
