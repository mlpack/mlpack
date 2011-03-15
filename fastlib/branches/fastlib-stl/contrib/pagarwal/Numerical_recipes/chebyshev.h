struct Chebyshev {
	Int n,m;
	VecDoub c;
	Doub a,b;

	Chebyshev(Doub func(Doub), Doub aa, Doub bb, Int nn);
	Chebyshev(VecDoub &cc, Doub aa, Doub bb)
		: n(cc.size()), m(n), c(cc), a(aa), b(bb) {}
	Int setm(Doub thresh) {while (m>1 && abs(c[m-1])<thresh) m--; return m;}
	
	Doub eval(Doub x, Int m);
	inline Doub operator() (Doub x) {return eval(x,m);}
	
	Chebyshev derivative();
	Chebyshev integral();

	VecDoub polycofs(Int m);
	inline VecDoub polycofs() {return polycofs(m);}
	Chebyshev(VecDoub &pc);
	
};
Chebyshev::Chebyshev(Doub func(Doub), Doub aa, Doub bb, Int nn=50)
	: n(nn), m(nn), c(n), a(aa), b(bb)
{
	const Doub pi=3.141592653589793;
	Int k,j;
	Doub fac,bpa,bma,y,sum;
	VecDoub f(n);
	bma=0.5*(b-a);
	bpa=0.5*(b+a);
	for (k=0;k<n;k++) {
		y=cos(pi*(k+0.5)/n);
		f[k]=func(y*bma+bpa);
	}
	fac=2.0/n;
	for (j=0;j<n;j++) {
		sum=0.0;
		for (k=0;k<n;k++)
			sum += f[k]*cos(pi*j*(k+0.5)/n);
		c[j]=fac*sum;
	}
}
Doub Chebyshev::eval(Doub x, Int m)
{
	Doub d=0.0,dd=0.0,sv,y,y2;
	Int j;
	if ((x-a)*(x-b) > 0.0) throw("x not in range in Chebyshev::eval");
	y2=2.0*(y=(2.0*x-a-b)/(b-a));
	for (j=m-1;j>0;j--) {
		sv=d;
		d=y2*d-dd+c[j];
		dd=sv;
	}
	return y*d-dd+0.5*c[0];
}
Chebyshev Chebyshev::derivative()
{
	Int j;
	Doub con;
	VecDoub cder(n);
	cder[n-1]=0.0;
	cder[n-2]=2*(n-1)*c[n-1];
	for (j=n-2;j>0;j--)
		cder[j-1]=cder[j+1]+2*j*c[j];
	con=2.0/(b-a);
	for (j=0;j<n;j++) cder[j] *= con;
	return Chebyshev(cder,a,b);
}
Chebyshev Chebyshev::integral()
{
	Int j;
	Doub sum=0.0,fac=1.0,con;
	VecDoub cint(n);
	con=0.25*(b-a);
	for (j=1;j<n-1;j++) {
		cint[j]=con*(c[j-1]-c[j+1])/j;
		sum += fac*cint[j];
		fac = -fac;
	}
	cint[n-1]=con*c[n-2]/(n-1);
	sum += fac*cint[n-1];
	cint[0]=2.0*sum;
	return Chebyshev(cint,a,b);
}
Chebyshev::Chebyshev(VecDoub &d)
	: n(d.size()), m(n), c(n), a(-1.), b(1.)
{
	c[n-1]=d[n-1];
	c[n-2]=2.0*d[n-2];
	for (Int j=n-3;j>=0;j--) {
		c[j]=2.0*d[j]+c[j+2];
		for (Int i=j+1;i<n-2;i++) {
				c[i] = (c[i]+c[i+2])/2;
		}
		c[n-2] /= 2;
		c[n-1] /= 2;
	}
}
VecDoub Chebyshev::polycofs(Int m)
{
	Int k,j;
	Doub sv;
	VecDoub d(m),dd(m);
	for (j=0;j<m;j++) d[j]=dd[j]=0.0;
	d[0]=c[m-1];
	for (j=m-2;j>0;j--) {
		for (k=m-j;k>0;k--) {
			sv=d[k];
			d[k]=2.0*d[k-1]-dd[k];
			dd[k]=sv;
		}
		sv=d[0];
		d[0] = -dd[0]+c[j];
		dd[0]=sv;
	}
	for (j=m-1;j>0;j--) d[j]=d[j-1]-dd[j];
	d[0] = -dd[0]+0.5*c[0];
	return d;
}
