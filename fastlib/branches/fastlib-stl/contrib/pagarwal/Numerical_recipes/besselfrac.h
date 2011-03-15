struct Bessel {
	static const Int NUSE1=7, NUSE2=8;
	static const Doub c1[NUSE1],c2[NUSE2];
	Doub xo,nuo;
	Doub jo,yo,jpo,ypo;
	Doub io,ko,ipo,kpo;
	Doub aio,bio,aipo,bipo;
	Doub sphjo,sphyo,sphjpo,sphypo;
	Int sphno;

	Bessel() : xo(9.99e99), nuo(9.99e99), sphno(-9999) {}

	void besseljy(const Doub nu, const Doub x);
	void besselik(const Doub nu, const Doub x);

	Doub jnu(const Doub nu, const Doub x) {
		if (nu != nuo || x != xo) besseljy(nu,x);
		return jo;
	}
	Doub ynu(const Doub nu, const Doub x) {
		if (nu != nuo || x != xo) besseljy(nu,x);
		return yo;
	}
	Doub inu(const Doub nu, const Doub x) {
		if (nu != nuo || x != xo) besselik(nu,x);
		return io;
	}
	Doub knu(const Doub nu, const Doub x) {
		if (nu != nuo || x != xo) besselik(nu,x);
		return ko;
	}

	void airy(const Doub x);
	Doub airy_ai(const Doub x);
	Doub airy_bi(const Doub x);

	void sphbes(const Int n, const Doub x);
	Doub sphbesj(const Int n, const Doub x);
	Doub sphbesy(const Int n, const Doub x);

	inline Doub chebev(const Doub *c, const Int m, const Doub x) {
		Doub d=0.0,dd=0.0,sv;
		Int j;
		for (j=m-1;j>0;j--) {
			sv=d;
			d=2.*x*d-dd+c[j];
			dd=sv;
		}
		return x*d-dd+0.5*c[0];
	}
};

const Doub Bessel::c1[7] = {-1.142022680371168e0,6.5165112670737e-3,
	3.087090173086e-4,-3.4706269649e-6,6.9437664e-9,3.67795e-11,
	-1.356e-13};
const Doub Bessel::c2[8] = {1.843740587300905e0,-7.68528408447867e-2,
	1.2719271366546e-3,-4.9717367042e-6,-3.31261198e-8,2.423096e-10,
	-1.702e-13,-1.49e-15};
void Bessel::besseljy(const Doub nu, const Doub x)
{
	const Int MAXIT=10000;
	const Doub EPS=numeric_limits<Doub>::epsilon();
	const Doub FPMIN=numeric_limits<Doub>::min()/EPS;
	const Doub XMIN=2.0, PI=3.141592653589793;
	Doub a,b,br,bi,c,cr,ci,d,del,del1,den,di,dlr,dli,dr,e,f,fact,fact2,
		fact3,ff,gam,gam1,gam2,gammi,gampl,h,p,pimu,pimu2,q,r,rjl,
		rjl1,rjmu,rjp1,rjpl,rjtemp,ry1,rymu,rymup,rytemp,sum,sum1,
		temp,w,x2,xi,xi2,xmu,xmu2,xx;
	Int i,isign,l,nl;

	if (x <= 0.0 || nu < 0.0) throw("bad arguments in besseljy");
	nl=(x < XMIN ? Int(nu+0.5) : MAX(0,Int(nu-x+1.5)));
	xmu=nu-nl;
	xmu2=xmu*xmu;
	xi=1.0/x;
	xi2=2.0*xi;
	w=xi2/PI;
	isign=1;
	h=nu*xi;
	if (h < FPMIN) h=FPMIN;
	b=xi2*nu;
	d=0.0;
	c=h;
	for (i=0;i<MAXIT;i++) {
		b += xi2;
		d=b-d;
		if (abs(d) < FPMIN) d=FPMIN;
		c=b-1.0/c;
		if (abs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		del=c*d;
		h=del*h;
		if (d < 0.0) isign = -isign;
		if (abs(del-1.0) <= EPS) break;
	}
	if (i >= MAXIT)
		throw("x too large in besseljy; try asymptotic expansion");
	rjl=isign*FPMIN;
	rjpl=h*rjl;
	rjl1=rjl;
	rjp1=rjpl;
	fact=nu*xi;
	for (l=nl-1;l>=0;l--) {
		rjtemp=fact*rjl+rjpl;
		fact -= xi;
		rjpl=fact*rjtemp-rjl;
		rjl=rjtemp;
	}
	if (rjl == 0.0) rjl=EPS;
	f=rjpl/rjl;
	if (x < XMIN) {
		x2=0.5*x;
		pimu=PI*xmu;
		fact = (abs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
		d = -log(x2);
		e=xmu*d;
		fact2 = (abs(e) < EPS ? 1.0 : sinh(e)/e);
		xx=8.0*SQR(xmu)-1.0;
		gam1=chebev(c1,NUSE1,xx);
		gam2=chebev(c2,NUSE2,xx);
		gampl= gam2-xmu*gam1;
		gammi= gam2+xmu*gam1;
		ff=2.0/PI*fact*(gam1*cosh(e)+gam2*fact2*d);
		e=exp(e);
		p=e/(gampl*PI);
		q=1.0/(e*PI*gammi);
		pimu2=0.5*pimu;
		fact3 = (abs(pimu2) < EPS ? 1.0 : sin(pimu2)/pimu2);
		r=PI*pimu2*fact3*fact3;
		c=1.0;
		d = -x2*x2;
		sum=ff+r*q;
		sum1=p;
		for (i=1;i<=MAXIT;i++) {
			ff=(i*ff+p+q)/(i*i-xmu2);
			c *= (d/i);
			p /= (i-xmu);
			q /= (i+xmu);
			del=c*(ff+r*q);
			sum += del;
			del1=c*p-i*del;
			sum1 += del1;
			if (abs(del) < (1.0+abs(sum))*EPS) break;
		}
		if (i > MAXIT) throw("bessy series failed to converge");
		rymu = -sum;
		ry1 = -sum1*xi2;
		rymup=xmu*xi*rymu-ry1;
		rjmu=w/(rymup-f*rymu);
	} else {
		a=0.25-xmu2;
		p = -0.5*xi;
		q=1.0;
		br=2.0*x;
		bi=2.0;
		fact=a*xi/(p*p+q*q);
		cr=br+q*fact;
		ci=bi+p*fact;
		den=br*br+bi*bi;
		dr=br/den;
		di = -bi/den;
		dlr=cr*dr-ci*di;
		dli=cr*di+ci*dr;
		temp=p*dlr-q*dli;
		q=p*dli+q*dlr;
		p=temp;
		for (i=1;i<MAXIT;i++) {
			a += 2*i;
			bi += 2.0;
			dr=a*dr+br;
			di=a*di+bi;
			if (abs(dr)+abs(di) < FPMIN) dr=FPMIN;
			fact=a/(cr*cr+ci*ci);
			cr=br+cr*fact;
			ci=bi-ci*fact;
			if (abs(cr)+abs(ci) < FPMIN) cr=FPMIN;
			den=dr*dr+di*di;
			dr /= den;
			di /= -den;
			dlr=cr*dr-ci*di;
			dli=cr*di+ci*dr;
			temp=p*dlr-q*dli;
			q=p*dli+q*dlr;
			p=temp;
			if (abs(dlr-1.0)+abs(dli) <= EPS) break;
		}
		if (i >= MAXIT) throw("cf2 failed in besseljy");
		gam=(p-f)/q;
		rjmu=sqrt(w/((p-f)*gam+q));
		rjmu=SIGN(rjmu,rjl);
		rymu=rjmu*gam;
		rymup=rymu*(p+q/gam);
		ry1=xmu*xi*rymu-rymup;
	}
	fact=rjmu/rjl;
	jo=rjl1*fact;
	jpo=rjp1*fact;
	for (i=1;i<=nl;i++) {
		rytemp=(xmu+i)*xi2*ry1-rymu;
		rymu=ry1;
		ry1=rytemp;
	}
	yo=rymu;
	ypo=nu*xi*rymu-ry1;
	xo = x;
	nuo = nu;
}
void Bessel::besselik(const Doub nu, const Doub x)
{
	const Int MAXIT=10000;
	const Doub EPS=numeric_limits<Doub>::epsilon();
	const Doub FPMIN=numeric_limits<Doub>::min()/EPS;
	const Doub XMIN=2.0, PI=3.141592653589793;
	Doub a,a1,b,c,d,del,del1,delh,dels,e,f,fact,fact2,ff,gam1,gam2,
		gammi,gampl,h,p,pimu,q,q1,q2,qnew,ril,ril1,rimu,rip1,ripl,
		ritemp,rk1,rkmu,rkmup,rktemp,s,sum,sum1,x2,xi,xi2,xmu,xmu2,xx;
	Int i,l,nl;
	if (x <= 0.0 || nu < 0.0) throw("bad arguments in besselik");
	nl=Int(nu+0.5);
	xmu=nu-nl;
	xmu2=xmu*xmu;
	xi=1.0/x;
	xi2=2.0*xi;
	h=nu*xi;
	if (h < FPMIN) h=FPMIN;
	b=xi2*nu;
	d=0.0;
	c=h;
	for (i=0;i<MAXIT;i++) {
		b += xi2;
		d=1.0/(b+d);
		c=b+1.0/c;
		del=c*d;
		h=del*h;
		if (abs(del-1.0) <= EPS) break;
	}
	if (i >= MAXIT)
		throw("x too large in besselik; try asymptotic expansion");
	ril=FPMIN;
	ripl=h*ril;
	ril1=ril;
	rip1=ripl;
	fact=nu*xi;
	for (l=nl-1;l >= 0;l--) {
		ritemp=fact*ril+ripl;
		fact -= xi;
		ripl=fact*ritemp+ril;
		ril=ritemp;
	}
	f=ripl/ril;
	if (x < XMIN) {
		x2=0.5*x;
		pimu=PI*xmu;
		fact = (abs(pimu) < EPS ? 1.0 : pimu/sin(pimu));
		d = -log(x2);
		e=xmu*d;
		fact2 = (abs(e) < EPS ? 1.0 : sinh(e)/e);
		xx=8.0*SQR(xmu)-1.0;
		gam1=chebev(c1,NUSE1,xx);
		gam2=chebev(c2,NUSE2,xx);
		gampl= gam2-xmu*gam1;
		gammi= gam2+xmu*gam1;
		ff=fact*(gam1*cosh(e)+gam2*fact2*d);
		sum=ff;
		e=exp(e);
		p=0.5*e/gampl;
		q=0.5/(e*gammi);
		c=1.0;
		d=x2*x2;
		sum1=p;
		for (i=1;i<=MAXIT;i++) {
			ff=(i*ff+p+q)/(i*i-xmu2);
			c *= (d/i);
			p /= (i-xmu);
			q /= (i+xmu);
			del=c*ff;
			sum += del;
			del1=c*(p-i*ff);
			sum1 += del1;
			if (abs(del) < abs(sum)*EPS) break;
		}
		if (i > MAXIT) throw("bessk series failed to converge");
		rkmu=sum;
		rk1=sum1*xi2;
	} else {
		b=2.0*(1.0+x);
		d=1.0/b;
		h=delh=d;
		q1=0.0;
		q2=1.0;
		a1=0.25-xmu2;
		q=c=a1;
		a = -a1;
		s=1.0+q*delh;
		for (i=1;i<MAXIT;i++) {
			a -= 2*i;
			c = -a*c/(i+1.0);
			qnew=(q1-b*q2)/a;
			q1=q2;
			q2=qnew;
			q += c*qnew;
			b += 2.0;
			d=1.0/(b+a*d);
			delh=(b*d-1.0)*delh;
			h += delh;
			dels=q*delh;
			s += dels;
			if (abs(dels/s) <= EPS) break;
		}
		if (i >= MAXIT) throw("besselik: failure to converge in cf2");
		h=a1*h;
		rkmu=sqrt(PI/(2.0*x))*exp(-x)/s;
		rk1=rkmu*(xmu+x+0.5-h)*xi;
	}
	rkmup=xmu*xi*rkmu-rk1;
	rimu=xi/(f*rkmu-rkmup);
	io=(rimu*ril1)/ril;
	ipo=(rimu*rip1)/ril;
	for (i=1;i <= nl;i++) {
		rktemp=(xmu+i)*xi2*rk1+rkmu;
		rkmu=rk1;
		rk1=rktemp;
	}
	ko=rkmu;
	kpo=nu*xi*rkmu-rk1;
	xo = x;
	nuo = nu;
}
void Bessel::airy(const Doub x) {
	static const Doub PI=3.141592653589793238,
		ONOVRT=0.577350269189626,THR=1./3.,TWOTHR=2.*THR;
	Doub absx,rootx,z;
	absx=abs(x);
	rootx=sqrt(absx);
	z=TWOTHR*absx*rootx;
	if (x > 0.0) {
		besselik(THR,z);
		aio = rootx*ONOVRT*ko/PI;
		bio = rootx*(ko/PI+2.0*ONOVRT*io);
		besselik(TWOTHR,z);
		aipo = -x*ONOVRT*ko/PI;
		bipo = x*(ko/PI+2.0*ONOVRT*io);
	} else if (x < 0.0) {
		besseljy(THR,z);
		aio = 0.5*rootx*(jo-ONOVRT*yo);
		bio = -0.5*rootx*(yo+ONOVRT*jo);
		besseljy(TWOTHR,z);
		aipo = 0.5*absx*(ONOVRT*yo+jo);
		bipo = 0.5*absx*(ONOVRT*jo-yo);
	} else {
		aio=0.355028053887817;
		bio=aio/ONOVRT;
		aipo = -0.258819403792807;
		bipo = -aipo/ONOVRT;
	}
}

Doub Bessel::airy_ai(const Doub x) {
	if (x != xo) airy(x);
	return aio;
}
Doub Bessel::airy_bi(const Doub x) {
	if (x != xo) airy(x);
	return bio;
}
void Bessel::sphbes(const Int n, const Doub x) {
	const Doub RTPIO2=1.253314137315500251;
	Doub factor,order;
	if (n < 0 || x <= 0.0) throw("bad arguments in sphbes");
	order=n+0.5;
	besseljy(order,x);
	factor=RTPIO2/sqrt(x);
	sphjo=factor*jo;
	sphyo=factor*yo;
	sphjpo=factor*jpo-sphjo/(2.*x);
	sphypo=factor*ypo-sphyo/(2.*x);
	sphno = n;
}

Doub Bessel::sphbesj(const Int n, const Doub x) {
	if (n != sphno || x != xo) sphbes(n,x);
	return sphjo;
}
Doub Bessel::sphbesy(const Int n, const Doub x) {
	if (n != sphno || x != xo) sphbes(n,x);
	return sphyo;
}
