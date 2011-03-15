template <class T>
void dfpmin(VecDoub_IO &p, const Doub gtol, Int &iter, Doub &fret, T &funcd)
{
	const Int ITMAX=200;
	const Doub EPS=numeric_limits<Doub>::epsilon();
	const Doub TOLX=4*EPS,STPMX=100.0;
	Bool check;
	Doub den,fac,fad,fae,fp,stpmax,sum=0.0,sumdg,sumxi,temp,test;
	Int n=p.size();
	VecDoub dg(n),g(n),hdg(n),pnew(n),xi(n);
	MatDoub hessin(n,n);
	fp=funcd(p);
	funcd.df(p,g);
	for (Int i=0;i<n;i++) {
		for (Int j=0;j<n;j++) hessin[i][j]=0.0;
		hessin[i][i]=1.0;
		xi[i] = -g[i];
		sum += p[i]*p[i];
	}
	stpmax=STPMX*MAX(sqrt(sum),Doub(n));
	for (Int its=0;its<ITMAX;its++) {
		iter=its;
		lnsrch(p,fp,g,xi,pnew,fret,stpmax,check,funcd);
		fp=fret;
		for (Int i=0;i<n;i++) {
			xi[i]=pnew[i]-p[i];
			p[i]=pnew[i];
		}
		test=0.0;
		for (Int i=0;i<n;i++) {
			temp=abs(xi[i])/MAX(abs(p[i]),1.0);
			if (temp > test) test=temp;
		}
		if (test < TOLX)
			return;
		for (Int i=0;i<n;i++) dg[i]=g[i];
		funcd.df(p,g);
		test=0.0;
		den=MAX(fret,1.0);
		for (Int i=0;i<n;i++) {
			temp=abs(g[i])*MAX(abs(p[i]),1.0)/den;
			if (temp > test) test=temp;
		}
		if (test < gtol)
			return;
		for (Int i=0;i<n;i++)
			dg[i]=g[i]-dg[i];
		for (Int i=0;i<n;i++) {
			hdg[i]=0.0;
			for (Int j=0;j<n;j++) hdg[i] += hessin[i][j]*dg[j];
		}
		fac=fae=sumdg=sumxi=0.0;
		for (Int i=0;i<n;i++) {
			fac += dg[i]*xi[i];
			fae += dg[i]*hdg[i];
			sumdg += SQR(dg[i]);
			sumxi += SQR(xi[i]);
		}
		if (fac > sqrt(EPS*sumdg*sumxi)) {
			fac=1.0/fac;
			fad=1.0/fae;
			for (Int i=0;i<n;i++) dg[i]=fac*xi[i]-fad*hdg[i];
			for (Int i=0;i<n;i++) {
				for (Int j=i;j<n;j++) {
					hessin[i][j] += fac*xi[i]*xi[j]
						-fad*hdg[i]*hdg[j]+fae*dg[i]*dg[j];
					hessin[j][i]=hessin[i][j];
				}
			}
		}
		for (Int i=0;i<n;i++) {
			xi[i]=0.0;
			for (Int j=0;j<n;j++) xi[i] -= hessin[i][j]*g[j];
		}
	}
	throw("too many iterations in dfpmin");
}
template <class T>
struct Funcd {
	Doub EPS;
	T &func;
	Doub f;
	Funcd(T &funcc) : EPS(1.0e-8), func(funcc) {}
	Doub operator() (VecDoub_I &x)
	{
		return f=func(x);
	}

	void df(VecDoub_I &x, VecDoub_O &df)
	{
		Int n=x.size();
		VecDoub xh=x;
		Doub fold=f;
		for (Int j=0;j<n;j++) {
			Doub temp=x[j];
			Doub h=EPS*abs(temp);
			if (h == 0.0) h=EPS;
			xh[j]=temp+h;
			h=xh[j]-temp;
			Doub fh=operator()(xh);
			xh[j]=temp;
			df[j]=(fh-fold)/h;
		}
	}
};
