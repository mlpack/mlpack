struct MCintegrate {
	Int ndim,nfun,n;
	VecDoub ff,fferr;
	VecDoub xlo,xhi,x,xx,fn,sf,sferr;
	Doub vol;

	VecDoub (*funcsp)(const VecDoub &);
	VecDoub (*xmapp)(const VecDoub &);
	Bool (*inregionp)(const VecDoub &);
	Ran ran;

	MCintegrate(const VecDoub &xlow, const VecDoub &xhigh,
	VecDoub funcs(const VecDoub &), Bool inregion(const VecDoub &),
	VecDoub xmap(const VecDoub &), Int ranseed);

	void step(Int nstep);

	void calcanswers();
};
MCintegrate::MCintegrate(const VecDoub &xlow, const VecDoub &xhigh,
	VecDoub funcs(const VecDoub &), Bool inregion(const VecDoub &),
	VecDoub xmap(const VecDoub &), Int ranseed)
	: ndim(xlow.size()), n(0), xlo(xlow), xhi(xhigh), x(ndim), xx(ndim),
	funcsp(funcs), xmapp(xmap), inregionp(inregion), vol(1.), ran(ranseed) {
	if (xmapp) nfun = funcs(xmapp(xlo)).size();
	else nfun = funcs(xlo).size();
	ff.resize(nfun);
	fferr.resize(nfun);
	fn.resize(nfun);
	sf.assign(nfun,0.);
	sferr.assign(nfun,0.);
	for (Int j=0;j<ndim;j++) vol *= abs(xhi[j]-xlo[j]);
}

void MCintegrate::step(Int nstep) {
	Int i,j;
	for (i=0;i<nstep;i++) {
		for (j=0;j<ndim;j++)
			x[j] = xlo[j]+(xhi[j]-xlo[j])*ran.doub();
		if (xmapp) xx = (*xmapp)(x);
		else xx = x;
		if ((*inregionp)(xx)) {
			fn = (*funcsp)(xx);
			for (j=0;j<nfun;j++) {
				sf[j] += fn[j];
				sferr[j] += SQR(fn[j]);
			}
		}
	}
	n += nstep;
}

void MCintegrate::calcanswers(){
	for (Int j=0;j<nfun;j++) {
		ff[j] = vol*sf[j]/n;
		fferr[j] = vol*sqrt((sferr[j]/n-SQR(sf[j]/n))/n);
	}
}
VecDoub torusfuncs(const VecDoub &x) {
	Doub den = 1.;
	VecDoub f(4);
	f[0] = den;
	for (Int i=1;i<4;i++) f[i] = x[i-1]*den;
	return f;
}

Bool torusregion(const VecDoub &x) {
	return SQR(x[2])+SQR(sqrt(SQR(x[0])+SQR(x[1]))-3.) <= 1.;
}
VecDoub torusmap(const VecDoub &s) {
	VecDoub xx(s);
	xx[2] = 0.2*log(5.*s[2]);
	return xx;
}
