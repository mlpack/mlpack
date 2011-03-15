void spread(const Doub y, VecDoub_IO &yy, const Doub x, const Int m) {
	static Int nfac[11]={0,1,1,2,6,24,120,720,5040,40320,362880};
	Int ihi,ilo,ix,j,nden,n=yy.size();
	Doub fac;
	if (m > 10) throw("factorial table too small in spread");
	ix=Int(x);
	if (x == Doub(ix)) yy[ix-1] += y;
	else {
		ilo=MIN(MAX(Int(x-0.5*m),0),Int(n-m));
		ihi=ilo+m;
		nden=nfac[m];
		fac=x-ilo-1;
		for (j=ilo+1;j<ihi;j++) fac *= (x-j-1);
		yy[ihi-1] += y*fac/(nden*(x-ihi));
		for (j=ihi-1;j>ilo;j--) {
			nden=(nden/(j-ilo))*(j-ihi);
			yy[j-1] += y*fac/(nden*(x-j));
		}
	}
}
void fasper(VecDoub_I &x, VecDoub_I &y, const Doub ofac, const Doub hifac,
	VecDoub_O &px, VecDoub_O &py, Int &nout, Int &jmax, Doub &prob) {
	const Int MACC=4;
	Int j,k,nwk,nfreq,nfreqt,n=x.size(),np=px.size();
	Doub ave,ck,ckk,cterm,cwt,den,df,effm,expy,fac,fndim,hc2wt,hs2wt,
		hypo,pmax,sterm,swt,var,xdif,xmax,xmin;
	nout=Int(0.5*ofac*hifac*n);
	nfreqt=Int(ofac*hifac*n*MACC);
	nfreq=64;
	while (nfreq < nfreqt) nfreq <<= 1;
	nwk=nfreq << 1;
	if (np < nout) {px.resize(nout); py.resize(nout);}
	avevar(y,ave,var);
	if (var == 0.0) throw("zero variance in fasper");
	xmin=x[0];
	xmax=xmin;
	for (j=1;j<n;j++) {
		if (x[j] < xmin) xmin=x[j];
		if (x[j] > xmax) xmax=x[j];
	}
	xdif=xmax-xmin;
	VecDoub wk1(nwk,0.);
	VecDoub wk2(nwk,0.);
	fac=nwk/(xdif*ofac);
	fndim=nwk;
	for (j=0;j<n;j++) {
		ck=fmod((x[j]-xmin)*fac,fndim);
		ckk=2.0*(ck++);
		ckk=fmod(ckk,fndim);
		++ckk;
		spread(y[j]-ave,wk1,ck,MACC);
		spread(1.0,wk2,ckk,MACC);
	}
	realft(wk1,1);
	realft(wk2,1);
	df=1.0/(xdif*ofac);
	pmax = -1.0;
	for (k=2,j=0;j<nout;j++,k+=2) {
		hypo=sqrt(wk2[k]*wk2[k]+wk2[k+1]*wk2[k+1]);
		hc2wt=0.5*wk2[k]/hypo;
		hs2wt=0.5*wk2[k+1]/hypo;
		cwt=sqrt(0.5+hc2wt);
		swt=SIGN(sqrt(0.5-hc2wt),hs2wt);
		den=0.5*n+hc2wt*wk2[k]+hs2wt*wk2[k+1];
		cterm=SQR(cwt*wk1[k]+swt*wk1[k+1])/den;
		sterm=SQR(cwt*wk1[k+1]-swt*wk1[k])/(n-den);
		px[j]=(j+1)*df;
		py[j]=(cterm+sterm)/(2.0*var);
		if (py[j] > pmax) pmax=py[jmax=j];
	}
	expy=exp(-pmax);
	effm=2.0*nout/ofac;
	prob=effm*expy;
	if (prob > 0.01) prob=1.0-pow(1.0-expy,effm);
}
