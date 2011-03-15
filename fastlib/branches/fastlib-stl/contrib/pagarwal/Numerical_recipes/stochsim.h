void sparmatfill(NRvector<NRsparseCol> &sparmat, MatDoub &fullmat) {
	Int n,m,nz,nn=fullmat.nrows(),mm=fullmat.ncols();
	if (sparmat.size() != mm) throw("bad sizes");
	for (m=0;m<mm;m++) {
		for (nz=n=0;n<nn;n++) if (fullmat[n][m]) nz++;
		sparmat[m].resize(nn,nz);
		for (nz=n=0;n<nn;n++) if (fullmat[n][m]) {
			sparmat[m].row_ind[nz] = n;
			sparmat[m].val[nz++] = fullmat[n][m];
		}
	}
}
struct Stochsim {
	VecDoub s;
	VecDoub a;
	MatDoub instate, outstate;
	NRvector<NRsparseCol> outchg, depend;
	VecInt pr;
	Doub t, asum;
	Ran ran;
	typedef Doub(Stochsim::*rateptr)();
	rateptr *dispatch;

	// begin user section
	static const Int mm=3;
	static const Int nn=4;
	Doub k0,k1,k2;
	Doub rate0() {return k0*s[0]*s[1];}
	Doub rate1() {return k1*s[1]*s[2];}
	Doub rate2() {return k2*s[2];}
	void describereactions () {
		k0 = 0.01;
		k1 = .1;
		k2 = 1.;
		Doub indat[] = {
			1.,0.,0.,
			1.,1.,0.,
			0.,1.,1.,
			0.,0.,0.
		};
		instate = MatDoub(nn,mm,indat);
		Doub outdat[] = {
			-1.,0.,0.,
			1.,-1.,0.,
			0.,1.,-1.,
			0.,0.,1.
		};
		outstate = MatDoub(nn,mm,outdat);
		dispatch[0] = &Stochsim::rate0;
		dispatch[1] = &Stochsim::rate1;
		dispatch[2] = &Stochsim::rate2;
	}
	// end user section

	Stochsim(VecDoub &sinit, Int seed=1)
	: s(sinit), a(mm,0.), outchg(mm), depend(mm), pr(mm), t(0.),
	asum(0.), ran(seed), dispatch(new rateptr[mm]) {
		Int i,j,k,d;
		describereactions();
		sparmatfill(outchg,outstate);
		MatDoub dep(mm,mm);
		for (i=0;i<mm;i++) for (j=0;j<mm;j++) {
			d = 0;
			for (k=0;k<nn;k++) d = d || (instate[k][i] && outstate[k][j]);
			dep[i][j] = d;
		}
		sparmatfill(depend,dep);
		for (i=0;i<mm;i++) {
			pr[i] = i;
			a[i] = (this->*dispatch[i])();
			asum += a[i];
		}
	}
	~Stochsim() {delete [] dispatch;}

	Doub step() {
		Int i,n,m,k=0;
		Doub tau,atarg,sum,anew;
		if (asum == 0.) {t *= 2.; return t;}
		tau = -log(ran.doub())/asum;
		atarg = ran.doub()*asum;
		sum = a[pr[0]];
		while (sum < atarg) sum += a[pr[++k]];
		m = pr[k];
		if (k > 0) SWAP(pr[k],pr[k-1]);
		if (k == mm-1) asum = sum;
		n = outchg[m].nvals;
		for (i=0;i<n;i++) {
			k = outchg[m].row_ind[i];
			s[k] += outchg[m].val[i];
		}
		n = depend[m].nvals;
		for (i=0;i<n;i++) {
			k = depend[m].row_ind[i];
			anew = (this->*dispatch[k])();
			asum += (anew - a[k]);
			a[k] = anew;
		}
		if (t*asum < 0.1)
			for (asum=0.,i=0;i<mm;i++) asum += a[i];		
		return (t += tau);
	}
};
