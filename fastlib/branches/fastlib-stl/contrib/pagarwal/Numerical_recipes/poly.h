struct Poly {
	VecDoub &c;
	Poly(VecDoub &cc) : c(cc) {}
	Doub operator() (Doub x) {
		Int j;
		Doub p = c[j=c.size()-1];
		while (j>0) p = p*x + c[--j];
		return p;
	}
};
void ddpoly(VecDoub_I &c, const Doub x, VecDoub_O &pd)
{
	Int nnd,j,i,nc=c.size()-1,nd=pd.size()-1;
	Doub cnst=1.0;
	pd[0]=c[nc];
	for (j=1;j<nd+1;j++) pd[j]=0.0;
	for (i=nc-1;i>=0;i--) {
		nnd=(nd < (nc-i) ? nd : nc-i);
		for (j=nnd;j>0;j--) pd[j]=pd[j]*x+pd[j-1];
		pd[0]=pd[0]*x+c[i];
	}
	for (i=2;i<nd+1;i++) {
		cnst *= i;
		pd[i] *= cnst;
	}
}
void poldiv(VecDoub_I &u, VecDoub_I &v, VecDoub_O &q, VecDoub_O &r)
{
	Int k,j,n=u.size()-1,nv=v.size()-1;
	while (nv >= 0 && v[nv] == 0.) nv--;
	if (nv < 0) throw("poldiv divide by zero polynomial");
	r = u;
	q.assign(u.size(),0.);
	for (k=n-nv;k>=0;k--) {
		q[k]=r[nv+k]/v[nv];
		for (j=nv+k-1;j>=k;j--) r[j] -= q[k]*v[j-k];
	}
	for (j=nv;j<=n;j++) r[j]=0.0;
}
struct Ratfn {
	VecDoub cofs;
	Int nn,dd;

	Ratfn(VecDoub_I &num, VecDoub_I &den) : cofs(num.size()+den.size()-1),
	nn(num.size()), dd(den.size()) {
		Int j;
		for (j=0;j<nn;j++) cofs[j] = num[j]/den[0];
		for (j=1;j<dd;j++) cofs[j+nn-1] = den[j]/den[0];
	}

	Ratfn(VecDoub_I &coffs, const Int n, const Int d) : cofs(coffs), nn(n),
	dd(d) {}

	Doub operator() (Doub x) const {
		Int j;
		Doub sumn = 0., sumd = 0.;
		for (j=nn-1;j>=0;j--) sumn = sumn*x + cofs[j];
		for (j=nn+dd-2;j>=nn;j--) sumd = sumd*x + cofs[j];
		return sumn/(1.0+x*sumd);
	}

};
