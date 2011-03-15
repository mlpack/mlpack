void usrfun(VecDoub_I &x, VecDoub_O &fvec, MatDoub_O &fjac);

void mnewt(const Int ntrial, VecDoub_IO &x, const Doub tolx, const Doub tolf) {
	Int i,n=x.size();
	VecDoub p(n),fvec(n);
	MatDoub fjac(n,n);
	for (Int k=0;k<ntrial;k++) {
		usrfun(x,fvec,fjac);
		Doub errf=0.0;
		for (i=0;i<n;i++) errf += abs(fvec[i]);
		if (errf <= tolf) return;
		for (i=0;i<n;i++) p[i] = -fvec[i];
		LUdcmp alu(fjac);
		alu.solve(p,p);
		Doub errx=0.0;
		for (i=0;i<n;i++) {
			errx += abs(p[i]);
			x[i] += p[i];
		}
		if (errx <= tolx) return;
	}
	return;
}
