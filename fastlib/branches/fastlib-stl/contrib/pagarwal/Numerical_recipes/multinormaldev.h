struct Multinormaldev : Ran {
	Int mm;
	VecDoub mean;
	MatDoub var;
	Cholesky chol;
	VecDoub spt, pt;

	Multinormaldev(Ullong j, VecDoub &mmean, MatDoub &vvar) :
	Ran(j), mm(mmean.size()), mean(mmean), var(vvar), chol(var),
	spt(mm), pt(mm) {
		if (var.ncols() != mm || var.nrows() != mm) throw("bad sizes");
	}

	VecDoub &dev() {
		Int i;
		Doub u,v,x,y,q;
		for (i=0;i<mm;i++) {
			do {
				u = doub();
				v = 1.7156*(doub()-0.5);
				x = u - 0.449871;
				y = abs(v) + 0.386595;
				q = SQR(x) + y*(0.19600*y-0.25472*x);
			} while (q > 0.27597
				&& (q > 0.27846 || SQR(v) > -4.*log(u)*SQR(u)));
			spt[i] = v/u;
		}
		chol.elmult(spt,pt);
		for (i=0;i<mm;i++) {pt[i] += mean[i];}
		return pt;
	}

};
