struct Kmeans {
	Int nn, mm, kk, nchg;
	MatDoub data, means;
	VecInt assign, count;
	Kmeans(MatDoub &ddata, MatDoub &mmeans) : nn(ddata.nrows()), mm(ddata.ncols()),
	kk(mmeans.nrows()), data(ddata), means(mmeans), assign(nn), count(kk) {
		estep();
		mstep();
	}
	Int estep() {
		Int k,m,n,kmin;
		Doub dmin,d;
		nchg = 0;
		for (k=0;k<kk;k++) count[k] = 0;
		for (n=0;n<nn;n++) {
			dmin = 9.99e99;
			for (k=0;k<kk;k++) {
				for (d=0.,m=0; m<mm; m++) d += SQR(data[n][m]-means[k][m]);
				if (d < dmin) {dmin = d; kmin = k;}
			}
			if (kmin != assign[n]) nchg++;
			assign[n] = kmin;
			count[kmin]++;
		}
		return nchg;
	}
	void mstep() {
		Int n,k,m;
		for (k=0;k<kk;k++) for (m=0;m<mm;m++) means[k][m] = 0.;
		for (n=0;n<nn;n++) for (m=0;m<mm;m++) means[assign[n]][m] += data[n][m];
		for (k=0;k<kk;k++) {
			if (count[k] > 0) for (m=0;m<mm;m++) means[k][m] /= count[k];
		}
	}
};
