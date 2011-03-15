void markovgen(const MatDoub_I &atrans, VecInt_O &out, Int istart=0,
	Int seed=1) {
	Int i, ilo, ihi, ii, j, m = atrans.nrows(), n = out.size();
	MatDoub cum(atrans);
	Doub r;
	Ran ran(seed);
	if (m != atrans.ncols()) throw("transition matrix must be square");
	for (i=0; i<m; i++) {
		for (j=1; j<m; j++) cum[i][j] += cum[i][j-1];
		if (abs(cum[i][m-1]-1.) > 0.01)
			throw("transition matrix rows must sum to 1");
	}
	j = istart;
	out[0] = j;
	for (ii=1; ii<n; ii++) {
		r = ran.doub()/cum[j][m-1];
		ilo = 0;
		ihi = m;
		while (ihi-ilo > 1) {
			i = (ihi+ilo) >> 1;
			if (r>cum[j][i-1]) ilo = i;
			else ihi = i;
		}
		out[ii] = j = ilo;
	}
}
