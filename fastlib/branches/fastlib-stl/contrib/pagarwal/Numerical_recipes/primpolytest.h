struct Primpolytest {
	Int N, nfactors;
	VecUllong factors;
	VecInt t,a,p;

	Primpolytest() : N(32), nfactors(5), factors(nfactors), t(N*N),
		a(N*N), p(N*N) {
		Ullong factordata[5] = {3,5,17,257,65537};
		for (Int i=0;i<nfactors;i++) factors[i] = factordata[i];
	}

	Int ispident() {
		Int i,j;
		for (i=0; i<N; i++) for (j=0; j<N; j++) {
			if (i == j) { if (p[i*N+j] != 1) return 0; }
			else {if (p[i*N+j] != 0) return 0; }
		}
		return 1;
	}

	void mattimeseq(VecInt &a, VecInt &b) {
		Int i,j,k,sum;
		VecInt tmp(N*N);
		for (i=0; i<N; i++) for (j=0; j<N; j++) {
			sum = 0;
			for (k=0; k<N; k++) sum += a[i*N+k] * b[k*N+j];
			tmp[i*N+j] = sum & 1;
		}
		for (k=0; k<N*N; k++) a[k] = tmp[k];
	}

	void matpow(Ullong n) {
		Int k;
		for (k=0; k<N*N; k++) p[k] = 0;
		for (k=0; k<N; k++) p[k*N+k] = 1;
		while (1) {
			if (n & 1) mattimeseq(p,a);
			n >>= 1;
			if (n == 0) break;
			mattimeseq(a,a);
		}
	}

	Int test(Ullong n) {
		Int i,k,j;
		Ullong pow, tnm1, nn = n;
		tnm1 = ((Ullong)1 << N) - 1;
		if (n > (tnm1 >> 1)) throw("not a polynomial of degree N");
		for (k=0; k<N*N; k++) t[k] = 0;
		for (i=1; i<N; i++) t[i*N+(i-1)] = 1;
		j=0;
		while (nn) {
			if (nn & 1) t[j] = 1;
			nn >>= 1;
			j++;
		}
		t[N-1] = 1;
		for (k=0; k<N*N; k++) a[k] = t[k];
		matpow(tnm1);
		if (ispident() != 1) return 0;
		for (i=0; i<nfactors; i++) {
			pow = tnm1/factors[i];
			for (k=0; k<N*N; k++) a[k] = t[k];
			matpow(pow);
			if (ispident() == 1) return 0;
		}
		return 1;
	}
};
