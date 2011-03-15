struct Factorial {
	static const Int NLTAB = 10000;
	VecDoub tab;
	VecDoub ltab;

	Factorial() : tab(171), ltab(NLTAB) {
		Int i;
		tab[0] = 1.;
		ltab[0] = 0.;
		for (i=1;i<171;i++) {
			tab[i] = i*tab[i-1];
			ltab[i] = log(tab[i]);
		}
		for (i=171;i<NLTAB;i++) ltab[i] = gammln(i+1.);
	}

	inline Doub fac(const Int i) {
		if (i<0 || i>170) throw("factorial overflows");
		return tab[i];
	}

	Doub facln(const Int i) {
		if (i<0) throw("negative facln arg");
		if (i<NLTAB) return ltab[i];
		return gammln(i+1.);
	}

	Doub bico(const Int n, const Int k) {
		if (n<0 || k<0 || k>n) throw("bico bad args");
		if (n<171) return floor(0.5+tab[n]/(tab[k]*tab[n-k]));
		if (n<NLTAB) return floor(0.5+exp(ltab[n]-ltab[k]-ltab[n-k]));
		return floor(0.5+exp(gammln(n+1.)-gammln(k+1.)-gammln(n-k+1.)));
	}
};
