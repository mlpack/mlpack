struct Cauchydist {
	Doub mu, sig;
	Cauchydist(Doub mmu = 0., Doub ssig = 1.) : mu(mmu), sig(ssig) {
		if (sig <= 0.) throw("bad sig in Cauchydist");
	}
	Doub p(Doub x) {
		return 0.318309886183790671/(sig*(1.+SQR((x-mu)/sig)));
	}
	Doub cdf(Doub x) {
		return 0.5+0.318309886183790671*atan2(x-mu,sig);
	}
	Doub invcdf(Doub p) {
		if (p <= 0. || p >= 1.) throw("bad p in Cauchydist");
		return mu + sig*tan(3.14159265358979324*(p-0.5));
	}
};
struct Expondist {
	Doub bet;
	Expondist(Doub bbet) : bet(bbet) {
		if (bet <= 0.) throw("bad bet in Expondist");	
	}
	Doub p(Doub x) {
		if (x < 0.) throw("bad x in Expondist");
		return bet*exp(-bet*x);
	}
	Doub cdf(Doub x) {
		if (x < 0.) throw("bad x in Expondist");
		return 1.-exp(-bet*x);
	}
	Doub invcdf(Doub p) {
		if (p < 0. || p >= 1.) throw("bad p in Expondist");
		return -log(1.-p)/bet;
	}
};
struct Logisticdist {
	Doub mu, sig;
	Logisticdist(Doub mmu = 0., Doub ssig = 1.) : mu(mmu), sig(ssig) {
		if (sig <= 0.) throw("bad sig in Logisticdist");
	}
	Doub p(Doub x) {
		Doub e = exp(-abs(1.81379936423421785*(x-mu)/sig));
		return 1.81379936423421785*e/(sig*SQR(1.+e));
	}
	Doub cdf(Doub x) {
		Doub e = exp(-abs(1.81379936423421785*(x-mu)/sig));
		if (x >= mu) return 1./(1.+e);
		else return e/(1.+e);
	}
	Doub invcdf(Doub p) {
		if (p <= 0. || p >= 1.) throw("bad p in Logisticdist");
		return mu + 0.551328895421792049*sig*log(p/(1.-p));
	}
};
