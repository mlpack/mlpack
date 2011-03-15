struct Expondev : Ran {
	Doub beta;
	Expondev(Doub bbeta, Ullong i) : Ran(i), beta(bbeta) {}
	Doub dev() {
		Doub u;
		do u = doub(); while (u == 0.);
		return -log(u)/beta;
	}
};
struct Logisticdev : Ran {
	Doub mu,sig;
	Logisticdev(Doub mmu, Doub ssig, Ullong i) : Ran(i), mu(mmu), sig(ssig) {}
	Doub dev() {
		Doub u;
		do u = doub(); while (u*(1.-u) == 0.);
		return mu + 0.551328895421792050*sig*log(u/(1.-u));
	}
};
struct Normaldev_BM : Ran {
	Doub mu,sig;
	Doub storedval;
	Normaldev_BM(Doub mmu, Doub ssig, Ullong i)
	: Ran(i), mu(mmu), sig(ssig), storedval(0.) {}
	Doub dev() {
		Doub v1,v2,rsq,fac;
		if (storedval == 0.) {
			do {
				v1=2.0*doub()-1.0;
				v2=2.0*doub()-1.0;
				rsq=v1*v1+v2*v2;
			} while (rsq >= 1.0 || rsq == 0.0);
			fac=sqrt(-2.0*log(rsq)/rsq);
			storedval = v1*fac;
			return mu + sig*v2*fac;
		} else {
			fac = storedval;
			storedval = 0.;
			return mu + sig*fac;
		}
	}
};
struct Cauchydev : Ran {
	Doub mu,sig;
	Cauchydev(Doub mmu, Doub ssig, Ullong i) : Ran(i), mu(mmu), sig(ssig) {}
	Doub dev() {
		Doub v1,v2;
		do {
			v1=2.0*doub()-1.0;
			v2=doub();
		} while (SQR(v1)+SQR(v2) >= 1. || v2 == 0.);
		return mu + sig*v1/v2;
	}
};
struct Normaldev : Ran {
	Doub mu,sig;
	Normaldev(Doub mmu, Doub ssig, Ullong i)
	: Ran(i), mu(mmu), sig(ssig){}
	Doub dev() {
		Doub u,v,x,y,q;
		do {
			u = doub();
			v = 1.7156*(doub()-0.5);
			x = u - 0.449871;
			y = abs(v) + 0.386595;
			q = SQR(x) + y*(0.19600*y-0.25472*x);
		} while (q > 0.27597
			&& (q > 0.27846 || SQR(v) > -4.*log(u)*SQR(u)));
		return mu + sig*v/u;
	}
};
struct Gammadev : Normaldev {
	Doub alph, oalph, bet;
	Doub a1,a2;
	Gammadev(Doub aalph, Doub bbet, Ullong i)
	: Normaldev(0.,1.,i), alph(aalph), oalph(aalph), bet(bbet) {
		if (alph <= 0.) throw("bad alph in Gammadev");
		if (alph < 1.) alph += 1.;
		a1 = alph-1./3.;
		a2 = 1./sqrt(9.*a1);
	}
	Doub dev() {
		Doub u,v,x;
		do {
			do {
				x = Normaldev::dev();
				v = 1. + a2*x;
			} while (v <= 0.);
			v = v*v*v;
			u = doub();
		} while (u > 1. - 0.331*SQR(SQR(x)) &&
			log(u) > 0.5*SQR(x) + a1*(1.-v+log(v)));
		if (alph == oalph) return a1*v/bet;
		else {
			do u=doub(); while (u == 0.);
			return pow(u,1./oalph)*a1*v/bet;
		}
	}
};
struct Poissondev : Ran {
	Doub lambda, sqlam, loglam, lamexp, lambold;
	VecDoub logfact;
	Int swch;
	Poissondev(Doub llambda, Ullong i) : Ran(i), lambda(llambda),
		logfact(1024,-1.), lambold(-1.) {}
	Int dev() {
		Doub u,u2,v,v2,p,t,lfac;
		Int k;
		if (lambda < 5.) {
			if (lambda != lambold) lamexp=exp(-lambda);
			k = -1;
			t=1.;
			do {
				++k;
				t *= doub();
			} while (t > lamexp);
		} else {
			if (lambda != lambold) {
				sqlam = sqrt(lambda);
				loglam = log(lambda);
			}
			for (;;) {
				u = 0.64*doub();
				v = -0.68 + 1.28*doub();
				if (lambda > 13.5) {
					v2 = SQR(v);
					if (v >= 0.) {if (v2 > 6.5*u*(0.64-u)*(u+0.2)) continue;}
					else {if (v2 > 9.6*u*(0.66-u)*(u+0.07)) continue;}
				}
				k = Int(floor(sqlam*(v/u)+lambda+0.5));
				if (k < 0) continue;
				u2 = SQR(u);
				if (lambda > 13.5) {
					if (v >= 0.) {if (v2 < 15.2*u2*(0.61-u)*(0.8-u)) break;}
					else {if (v2 < 6.76*u2*(0.62-u)*(1.4-u)) break;}
				}
				if (k < 1024) {
					if (logfact[k] < 0.) logfact[k] = gammln(k+1.);
					lfac = logfact[k];
				} else lfac = gammln(k+1.);
				p = sqlam*exp(-lambda + k*loglam - lfac);
				if (u2 < p) break;
			}
		}
		lambold = lambda;
		return k;
	}
	Int dev(Doub llambda) {
		lambda = llambda;
		return dev();
	}
};
struct Binomialdev : Ran {
	Doub pp,p,pb,expnp,np,glnp,plog,pclog,sq;
	Int n,swch;
	Ullong uz,uo,unfin,diff,rltp;
	Int pbits[5];
	Doub cdf[64];
	Doub logfact[1024];
	Binomialdev(Int nn, Doub ppp, Ullong i) : Ran(i), pp(ppp), n(nn) {
		Int j;
		pb = p = (pp <= 0.5 ? pp : 1.0-pp);
		if (n <= 64) {
			uz=0;
			uo=0xffffffffffffffffLL;
			rltp = 0;
			for (j=0;j<5;j++) pbits[j] = 1 & ((Int)(pb *= 2.));
			pb -= floor(pb);
			swch = 0;
		} else if (n*p < 30.) {
			cdf[0] = exp(n*log(1-p));
			for (j=1;j<64;j++) cdf[j] =  cdf[j-1] + exp(gammln(n+1.)
				-gammln(j+1.)-gammln(n-j+1.)+j*log(p)+(n-j)*log(1.-p));
			swch = 1;
		} else {
			np = n*p;
			glnp=gammln(n+1.);
			plog=log(p);
			pclog=log(1.-p);
			sq=sqrt(np*(1.-p));
			if (n < 1024) for (j=0;j<=n;j++) logfact[j] = gammln(j+1.);
			swch = 2;
		}
	}	
	Int dev() {
		Int j,k,kl,km;
		Doub y,u,v,u2,v2,b;
		if (swch == 0) {
			unfin = uo;
			for (j=0;j<5;j++) {
				diff = unfin & (int64()^(pbits[j]? uo : uz));
				if (pbits[j]) rltp |= diff;
				else rltp = rltp & ~diff;
				unfin = unfin & ~diff;
			}
			k=0;
			for (j=0;j<n;j++) {
				if (unfin & 1) {if (doub() < pb) ++k;}
				else {if (rltp & 1) ++k;}
				unfin >>= 1;
				rltp >>= 1;
			}
		} else if (swch == 1) {
			y = doub();
			kl = -1;
			k = 64;
			while (k-kl>1) {
				km = (kl+k)/2;
				if (y < cdf[km]) k = km;
				else kl = km;
			}
		} else {
			for (;;) {
				u = 0.645*doub();
				v = -0.63 + 1.25*doub();
				v2 = SQR(v);
				if (v >= 0.) {if (v2 > 6.5*u*(0.645-u)*(u+0.2)) continue;}
				else {if (v2 > 8.4*u*(0.645-u)*(u+0.1)) continue;}
				k = Int(floor(sq*(v/u)+np+0.5));
				if (k < 0) continue;
				u2 = SQR(u);
				if (v >= 0.) {if (v2 < 12.25*u2*(0.615-u)*(0.92-u)) break;}
				else {if (v2 < 7.84*u2*(0.615-u)*(1.2-u)) break;}
				b = sq*exp(glnp+k*plog+(n-k)*pclog
					- (n < 1024 ? logfact[k]+logfact[n-k]
						: gammln(k+1.)+gammln(n-k+1.)));
				if (u2 < b) break;
			}
		}
		if (p != pp) k = n - k;
		return k;
	}
};
