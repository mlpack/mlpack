struct Spectreg {
	Int m,m2,nsum;
	VecDoub specsum, wksp;

	Spectreg(Int em) : m(em), m2(2*m), nsum(0), specsum(m+1,0.), wksp(m2) {
		if (m & (m-1)) throw("m must be power of 2");
	}

	template<class D>
	void adddataseg(VecDoub_I &data, D &window) {
		Int i;
		Doub w,fac,sumw = 0.;
		if (data.size() != m2) throw("wrong size data segment");
		for (i=0;i<m2;i++) {
			w = window(i,m2);
			wksp[i] = w*data[i];
			sumw += SQR(w);
		}
		fac = 2./(sumw*m2);
		realft(wksp,1);
		specsum[0] += 0.5*fac*SQR(wksp[0]);
		for (i=1;i<m;i++) specsum[i] += fac*(SQR(wksp[2*i])+SQR(wksp[2*i+1]));
		specsum[m] += 0.5*fac*SQR(wksp[1]);
		nsum++;
	}

	VecDoub spectrum() {
		VecDoub spec(m+1);
		if (nsum == 0) throw("no data yet");
		for (Int i=0;i<=m;i++) spec[i] = specsum[i]/nsum;
		return spec;
	}

	VecDoub frequencies() {
		VecDoub freq(m+1);
		for (Int i=0;i<=m;i++) freq[i] = i*0.5/m;
		return freq;
	}
};
Doub square(Int j,Int n) {return 1.;}

Doub bartlett(Int j,Int n) {return 1.-abs(2.*j/(n-1.)-1.);}

Doub welch(Int j,Int n) {return 1.-SQR(2.*j/(n-1.)-1.);}

struct Hann {
	Int nn;
	VecDoub win;
	Hann(Int n) : nn(n), win(n) {
		Doub twopi = 8.*atan(1.);
		for (Int i=0;i<nn;i++) win[i] = 0.5*(1.-cos(i*twopi/(nn-1.)));
	}
	Doub operator() (Int j, Int n) {
		if (n != nn) throw("incorrect n for this Hann");
		return win[j];
	}
};
struct Spectolap : Spectreg {
	Int first;
	VecDoub fullseg;

	Spectolap(Int em) : Spectreg(em), first(1), fullseg(2*em) {}

	template<class D>
	void adddataseg(VecDoub_I &data, D &window) {
		Int i;
		if (data.size() != m) throw("wrong size data segment");
		if (first) {
			for (i=0;i<m;i++) fullseg[i+m] = data [i];
			first = 0;
		} else {
			for (i=0;i<m;i++) {
				fullseg[i] = fullseg[i+m];
				fullseg[i+m] = data [i];
			}
			Spectreg::adddataseg(fullseg,window);
		}
	}

	template<class D>
	void addlongdata(VecDoub_I &data, D &window) {
		Int i, k, noff, nt=data.size(), nk=(nt-1)/m;
		Doub del = nk > 1 ? (nt-m2)/(nk-1.) : 0.;
		if (nt < m2) throw("data length too short");
		for (k=0;k<nk;k++) {
			noff = (Int)(k*del+0.5);
			for (i=0;i<m2;i++) fullseg[i] = data[noff+i];
			Spectreg::adddataseg(fullseg,window);
		}
	}
};
struct Slepian  : Spectreg {
	Int jres, kt;
	MatDoub dpss;
	Doub p,pp,d,dd;
	Slepian(Int em, Int jjres, Int kkt)
	: Spectreg(em), jres(jjres), kt(kkt), dpss(kkt,2*em) {
		if (jres < 1 || kt >= 2*jres) throw("kt too big or jres too small");
		filltable();
	}
	void filltable();
	void renorm(Int n) {
		p = ldexp(p,n); pp = ldexp(pp,n); d = ldexp(d,n); dd = ldexp(dd,n);
	}
	struct Slepwindow {
		Int k;
		MatDoub &dps;
		Slepwindow(Int kkt, MatDoub &dpss) : k(kkt), dps(dpss) {}
		Doub operator() (Int j, Int n) {return dps[k][j];}
	};

	void adddataseg(VecDoub_I &data) {
		Int k;
		if (data.size() != m2) throw("wrong size data segment");
		for (k=0;k<kt;k++) {
			Slepwindow window(k,dpss);
			Spectreg::adddataseg(data,window);
		}
	}
};
void Slepian::filltable() {
	const Doub EPS = 1.e-10, PI = 4.*atan(1.);
	Doub xx,xnew,xold,sw,ppp,ddd,sum,bet,ssub,ssup,*u;
	Int i,j,k,nl;
	VecDoub dg(m2),dgg(m2),gam(m2),sup(m2-1),sub(m2-1);
	sw = 2.*SQR(sin(jres*PI/m2));
	dg[0] = 0.25*(2*m2+sw*SQR(m2-1.)-1.);
	for (i=1;i<m2;i++) {
		dg[i] = 0.25*(sw*SQR(m2-1.-2*i)+(2*(m2-i)-1.)*(2*i+1.));
		sub[i-1] = sup[i-1] = -i*(Doub)(m2-i)/2.;
	}
	xx = -0.10859 - 0.068762/jres + 1.5692*jres;
	xold = xx + 0.47276 + 0.20273/jres - 3.1387*jres;
	for (k=0; k<kt; k++) {
		u = &dpss[k][0];
		for (i=0;i<20;i++) {
			pp = 1.;
			p = dg[0] - xx;
			dd = 0.;
			d = -1.;
			for (j=1; j<m2; j++) {
				ppp = pp; pp = p;
				ddd = dd; dd = d;
				p = pp*(dg[j]-xx) - ppp*SQR(sup[j-1]);
				d = -pp + dd*(dg[j]-xx) - ddd*SQR(sup[j-1]);
				if (abs(p)>1.e30) renorm(-100);
				else if (abs(p)<1.e-30) renorm(100);
			}
			xnew = xx - p/d;
			if (abs(xx-xnew) < EPS*abs(xnew)) break;
			xx = xnew;
		}
		xx = xnew - (xold - xnew);
		xold = xnew;
		for (i=0;i<m2;i++) dgg[i] = dg[i] - xnew;
		nl = m2/3;
		dgg[nl] = 1.;
		ssup = sup[nl]; ssub = sub[nl-1];
		u[0] = sup[nl] = sub[nl-1] = 0.;
		bet = dgg[0];
		for (i=1; i<m2; i++) {
			gam[i] = sup[i-1]/bet;
			bet = dgg[i] - sub[i-1]*gam[i];
			u[i] = ((i==nl? 1. : 0.) - sub[i-1]*u[i-1])/bet;
		}
		for (i=m2-2; i>=0; i--) u[i] -= gam[i+1]*u[i+1];
		sup[nl] = ssup; sub[nl-1] = ssub;
		sum = 0.;
		for (i=0; i<m2; i++) sum += SQR(u[i]);
		sum = (u[3] > 0.)? sqrt(sum) : -sqrt(sum);
		for (i=0; i<m2; i++) u[i] /= sum;
	}
}
