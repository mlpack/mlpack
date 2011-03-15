struct State {
	Doub lam1, lam2;
	Doub tc;
	Int k1, k2;
	Doub plog;

	State(Doub la1, Doub la2, Doub t, Int kk1, Int kk2) :
		lam1(la1), lam2(la2), tc(t), k1(kk1), k2(kk2) {}
	State() {};
};
struct Plog {
	VecDoub &dat;
	Int ndat;
	VecDoub stau, slogtau;

	Plog(VecDoub &data) : dat(data), ndat(data.size()),
	stau(ndat), slogtau(ndat) {
		Int i;
		stau[0] = slogtau[0] = 0.;
		for (i=1;i<ndat;i++) {
			stau[i] = dat[i]-dat[0];	
			slogtau[i] = slogtau[i-1] + log(dat[i]-dat[i-1]);
		}
	}

	Doub operator() (State &s) {
		Int i,ilo,ihi,n1,n2;
		Doub st1,st2,stl1,stl2, ans;
		ilo = 0;
		ihi = ndat-1;
		while (ihi-ilo>1) {
			i = (ihi+ilo) >> 1;
			if (s.tc > dat[i]) ilo=i;
			else ihi=i;
		}
		n1 = ihi;
		n2 = ndat-1-ihi;
		st1 = stau[ihi];
		st2 = stau[ndat-1]-st1;
		stl1 = slogtau[ihi];
		stl2 = slogtau[ndat-1]-stl1;
		ans =  n1*(s.k1*log(s.lam1)-factln(s.k1-1))+(s.k1-1)*stl1-s.lam1*st1;
		ans += n2*(s.k2*log(s.lam2)-factln(s.k2-1))+(s.k2-1)*stl2-s.lam2*st2;
		return (s.plog = ans);
	}
};
struct Proposal {
	Normaldev gau;
	Doub logstep;

	Proposal(Int ranseed, Doub lstep) : gau(0.,1.,ranseed), logstep(lstep) {}

	void operator() (const State &s1, State &s2, Doub &qratio) {
		Doub r=gau.doub();
		if (r < 0.9) {
			s2.lam1 = s1.lam1 * exp(logstep*gau.dev());
			s2.lam2 = s1.lam2 * exp(logstep*gau.dev());
			s2.tc = s1.tc * exp(logstep*gau.dev());
			s2.k1 = s1.k1;
			s2.k2 = s1.k2;
			qratio = (s2.lam1/s1.lam1)*(s2.lam2/s1.lam2)*(s2.tc/s1.tc);
		} else {
			r=gau.doub();
			if (s1.k1>1) {
				if (r<0.5) s2.k1 = s1.k1;
				else if (r<0.75) s2.k1 = s1.k1 + 1;
				else s2.k1 = s1.k1 - 1;
			} else {
				if (r<0.75) s2.k1 = s1.k1;
				else s2.k1 = s1.k1 + 1;
			}
			s2.lam1 = s2.k1*s1.lam1/s1.k1;
			r=gau.doub();
			if (s1.k2>1) {
				if (r<0.5) s2.k2 = s1.k2;
				else if (r<0.75) s2.k2 = s1.k2 + 1;
				else s2.k2 = s1.k2 - 1;
			} else {
				if (r<0.75) s2.k2 = s1.k2;
				else s2.k2 = s1.k2 + 1;
			}
			s2.lam2 = s2.k2*s1.lam2/s1.k2;
			s2.tc = s1.tc;
			qratio = 1.;
		}
	}
};
Doub mcmcstep(Int m, State &s, Plog &plog, Proposal &propose) {
	State sprop;
	Doub qratio,alph,ran;
	Int accept=0;
	plog(s);
	for (Int i=0;i<m;i++) {
		propose(s,sprop,qratio);
		alph = min(1.,qratio*exp(plog(sprop)-s.plog));
		ran = propose.gau.doub();
		if (ran < alph) {
			s = sprop;
			plog(s);
			accept++;
		}
	}
	return accept/Doub(m);
}
