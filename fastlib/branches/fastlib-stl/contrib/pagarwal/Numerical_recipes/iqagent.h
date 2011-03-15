struct IQagent {
	static const Int nbuf = 1000;
	Int nq, nt, nd;
	VecDoub pval,dbuf,qile;
	Doub q0, qm;

	IQagent() : nq(251), nt(0), nd(0), pval(nq), dbuf(nbuf),
	qile(nq,0.), q0(1.e99), qm(-1.e99) {
		for (Int j=85;j<=165;j++) pval[j] = (j-75.)/100.;
		for (Int j=84;j>=0;j--) {
			pval[j] = 0.87191909*pval[j+1];
			pval[250-j] = 1.-pval[j];
		}
	}
		
	void add(Doub datum) {
		dbuf[nd++] = datum;
		if (datum < q0) {q0 = datum;}
		if (datum > qm) {qm = datum;}
		if (nd == nbuf) update();
	}

	void update() {
		Int jd=0,jq=1,iq;
		Doub target, told=0., tnew=0., qold, qnew;
		VecDoub newqile(nq);
		sort(dbuf,nd);
		qold = qnew = qile[0] = newqile[0] = q0;
		qile[nq-1] = newqile[nq-1] = qm;
		pval[0] = min(0.5/(nt+nd),0.5*pval[1]);
		pval[nq-1] = max(1.-0.5/(nt+nd),0.5*(1.+pval[nq-2]));
		for (iq=1;iq<nq-1;iq++) {
			target = (nt+nd)*pval[iq];
			if (tnew < target) for (;;) {
				if (jq < nq && (jd >= nd || qile[jq] < dbuf[jd])) {
					qnew = qile[jq];
					tnew = jd + nt*pval[jq++];
					if (tnew >= target) break;
				} else {
					qnew = dbuf[jd];
					tnew = told;
					if (qile[jq]>qile[jq-1]) tnew += nt*(pval[jq]-pval[jq-1])
						*(qnew-qold)/(qile[jq]-qile[jq-1]);
					jd++;
					if (tnew >= target) break;
					told = tnew++;
					qold = qnew;
					if (tnew >= target) break;
				}
				told = tnew;
				qold = qnew;
			}
			if (tnew == told) newqile[iq] = 0.5*(qold+qnew);
			else newqile[iq] = qold + (qnew-qold)*(target-told)/(tnew-told);
			told = tnew;
			qold = qnew;
		}
		qile = newqile;
		nt += nd;
		nd = 0;
	}

	Doub report(Doub p) {
		Doub q;
		if (nd > 0) update();
		Int jl=0,jh=nq-1,j;
		while (jh-jl>1) {
			j = (jh+jl)>>1;
			if (p > pval[j]) jl=j;
			else jh=j;
		}
		j = jl;
		q =  qile[j] + (qile[j+1]-qile[j])*(p-pval[j])/(pval[j+1]-pval[j]);
		return MAX(qile[0],MIN(qile[nq-1],q));
	}
};
