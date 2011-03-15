struct preGaumixmod {
	static Int mmstat;
	struct Mat_mm : MatDoub {Mat_mm() : MatDoub(mmstat,mmstat) {} };
	preGaumixmod(Int mm) {mmstat = mm;}
};
Int preGaumixmod::mmstat = -1;

struct Gaumixmod : preGaumixmod {
	Int nn, kk, mm;
	MatDoub data, means, resp;
	VecDoub frac, lndets;
	vector<Mat_mm> sig;
	Doub loglike;
	Gaumixmod(MatDoub &ddata, MatDoub &mmeans) : preGaumixmod(ddata.ncols()),
	nn(ddata.nrows()), kk(mmeans.nrows()), mm(mmstat), data(ddata), means(mmeans),
	resp(nn,kk), frac(kk), lndets(kk), sig(kk) {
		Int i,j,k;
		for (k=0;k<kk;k++) {
			frac[k] = 1./kk;
			for (i=0;i<mm;i++) {
				for (j=0;j<mm;j++) sig[k][i][j] = 0.;
				sig[k][i][i] = 1.0e-10;
			}
		}
		estep();
		mstep();
	}
	Doub estep() {
		Int k,m,n;
		Doub tmp,sum,max,oldloglike;
		VecDoub u(mm),v(mm);
		oldloglike = loglike;
		for (k=0;k<kk;k++) {
			Cholesky choltmp(sig[k]);
			lndets[k] = choltmp.logdet();
			for (n=0;n<nn;n++) {
				for (m=0;m<mm;m++) u[m] = data[n][m]-means[k][m];
				choltmp.elsolve(u,v);
				for (sum=0.,m=0; m<mm; m++) sum += SQR(v[m]);
				resp[n][k] = -0.5*(sum + lndets[k]) + log(frac[k]);
			}
		}
		loglike = 0;
		for (n=0;n<nn;n++) {
			max = -99.9e99;
			for (k=0;k<kk;k++) if (resp[n][k] > max) max = resp[n][k];
			for (sum=0.,k=0; k<kk; k++) sum += exp(resp[n][k]-max);
			tmp = max + log(sum);
			for (k=0;k<kk;k++) resp[n][k] = exp(resp[n][k] - tmp);
			loglike +=tmp;
		}
		return loglike - oldloglike;
	}
	void mstep() {
		Int j,n,k,m;
		Doub wgt,sum;
		for (k=0;k<kk;k++) {
			wgt=0.;
			for (n=0;n<nn;n++) wgt += resp[n][k];
			frac[k] = wgt/nn;
			for (m=0;m<mm;m++) {
				for (sum=0.,n=0; n<nn; n++) sum += resp[n][k]*data[n][m];
				means[k][m] = sum/wgt;
				for (j=0;j<mm;j++) {
					for (sum=0.,n=0; n<nn; n++) {
						sum += resp[n][k]*
							(data[n][m]-means[k][m])*(data[n][j]-means[k][j]);
					}
					sig[k][m][j] = sum/wgt;
				}
			}
		}
	}
};
