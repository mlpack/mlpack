struct Fitmrq {
	static const Int NDONE=4, ITMAX=1000;
	Int ndat, ma, mfit;
	VecDoub_I &x,&y,&sig;
	Doub tol;
	void (*funcs)(const Doub, VecDoub_I &, Doub &, VecDoub_O &);
	VecBool ia;
	VecDoub a;
	MatDoub covar;
	MatDoub alpha;
	Doub chisq;

	Fitmrq(VecDoub_I &xx, VecDoub_I &yy, VecDoub_I &ssig, VecDoub_I &aa,
	void funks(const Doub, VecDoub_I &, Doub &, VecDoub_O &), const Doub
	TOL=1.e-3) : ndat(xx.size()), ma(aa.size()), x(xx), y(yy), sig(ssig),
	tol(TOL), funcs(funks), ia(ma), alpha(ma,ma), a(aa), covar(ma,ma) {
		for (Int i=0;i<ma;i++) ia[i] = true;
	}

	void hold(const Int i, const Doub val) {ia[i]=false; a[i]=val;}
	void free(const Int i) {ia[i]=true;}

	void fit() {
		Int j,k,l,iter,done=0;
		Doub alamda=.001,ochisq;
		VecDoub atry(ma),beta(ma),da(ma);
		mfit=0;
		for (j=0;j<ma;j++) if (ia[j]) mfit++;
		MatDoub oneda(mfit,1), temp(mfit,mfit);
		mrqcof(a,alpha,beta);
		for (j=0;j<ma;j++) atry[j]=a[j];
		ochisq=chisq;
		for (iter=0;iter<ITMAX;iter++) {
			if (done==NDONE) alamda=0.;
			for (j=0;j<mfit;j++) {
				for (k=0;k<mfit;k++) covar[j][k]=alpha[j][k];
				covar[j][j]=alpha[j][j]*(1.0+alamda);
				for (k=0;k<mfit;k++) temp[j][k]=covar[j][k];
				oneda[j][0]=beta[j];
			}
			gaussj(temp,oneda);
			for (j=0;j<mfit;j++) {
				for (k=0;k<mfit;k++) covar[j][k]=temp[j][k];
				da[j]=oneda[j][0];
			}
			if (done==NDONE) {
				covsrt(covar);
				covsrt(alpha);
				return;
			}
			for (j=0,l=0;l<ma;l++)
				if (ia[l]) atry[l]=a[l]+da[j++];
			mrqcof(atry,covar,da);
			if (abs(chisq-ochisq) < MAX(tol,tol*chisq)) done++;
			if (chisq < ochisq) {
				alamda *= 0.1;
				ochisq=chisq;
				for (j=0;j<mfit;j++) {
					for (k=0;k<mfit;k++) alpha[j][k]=covar[j][k];
						beta[j]=da[j];
				}
				for (l=0;l<ma;l++) a[l]=atry[l];
			} else {
				alamda *= 10.0;
				chisq=ochisq;
			}
		}
		throw("Fitmrq too many iterations");
	}


	void mrqcof(VecDoub_I &a, MatDoub_O &alpha, VecDoub_O &beta) {
		Int i,j,k,l,m;
		Doub ymod,wt,sig2i,dy;
		VecDoub dyda(ma);
		for (j=0;j<mfit;j++) {
			for (k=0;k<=j;k++) alpha[j][k]=0.0;
			beta[j]=0.;
		}
		chisq=0.;
		for (i=0;i<ndat;i++) {
			funcs(x[i],a,ymod,dyda);
			sig2i=1.0/(sig[i]*sig[i]);
			dy=y[i]-ymod;
			for (j=0,l=0;l<ma;l++) {
				if (ia[l]) {
					wt=dyda[l]*sig2i;
					for (k=0,m=0;m<l+1;m++)
						if (ia[m]) alpha[j][k++] += wt*dyda[m];
					beta[j++] += dy*wt;
				}
			}
			chisq += dy*dy*sig2i;
		}
		for (j=1;j<mfit;j++)
			for (k=0;k<j;k++) alpha[k][j]=alpha[j][k];
	}

	void covsrt(MatDoub_IO &covar) {
		Int i,j,k;
		for (i=mfit;i<ma;i++)
			for (j=0;j<i+1;j++) covar[i][j]=covar[j][i]=0.0;
		k=mfit-1;
		for (j=ma-1;j>=0;j--) {
			if (ia[j]) {
				for (i=0;i<ma;i++) SWAP(covar[i][k],covar[i][j]);
				for (i=0;i<ma;i++) SWAP(covar[k][i],covar[j][i]);
				k--;
			}
		}
	}

};
