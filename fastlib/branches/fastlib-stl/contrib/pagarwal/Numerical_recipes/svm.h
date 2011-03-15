struct Svmgenkernel {
	Int m, kcalls;
	MatDoub ker;
	VecDoub_I &y;
	MatDoub_I &data;
	Svmgenkernel(VecDoub_I &yy, MatDoub_I &ddata)
		: m(yy.size()),kcalls(0),ker(m,m),y(yy),data(ddata) {}
	virtual Doub kernel(const Doub *xi, const Doub *xj) = 0;
	inline Doub kernel(Int i, Doub *xj) {return kernel(&data[i][0],xj);}
	void fill() {
		Int i,j;		
		for (i=0;i<m;i++) for (j=0;j<=i;j++) {
			ker[i][j] = ker[j][i] = kernel(&data[i][0],&data[j][0]);
		}
	}
};
struct Svmlinkernel : Svmgenkernel {
	Int n;
	VecDoub mu;
	Svmlinkernel(MatDoub_I &ddata, VecDoub_I &yy)
		: Svmgenkernel(yy,ddata), n(data.ncols()), mu(n) {
		Int i,j;
		for (j=0;j<n;j++) mu[j] = 0.;
		for (i=0;i<m;i++) for (j=0;j<n;j++) mu[j] += data[i][j];
		for (j=0;j<n;j++) mu[j] /= m;
		fill();
	}
	Doub kernel(const Doub *xi, const Doub *xj) {
		Doub dott = 0.;
		for (Int k=0; k<n; k++) dott += (xi[k]-mu[k])*(xj[k]-mu[k]);
		return dott;
	}
};

struct Svmpolykernel : Svmgenkernel {
	Int n;
	Doub a, b, d;
	Svmpolykernel(MatDoub_I &ddata, VecDoub_I &yy, Doub aa, Doub bb, Doub dd)
		: Svmgenkernel(yy,ddata), n(data.ncols()), a(aa), b(bb), d(dd) {fill();}
	Doub kernel(const Doub *xi, const Doub *xj) {
		Doub dott = 0.;
		for (Int k=0; k<n; k++) dott += xi[k]*xj[k];
		return pow(a*dott+b,d);
	}
};

struct Svmgausskernel : Svmgenkernel {
	Int n;
	Doub sigma;
	Svmgausskernel(MatDoub_I &ddata, VecDoub_I &yy, Doub ssigma)
		: Svmgenkernel(yy,ddata), n(data.ncols()), sigma(ssigma) {fill();}
	Doub kernel(const Doub *xi, const Doub *xj) {
		Doub dott = 0.;
		for (Int k=0; k<n; k++) dott += SQR(xi[k]-xj[k]);
		return exp(-0.5*dott/(sigma*sigma));
	}
};
struct Svm {
	Svmgenkernel &gker;
	Int m, fnz, fub, niter;
	VecDoub alph, alphold;
	Ran ran;
	Bool alphinit;
	Doub dalph;
	Svm(Svmgenkernel &inker) : gker(inker), m(gker.y.size()),
		alph(m), alphold(m), ran(21), alphinit(false) {}
	Doub relax(Doub lambda, Doub om) {
		Int iter,j,jj,k,kk;
		Doub sum;
		VecDoub pinsum(m);
		if (alphinit == false) {
			for (j=0; j<m; j++) alph[j] = 0.;
			alphinit = true;
		}
		alphold = alph;
		Indexx x(alph);
		for (fnz=0; fnz<m; fnz++) if (alph[x.indx[fnz]] != 0.) break;	
		for (j=fnz; j<m-2; j++) {
			k = j + (ran.int32() % (m-j));
			SWAP(x.indx[j],x.indx[k]);
		}
		for (jj=0; jj<m; jj++) {
			j = x.indx[jj];
			sum = 0.;
			for (kk=fnz; kk<m; kk++) {
				k = x.indx[kk];
				sum += (gker.ker[j][k] + 1.)*gker.y[k]*alph[k];
			}
			alph[j] = alph[j] - (om/(gker.ker[j][j]+1.))*(gker.y[j]*sum-1.);
			alph[j] = MAX(0.,MIN(lambda,alph[j]));
			if (jj < fnz && alph[j]) SWAP(x.indx[--fnz],x.indx[jj]);
		}
		Indexx y(alph);
		for (fnz=0; fnz<m; fnz++) if (alph[y.indx[fnz]] != 0.) break;	
		for (fub=fnz; fub<m; fub++) if (alph[y.indx[fub]] == lambda) break;
		for (j=fnz; j<fub-2; j++) {
			k = j + (ran.int32() % (fub-j));
			SWAP(y.indx[j],y.indx[k]);
		}
		for (jj=fnz; jj<fub; jj++) {
			j = y.indx[jj];
			sum = 0.;
			for (kk=fub; kk<m; kk++) {
				k = y.indx[kk];
				sum += (gker.ker[j][k] + 1.)*gker.y[k]*alph[k];
			}
			pinsum[jj] = sum;
		}
		niter = MAX(Int(0.5*(m+1.0)*(m-fnz+1.0)/(SQR(fub-fnz+1.0))),1);
		for (iter=0; iter<niter; iter++) {
			for (jj=fnz; jj<fub; jj++) {
				j = y.indx[jj];
				sum = pinsum[jj];
				for (kk=fnz; kk<fub; kk++) {
					k = y.indx[kk];
					sum += (gker.ker[j][k] + 1.)*gker.y[k]*alph[k];
				}
				alph[j] = alph[j] - (om/(gker.ker[j][j]+1.))*(gker.y[j]*sum-1.);
				alph[j] = MAX(0.,MIN(lambda,alph[j]));
			}		
		}
		dalph = 0.;
		for (j=0;j<m;j++) dalph += SQR(alph[j]-alphold[j]);
		return sqrt(dalph);
	}
	Doub predict(Int k) {
		Doub sum = 0.;
		for (Int j=0; j<m; j++) sum += alph[j]*gker.y[j]*(gker.ker[j][k]+1.0);
		return sum;
	}
	Doub predict(Doub *x) {
		Doub sum = 0.;
		for (Int j=0; j<m; j++) sum += alph[j]*gker.y[j]*(gker.kernel(j,x)+1.0);
		return sum;
	}
};
