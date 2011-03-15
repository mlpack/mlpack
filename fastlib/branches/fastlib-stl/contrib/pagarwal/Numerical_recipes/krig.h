template<class T>
struct Krig {
	const MatDoub &x;
	const T &vgram;
	Int ndim, npt;
	Doub lastval, lasterr;
	VecDoub y,dstar,vstar,yvi;
	MatDoub v;
	LUdcmp *vi;
	
	Krig(MatDoub_I &xx, VecDoub_I &yy, T &vargram, const Doub *err=NULL)
	: x(xx),vgram(vargram),npt(xx.nrows()),ndim(xx.ncols()),dstar(npt+1),
	vstar(npt+1),v(npt+1,npt+1),y(npt+1),yvi(npt+1) {
		Int i,j;
		for (i=0;i<npt;i++) {
			y[i] = yy[i];
			for (j=i;j<npt;j++) {
				v[i][j] = v[j][i] = vgram(rdist(&x[i][0],&x[j][0]));
			}
			v[i][npt] = v[npt][i] = 1.;
		}
		v[npt][npt] = y[npt] = 0.;
		if (err) for (i=0;i<npt;i++) v[i][i] -= SQR(err[i]);
		vi = new LUdcmp(v);
		vi->solve(y,yvi);
	}
	~Krig() { delete vi; }

	Doub interp(VecDoub_I &xstar) {
		Int i;
		for (i=0;i<npt;i++) vstar[i] = vgram(rdist(&xstar[0],&x[i][0]));
		vstar[npt] = 1.;
		lastval = 0.;
		for (i=0;i<=npt;i++) lastval += yvi[i]*vstar[i];
		return lastval;
	}

	Doub interp(VecDoub_I &xstar, Doub &esterr) {
		lastval = interp(xstar);
		vi->solve(vstar,dstar);
		lasterr = 0;
		for (Int i=0;i<=npt;i++) lasterr += dstar[i]*vstar[i];
		esterr = lasterr = sqrt(MAX(0.,lasterr));
		return lastval;
	}

	Doub rdist(const Doub *x1, const Doub *x2) {
		Doub d=0.;
		for (Int i=0;i<ndim;i++) d += SQR(x1[i]-x2[i]);
		return sqrt(d);
	}
};
struct Powvargram {
	Doub alph, bet, nugsq;

	Powvargram(MatDoub_I &x, VecDoub_I &y, const Doub beta=1.5, const Doub nug=0.)
	: bet(beta), nugsq(nug*nug) {
		Int i,j,k,npt=x.nrows(),ndim=x.ncols();
		Doub rb,num=0.,denom=0.;
		for (i=0;i<npt;i++) for (j=i+1;j<npt;j++) {
			rb = 0.;
			for (k=0;k<ndim;k++) rb += SQR(x[i][k]-x[j][k]);
			rb = pow(rb,0.5*beta);
			num += rb*(0.5*SQR(y[i]-y[j]) - nugsq);
			denom += SQR(rb);
		}
		alph = num/denom;
	}

	Doub operator() (const Doub r) const {return nugsq+alph*pow(r,bet);}
};
