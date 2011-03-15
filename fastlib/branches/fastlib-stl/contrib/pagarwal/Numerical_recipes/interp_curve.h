struct Curve_interp {
	Int dim, n, in;
	Bool cls;
	MatDoub pts;
	VecDoub s;
	VecDoub ans;
	NRvector<Spline_interp*> srp;

	Curve_interp(MatDoub &ptsin, Bool close=0)
	: n(ptsin.nrows()), dim(ptsin.ncols()), in(close ? 2*n : n),
	cls(close), pts(dim,in), s(in), ans(dim), srp(dim) {
		Int i,ii,im,j,ofs;
		Doub ss,soff,db,de;
		ofs = close ? n/2 : 0;
		s[0] = 0.;
		for (i=0;i<in;i++) {
			ii = (i-ofs+n) % n;
			im = (ii-1+n) % n;
			for (j=0;j<dim;j++) pts[j][i] = ptsin[ii][j];
			if (i>0) {
				s[i] = s[i-1] + rad(&ptsin[ii][0],&ptsin[im][0]);
				if (s[i] == s[i-1]) throw("error in Curve_interp");
			}
		}
		ss = close ? s[ofs+n]-s[ofs] : s[n-1]-s[0];
		soff = s[ofs];
		for (i=0;i<in;i++) s[i] = (s[i]-soff)/ss;
		for (j=0;j<dim;j++) {
			db = in < 4 ? 1.e99 : fprime(&s[0],&pts[j][0],1);
			de = in < 4 ? 1.e99 : fprime(&s[in-1],&pts[j][in-1],-1);
			srp[j] = new Spline_interp(s,&pts[j][0],db,de);
		}
	}
	~Curve_interp() {for (Int j=0;j<dim;j++) delete srp[j];}
	
	VecDoub &interp(Doub t) {
		if (cls) t = t - floor(t);
		for (Int j=0;j<dim;j++) ans[j] = (*srp[j]).interp(t);
		return ans;
	}

	Doub fprime(Doub *x, Doub *y, Int pm) {
		Doub s1 = x[0]-x[pm*1], s2 = x[0]-x[pm*2], s3 = x[0]-x[pm*3],
			s12 = s1-s2, s13 = s1-s3, s23 = s2-s3;
		return -(s1*s2/(s13*s23*s3))*y[pm*3]+(s1*s3/(s12*s2*s23))*y[pm*2]
			-(s2*s3/(s1*s12*s13))*y[pm*1]+(1./s1+1./s2+1./s3)*y[0];
	}

	Doub rad(const Doub *p1, const Doub *p2) {
		Doub sum = 0.;
		for (Int i=0;i<dim;i++) sum += SQR(p1[i]-p2[i]);
		return sqrt(sum);
	}

};
