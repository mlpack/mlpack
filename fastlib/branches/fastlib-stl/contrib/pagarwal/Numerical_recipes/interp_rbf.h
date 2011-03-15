struct RBF_fn {
	virtual Doub rbf(Doub r) = 0;
};

struct RBF_interp {
	Int dim, n;
	const MatDoub &pts;
	const VecDoub &vals;
	VecDoub w;
	RBF_fn &fn;
	Bool norm;

	RBF_interp(MatDoub_I &ptss, VecDoub_I &valss, RBF_fn &func, Bool nrbf=false)
	: dim(ptss.ncols()), n(ptss.nrows()) , pts(ptss), vals(valss),
	w(n), fn(func), norm(nrbf) {
		Int i,j;
		Doub sum;
		MatDoub rbf(n,n);
		VecDoub rhs(n);
		for (i=0;i<n;i++) {
			sum = 0.;
			for (j=0;j<n;j++) {
				sum += (rbf[i][j] = fn.rbf(rad(&pts[i][0],&pts[j][0])));
			}
			if (norm) rhs[i] = sum*vals[i];
			else rhs[i] = vals[i];
		}
		LUdcmp lu(rbf);
		lu.solve(rhs,w);
	}

	Doub interp(VecDoub_I &pt) {
		Doub fval, sum=0., sumw=0.;
		if (pt.size() != dim) throw("RBF_interp bad pt size");
		for (Int i=0;i<n;i++) {
			fval = fn.rbf(rad(&pt[0],&pts[i][0]));
			sumw += w[i]*fval;
			sum += fval;
		}
		return norm ? sumw/sum : sumw;
	}

	Doub rad(const Doub *p1, const Doub *p2) {
		Doub sum = 0.;
		for (Int i=0;i<dim;i++) sum += SQR(p1[i]-p2[i]);
		return sqrt(sum);
	}
};
struct RBF_multiquadric : RBF_fn {
	Doub r02;
	RBF_multiquadric(Doub scale=1.) : r02(SQR(scale)) {}
	Doub rbf(Doub r) { return sqrt(SQR(r)+r02); }
};

struct RBF_thinplate : RBF_fn {
	Doub r0;
	RBF_thinplate(Doub scale=1.) : r0(scale) {}
	Doub rbf(Doub r) { return r <= 0. ? 0. : SQR(r)*log(r/r0); }
};

struct RBF_gauss : RBF_fn {
	Doub r0;
	RBF_gauss(Doub scale=1.) : r0(scale) {}
	Doub rbf(Doub r) { return exp(-0.5*SQR(r/r0)); }
};

struct RBF_inversemultiquadric : RBF_fn {
	Doub r02;
	RBF_inversemultiquadric(Doub scale=1.) : r02(SQR(scale)) {}
	Doub rbf(Doub r) { return 1./sqrt(SQR(r)+r02); }
};
struct Shep_interp {
	Int dim, n;
	const MatDoub &pts;
	const VecDoub &vals;
	Doub pneg;

	Shep_interp(MatDoub_I &ptss, VecDoub_I &valss, Doub p=2.)
	: dim(ptss.ncols()), n(ptss.nrows()) , pts(ptss),
	vals(valss), pneg(-p) {}

	Doub interp(VecDoub_I &pt) {
		Doub r, w, sum=0., sumw=0.;
		if (pt.size() != dim) throw("RBF_interp bad pt size");
		for (Int i=0;i<n;i++) {
			if ((r=rad(&pt[0],&pts[i][0])) == 0.) return vals[i];
			sum += (w = pow(r,pneg));
			sumw += w*vals[i];
		}
		return sumw/sum;
	}

	Doub rad(const Doub *p1, const Doub *p2) {
		Doub sum = 0.;
		for (Int i=0;i<dim;i++) sum += SQR(p1[i]-p2[i]);
		return sqrt(sum);
	}
};
