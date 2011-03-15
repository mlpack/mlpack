struct Bilin_interp {
	Int m,n;
	const MatDoub &y;
	Linear_interp x1terp, x2terp;

	Bilin_interp(VecDoub_I &x1v, VecDoub_I &x2v, MatDoub_I &ym)
		: m(x1v.size()), n(x2v.size()), y(ym),
		x1terp(x1v,x1v), x2terp(x2v,x2v) {}

	Doub interp(Doub x1p, Doub x2p) {
		Int i,j;
		Doub yy, t, u;
		i = x1terp.cor ? x1terp.hunt(x1p) : x1terp.locate(x1p);
		j = x2terp.cor ? x2terp.hunt(x2p) : x2terp.locate(x2p);
		t = (x1p-x1terp.xx[i])/(x1terp.xx[i+1]-x1terp.xx[i]);
		u = (x2p-x2terp.xx[j])/(x2terp.xx[j+1]-x2terp.xx[j]);
		yy = (1.-t)*(1.-u)*y[i][j] + t*(1.-u)*y[i+1][j]
			+ (1.-t)*u*y[i][j+1] + t*u*y[i+1][j+1];
		return yy;
	}
};
struct Poly2D_interp {
	Int m,n,mm,nn;
	const MatDoub &y;
	VecDoub yv;
	Poly_interp x1terp, x2terp;

	Poly2D_interp(VecDoub_I &x1v, VecDoub_I &x2v, MatDoub_I &ym,
		Int mp, Int np) : m(x1v.size()), n(x2v.size()),
		mm(mp), nn(np), y(ym), yv(m),
		x1terp(x1v,yv,mm), x2terp(x2v,x2v,nn) {}

	Doub interp(Doub x1p, Doub x2p) {
		Int i,j,k;
		i = x1terp.cor ? x1terp.hunt(x1p) : x1terp.locate(x1p);
		j = x2terp.cor ? x2terp.hunt(x2p) : x2terp.locate(x2p);
		for (k=i;k<i+mm;k++) {
			x2terp.yy = &y[k][0];
			yv[k] = x2terp.rawinterp(j,x2p);
		}
		return x1terp.rawinterp(i,x1p);
	}
};
struct Spline2D_interp {
	Int m,n;
	const MatDoub &y;
	const VecDoub &x1;
	VecDoub yv;
	NRvector<Spline_interp*> srp;

	Spline2D_interp(VecDoub_I &x1v, VecDoub_I &x2v, MatDoub_I &ym)
		: m(x1v.size()), n(x2v.size()), y(ym), yv(m), x1(x1v), srp(m) {
		for (Int i=0;i<m;i++) srp[i] = new Spline_interp(x2v,&y[i][0]);
	}

	~Spline2D_interp(){
		for (Int i=0;i<m;i++) delete srp[i];
	}

	Doub interp(Doub x1p, Doub x2p) {
		for (Int i=0;i<m;i++) yv[i] = (*srp[i]).interp(x2p);
		Spline_interp scol(x1,yv);
		return scol.interp(x1p);
	}
};
void bcucof(VecDoub_I &y, VecDoub_I &y1, VecDoub_I &y2, VecDoub_I &y12,
	const Doub d1, const Doub d2, MatDoub_O &c) {
	static Int wt_d[16*16]=
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		-3, 0, 0, 3, 0, 0, 0, 0,-2, 0, 0,-1, 0, 0, 0, 0,
		2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0,-3, 0, 0, 3, 0, 0, 0, 0,-2, 0, 0,-1,
		0, 0, 0, 0, 2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1,
		-3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0,
		9,-9, 9,-9, 6, 3,-3,-6, 6,-6,-3, 3, 4, 2, 1, 2,
		-6, 6,-6, 6,-4,-2, 2, 4,-3, 3, 3,-3,-2,-1,-1,-2,
		2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0,
		-6, 6,-6, 6,-3,-3, 3, 3,-4, 4, 2,-2,-2,-2,-1,-1,
		4,-4, 4,-4, 2, 2,-2,-2, 2,-2,-2, 2, 1, 1, 1, 1};
	Int l,k,j,i;
	Doub xx,d1d2=d1*d2;
	VecDoub cl(16),x(16);
	static MatInt wt(16,16,wt_d);
	for (i=0;i<4;i++) {
		x[i]=y[i];
		x[i+4]=y1[i]*d1;
		x[i+8]=y2[i]*d2;
		x[i+12]=y12[i]*d1d2;
	}
	for (i=0;i<16;i++) {
		xx=0.0;
		for (k=0;k<16;k++) xx += wt[i][k]*x[k];
		cl[i]=xx;
	}
	l=0;
	for (i=0;i<4;i++)
		for (j=0;j<4;j++) c[i][j]=cl[l++];
}
void bcuint(VecDoub_I &y, VecDoub_I &y1, VecDoub_I &y2, VecDoub_I &y12,
	const Doub x1l, const Doub x1u, const Doub x2l, const Doub x2u,
	const Doub x1, const Doub x2, Doub &ansy, Doub &ansy1, Doub &ansy2) {
	Int i;
	Doub t,u,d1=x1u-x1l,d2=x2u-x2l;
	MatDoub c(4,4);
	bcucof(y,y1,y2,y12,d1,d2,c);
	if (x1u == x1l || x2u == x2l)
		throw("Bad input in routine bcuint");
	t=(x1-x1l)/d1;
	u=(x2-x2l)/d2;
	ansy=ansy2=ansy1=0.0;
	for (i=3;i>=0;i--) {
		ansy=t*ansy+((c[i][3]*u+c[i][2])*u+c[i][1])*u+c[i][0];
		ansy2=t*ansy2+(3.0*c[i][3]*u+2.0*c[i][2])*u+c[i][1];
		ansy1=u*ansy1+(3.0*c[3][i]*t+2.0*c[2][i])*t+c[1][i];
	}
	ansy1 /= d1;
	ansy2 /= d2;
}
