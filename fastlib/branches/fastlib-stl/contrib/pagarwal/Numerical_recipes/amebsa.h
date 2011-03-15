template <class T>
struct Amebsa {
	T &funk;
	const Doub ftol;
	Ranq1 ran;
	Doub yb;
	Int ndim;
	VecDoub pb;
	Int mpts;
	VecDoub y;
	MatDoub p;
	Doub tt;
	Amebsa(VecDoub_I &point, const Doub del, T &funkk, const Doub ftoll) :
		funk(funkk), ftol(ftoll), ran(1234),
		yb(numeric_limits<Doub>::max()), ndim(point.size()), pb(ndim),
		mpts(ndim+1), y(mpts), p(mpts,ndim)
	{
		for (Int i=0;i<mpts;i++) {
			for (Int j=0;j<ndim;j++)
				p[i][j]=point[j];
			if (i != 0) p[i][i-1] += del;
		}
		inity();
	}
	Amebsa(VecDoub_I &point, VecDoub_I &dels, T &funkk, const Doub ftoll) :
		funk(funkk), ftol(ftoll), ran(1234),
		yb(numeric_limits<Doub>::max()), ndim(point.size()), pb(ndim),
		mpts(ndim+1), y(mpts), p(mpts,ndim)
	{
		for (Int i=0;i<mpts;i++) {
			for (Int j=0;j<ndim;j++)
				p[i][j]=point[j];
			if (i != 0) p[i][i-1] += dels[i-1];
		}
		inity();
	}
	Amebsa(MatDoub_I &pp, T &funkk, const Doub ftoll) : funk(funkk),
		ftol(ftoll), ran(1234), yb(numeric_limits<Doub>::max()),
		ndim(pp.ncols()), pb(ndim), mpts(pp.nrows()), y(mpts), p(pp)
	{ inity(); }

	void inity() {
		VecDoub x(ndim);
		for (Int i=0;i<mpts;i++) {
			for (Int j=0;j<ndim;j++)
				x[j]=p[i][j];
			y[i]=funk(x);
		}
	}
	Bool anneal(Int &iter, const Doub temperature)
	{
		VecDoub psum(ndim);
		tt = -temperature;
		get_psum(p,psum);
		for (;;) {
			Int ilo=0;
			Int ihi=1;
			Doub ylo=y[0]+tt*log(ran.doub());
			Doub ynhi=ylo;
			Doub yhi=y[1]+tt*log(ran.doub());
			if (ylo > yhi) {
				ihi=0;
				ilo=1;
				ynhi=yhi;
				yhi=ylo;
				ylo=ynhi;
			}
			for (Int i=3;i<=mpts;i++) {
				Doub yt=y[i-1]+tt*log(ran.doub());
				if (yt <= ylo) {
					ilo=i-1;
					ylo=yt;
				}
				if (yt > yhi) {
					ynhi=yhi;
					ihi=i-1;
					yhi=yt;
				} else if (yt > ynhi) {
					ynhi=yt;
				}
			}
			Doub rtol=2.0*abs(yhi-ylo)/(abs(yhi)+abs(ylo));
			if (rtol < ftol || iter < 0) {
				SWAP(y[0],y[ilo]);
				for (Int n=0;n<ndim;n++)
					SWAP(p[0][n],p[ilo][n]);
				if (rtol < ftol)
					return true;
				else
					return false;
			}
			iter -= 2;
			Doub ytry=amotsa(p,y,psum,ihi,yhi,-1.0);
			if (ytry <= ylo) {
				ytry=amotsa(p,y,psum,ihi,yhi,2.0);
			} else if (ytry >= ynhi) {
				Doub ysave=yhi;
				ytry=amotsa(p,y,psum,ihi,yhi,0.5);
				if (ytry >= ysave) {
					for (Int i=0;i<mpts;i++) {
						if (i != ilo) {
							for (Int j=0;j<ndim;j++) {
								psum[j]=0.5*(p[i][j]+p[ilo][j]);
								p[i][j]=psum[j];
							}
							y[i]=funk(psum);
						}
					}
					iter -= ndim;
					get_psum(p,psum);
				}
			} else ++iter;
		}
	}
	inline void get_psum(MatDoub_I &p, VecDoub_O &psum)
	{
		for (Int n=0;n<ndim;n++) {
			Doub sum=0.0;
			for (Int m=0;m<mpts;m++) sum += p[m][n];
			psum[n]=sum;
		}
	}
	Doub amotsa(MatDoub_IO &p, VecDoub_O &y, VecDoub_IO &psum,
		const Int ihi, Doub &yhi, const Doub fac)
	{
		VecDoub ptry(ndim);
		Doub fac1=(1.0-fac)/ndim;
		Doub fac2=fac1-fac;
		for (Int j=0;j<ndim;j++)
			ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
		Doub ytry=funk(ptry);
		if (ytry <= yb) {
			for (Int j=0;j<ndim;j++) pb[j]=ptry[j];
			yb=ytry;
		}
		Doub yflu=ytry-tt*log(ran.doub());
		if (yflu < yhi) {
			y[ihi]=ytry;
			yhi=yflu;
			for (Int j=0;j<ndim;j++) {
				psum[j] += ptry[j]-p[ihi][j];
				p[ihi][j]=ptry[j];
			}
		}
		return yflu;
	}
};
