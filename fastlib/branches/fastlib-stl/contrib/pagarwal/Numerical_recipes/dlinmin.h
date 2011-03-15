template <class T>
struct Dlinmin {
	VecDoub &p;
	VecDoub &xi;
	Doub &fret;
	Df1dim<T> df1dim;
	Mnbrak mnbrak;
	Dbrent dbrent;
	Dlinmin(VecDoub_IO &pp, VecDoub_IO &xii, Doub &frett, T &funcd) :
		p(pp), xi(xii), fret(frett), df1dim(p,xi,funcd) {}
	void min()
	{
		const Doub TOL=1.0e-8;
		Doub xx,xmin,fx,fb,fa,bx,ax;
		Int n=p.size();
		ax=0.0;
		xx=1.0;
		mnbrak.bracket(ax,xx,bx,fa,fx,fb,df1dim);
		fret=dbrent.min(ax,xx,bx,df1dim,TOL,xmin);
		for (Int j=0;j<n;j++) {
			xi[j] *= xmin;
			p[j] += xi[j];
		}
	}
};
