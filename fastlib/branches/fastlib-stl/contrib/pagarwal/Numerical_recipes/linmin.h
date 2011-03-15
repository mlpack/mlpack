template <class T>
struct Linmin {
	VecDoub &p;
	VecDoub &xi;
	Doub &fret;
	F1dim<T> f1dim;
	Mnbrak mnbrak;
	Brent brent;
	Linmin(VecDoub_IO &pp, VecDoub_IO &xii, Doub &frett, T &func) :
		p(pp), xi(xii), fret(frett), f1dim(p,xi,func) {}
	void min()
	{
		const Doub TOL=1.0e-8;
		Doub xx,xmin,fx,fb,fa,bx,ax;
		Int n=p.size();
		ax=0.0;
		xx=1.0;
		mnbrak.bracket(ax,xx,bx,fa,fx,fb,f1dim);
		fret=brent.min(ax,xx,bx,f1dim,TOL,xmin);
		for (Int j=0;j<n;j++) {
			xi[j] *= xmin;
			p[j] += xi[j];
		}
	}
};
