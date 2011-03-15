struct Voredge {
	Point<2> p[2];
	Int nearpt;
	Voredge() {}
	Voredge(Point<2> pa, Point<2> pb, Int np) : nearpt(np) {
		p[0] = pa; p[1] = pb;
	}
};
struct Voronoi : Delaunay {
	Int nseg;
	VecInt trindx;
	vector<Voredge> segs;
	Voronoi(vector< Point<2> > pvec);
};

Voronoi::Voronoi(vector< Point<2> > pvec) :
	Delaunay(pvec,1), nseg(0), trindx(npts), segs(6*npts+12) {
	Int i,j,k,p,jfirst;
	Ullong key;
	Triel tt;
	Point<2> cc, ccp;
	for (j=0; j<ntree; j++) {
		if (thelist[j].stat <= 0) continue;
		tt = thelist[j];
		for (k=0; k<3; k++) trindx[tt.p[k]] = j;
	}
	for (p=0; p<npts; p++) {
		tt = thelist[trindx[p]];
		if (tt.p[0] == p) {i = tt.p[1]; j = tt.p[2];}
		else if (tt.p[1] == p) {i = tt.p[2]; j = tt.p[0];}
		else if (tt.p[2] == p) {i = tt.p[0]; j = tt.p[1];}
		else throw("triangle should contain p");
		jfirst = j;
		ccp = circumcircle(pts[p],pts[i],pts[j]).center;
		while (1) {
			key = hashfn.int64(i) - hashfn.int64(p);
			if ( ! linehash->get(key,k) ) throw("Delaunay is incomplete");
			cc = circumcircle(pts[p],pts[k],pts[i]).center;
			segs[nseg++] = Voredge(ccp,cc,p);
			if (k == jfirst) break;
			ccp = cc;
			j=i;
			i=k;
		}
	}
}
