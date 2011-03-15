struct Triel {
	Point<2> *pts;
	Int p[3];
	Int d[3];
	Int stat;
	void setme(Int a, Int b, Int c, Point<2> *ptss) {
		pts = ptss;
		p[0] = a; p[1] = b; p[2] = c;
		d[0] = d[1] = d[2] = -1;
		stat = 1;
	}
	Int contains(Point<2> point) {
		Doub d;
		Int i,j,ztest = 0;
		for (i=0; i<3; i++) {
			j = (i+1) %3;
			d = (pts[p[j]].x[0]-pts[p[i]].x[0])*(point.x[1]-pts[p[i]].x[1]) -
				(pts[p[j]].x[1]-pts[p[i]].x[1])*(point.x[0]-pts[p[i]].x[0]);
			if (d < 0.0) return -1;
			if (d == 0.0) ztest = 1;
		}
		return (ztest? 0 : 1);
	}
};
Doub incircle(Point<2> d, Point<2> a, Point<2> b, Point<2> c) {
	Circle cc = circumcircle(a,b,c);
	Doub radd = SQR(d.x[0]-cc.center.x[0]) + SQR(d.x[1]-cc.center.x[1]);
	return (SQR(cc.radius) - radd);
}

struct Nullhash {
	Nullhash(Int nn) {}
	inline Ullong fn(const void *key) const { return *((Ullong *)key); }
};
struct Delaunay {
	Int npts,ntri,ntree,ntreemax,opt;
	Doub delx,dely;
	vector< Point<2> > pts;
	vector<Triel> thelist;
	Hash<Ullong,Int,Nullhash> *linehash;
	Hash<Ullong,Int,Nullhash> *trihash;
	Int *perm;
	Delaunay(vector<Point<2> > &pvec, Int options = 0);
	Ranhash hashfn;
	Doub interpolate(const Point<2> &p, const vector<Doub> &fnvals,
		Doub defaultval=0.0);
	void insertapoint(Int r);
	Int whichcontainspt(const Point<2> &p, Int strict = 0);
	Int storetriangle(Int a, Int b, Int c);
	void erasetriangle(Int a, Int b, Int c, Int d0, Int d1, Int d2);
	static Uint jran;
	static const Doub fuzz, bigscale;
};
const Doub Delaunay::fuzz  = 1.0e-6;
const Doub Delaunay::bigscale = 1000.0;
Uint Delaunay::jran = 14921620;
Delaunay::Delaunay(vector< Point<2> > &pvec, Int options) :
	npts(pvec.size()), ntri(0), ntree(0), ntreemax(10*npts+1000),
	opt(options), pts(npts+3), thelist(ntreemax) {
	Int j;
	Doub xl,xh,yl,yh;
	linehash = new Hash<Ullong,Int,Nullhash>(6*npts+12,6*npts+12);
	trihash = new Hash<Ullong,Int,Nullhash>(2*npts+6,2*npts+6);
	perm = new Int[npts];
	xl = xh = pvec[0].x[0];
	yl = yh = pvec[0].x[1];
	for (j=0; j<npts; j++) {
		pts[j] = pvec[j];
		perm[j] = j;
		if (pvec[j].x[0] < xl) xl = pvec[j].x[0];
		if (pvec[j].x[0] > xh) xh = pvec[j].x[0];
		if (pvec[j].x[1] < yl) yl = pvec[j].x[1];
		if (pvec[j].x[1] > yh) yh = pvec[j].x[1];
	}
	delx = xh - xl;
	dely = yh - yl;
	pts[npts] = Point<2>(0.5*(xl + xh), yh + bigscale*dely);
	pts[npts+1] = Point<2>(xl - 0.5*bigscale*delx,yl - 0.5*bigscale*dely);
	pts[npts+2] = Point<2>(xh + 0.5*bigscale*delx,yl - 0.5*bigscale*dely);
	storetriangle(npts,npts+1,npts+2);
	for (j=npts; j>0; j--) SWAP(perm[j-1],perm[hashfn.int64(jran++) % j]);
	for (j=0; j<npts; j++) insertapoint(perm[j]);
	for (j=0; j<ntree; j++) {
	  if (thelist[j].stat > 0) {
			if (thelist[j].p[0] >= npts || thelist[j].p[1] >= npts ||
			thelist[j].p[2] >= npts) {
				thelist[j].stat = -1;
				ntri--;
			}
		}
	}
	if (!(opt & 1)) {
		delete [] perm;
		delete trihash;
		delete linehash;
	}
}
void Delaunay::insertapoint(Int r) {
	Int i,j,k,l,s,tno,ntask,d0,d1,d2;
	Ullong key;
	Int tasks[50], taski[50], taskj[50];
	for (j=0; j<3; j++) {
		tno = whichcontainspt(pts[r],1);
		if (tno >= 0) break;
		pts[r].x[0] += fuzz * delx * (hashfn.doub(jran++)-0.5);
		pts[r].x[1] += fuzz * dely * (hashfn.doub(jran++)-0.5);
	}
	if (j == 3) throw("points degenerate even after fuzzing");
	ntask = 0;
	i = thelist[tno].p[0]; j = thelist[tno].p[1]; k = thelist[tno].p[2];
	if (opt & 2 && i < npts && j < npts && k < npts) return;
	d0 =storetriangle(r,i,j);
	tasks[++ntask] = r; taski[ntask] = i; taskj[ntask] = j;
	d1 = storetriangle(r,j,k);
	tasks[++ntask] = r; taski[ntask] = j; taskj[ntask] = k;
	d2 = storetriangle(r,k,i);
	tasks[++ntask] = r; taski[ntask] = k; taskj[ntask] = i;
	erasetriangle(i,j,k,d0,d1,d2);
	while (ntask) {
		s=tasks[ntask]; i=taski[ntask]; j=taskj[ntask--];
		key = hashfn.int64(j) - hashfn.int64(i);
		if ( ! linehash->get(key,l) ) continue;
		if (incircle(pts[l],pts[j],pts[s],pts[i]) > 0.0){
			d0 = storetriangle(s,l,j);
			d1 = storetriangle(s,i,l);
			erasetriangle(s,i,j,d0,d1,-1);
			erasetriangle(l,j,i,d0,d1,-1);
			key = hashfn.int64(i)-hashfn.int64(j);
			linehash->erase(key);
			key = 0 - key;
			linehash->erase(key);
			tasks[++ntask] = s; taski[ntask] = l; taskj[ntask] = j;
			tasks[++ntask] = s; taski[ntask] = i; taskj[ntask] = l;
		}
	}
}
Int Delaunay::whichcontainspt(const Point<2> &p, Int strict) {
	Int i,j,k=0;
	while (thelist[k].stat <= 0) {
		for (i=0; i<3; i++) {
			if ((j = thelist[k].d[i]) < 0) continue;
			if (strict) {
				if (thelist[j].contains(p) > 0) break;
			} else {
				if (thelist[j].contains(p) >= 0) break;
			}
		}
		if (i == 3) return -1;
		k = j;
	}
	return k;
}

void Delaunay::erasetriangle(Int a, Int b, Int c, Int d0, Int d1, Int d2) {
	Ullong key;
	Int j;
	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	if (trihash->get(key,j) == 0) throw("nonexistent triangle");
	trihash->erase(key);
	thelist[j].d[0] = d0; thelist[j].d[1] = d1; thelist[j].d[2] = d2;
	thelist[j].stat = 0;
	ntri--;
}

Int Delaunay::storetriangle(Int a, Int b, Int c) {
	Ullong key;
	thelist[ntree].setme(a,b,c,&pts[0]);
	key = hashfn.int64(a) ^ hashfn.int64(b) ^ hashfn.int64(c);
	trihash->set(key,ntree);
	key = hashfn.int64(b)-hashfn.int64(c);
	linehash->set(key,a);
	key = hashfn.int64(c)-hashfn.int64(a);
	linehash->set(key,b);
	key = hashfn.int64(a)-hashfn.int64(b);
	linehash->set(key,c);
	if (++ntree == ntreemax) throw("thelist is sized too small");
	ntri++;
	return (ntree-1);
}
Doub Delaunay::interpolate(const Point<2> &p,
const vector<Doub> &fnvals, Doub defaultval) {
	Int n,i,j,k;
	Doub wgts[3];
	Int ipts[3];
	Doub sum, ans = 0.0;
	n = whichcontainspt(p);
	if (n < 0) return defaultval;
 	for (i=0; i<3; i++) ipts[i] = thelist[n].p[i];
	for (i=0,j=1,k=2; i<3; i++,j++,k++) {
		if (j == 3) j=0;
		if (k == 3) k=0;
		wgts[k]=(pts[ipts[j]].x[0]-pts[ipts[i]].x[0])*(p.x[1]-pts[ipts[i]].x[1])
			- (pts[ipts[j]].x[1]-pts[ipts[i]].x[1])*(p.x[0]-pts[ipts[i]].x[0]);
	}
	sum = wgts[0] + wgts[1] + wgts[2];
	if (sum == 0) throw("degenerate triangle");
	for (i=0; i<3; i++) ans += wgts[i]*fnvals[ipts[i]]/sum;
	return ans;
}
struct Convexhull : Delaunay {
	Int nhull;
	Int *hullpts;
	Convexhull(vector< Point<2> > pvec);
};

Convexhull::Convexhull(vector< Point<2> > pvec) : Delaunay(pvec,2), nhull(0) {
	Int i,j,k,pstart;
	vector<Int> nextpt(npts);
	for (j=0; j<ntree; j++) {
		if (thelist[j].stat != -1) continue;
		for (i=0,k=1; i<3; i++,k++) {
			if (k == 3) k=0;
			if (thelist[j].p[i] < npts && thelist[j].p[k] < npts) break;
		}
		if (i==3) continue;
		++nhull;
		nextpt[(pstart = thelist[j].p[k])] = thelist[j].p[i];
	}
	if (nhull == 0) throw("no hull segments found");
	hullpts = new Int[nhull];
	j=0;
	i = hullpts[j++] = pstart;
	while ((i=nextpt[i]) != pstart) hullpts[j++] = i;
}
struct Minspantree : Delaunay {
	Int nspan;
	VecInt minsega, minsegb;
	Minspantree(vector< Point<2> > pvec);
};

Minspantree::Minspantree(vector< Point<2> > pvec) :
	Delaunay(pvec,0), nspan(npts-1), minsega(nspan), minsegb(nspan) {
	Int i,j,k,jj,kk,m,tmp,nline,n = 0;
	Triel tt;
	nline = ntri + npts -1;
	VecInt sega(nline);
	VecInt segb(nline);
	VecDoub segd(nline);
	VecInt mo(npts);
	for (j=0; j<ntree; j++) {
		if (thelist[j].stat == 0) continue;
		tt = thelist[j];
		for (i=0,k=1; i<3; i++,k++) {
			if (k==3) k=0;
			if (tt.p[i] > tt.p[k]) continue;
			if (tt.p[i] >= npts || tt.p[k] >= npts) continue;
			sega[n] = tt.p[i];
			segb[n] = tt.p[k];
			segd[n] = dist(pts[sega[n]],pts[segb[n]]);
			n++;
		}
	}
	Indexx idx(segd);
	for (j=0; j<npts; j++) mo[j] = j;
	n = -1;
	for (i=0; i<nspan; i++) {
		 for (;;) {
			jj = j = idx.el(sega,++n);
			kk = k = idx.el(segb,n);	
			while (mo[jj] != jj) jj = mo[jj];
			while (mo[kk] != kk) kk = mo[kk];
			if (jj != kk) {
				minsega[i] = j;
				minsegb[i] = k;
				m = mo[jj] = kk;
				jj = j;
				while (mo[jj] != m) {
					tmp = mo[jj];
					mo[jj] = m;
					jj = tmp;
				}
				kk = k;
				while (mo[kk] != m) {
					tmp = mo[kk];
					mo[kk] = m;
					kk = tmp;
				}
				break;
			}
		}
	}
}
