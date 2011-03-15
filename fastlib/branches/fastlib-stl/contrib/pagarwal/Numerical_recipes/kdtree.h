template<Int DIM> struct Boxnode : Box<DIM> {
	Int mom, dau1, dau2, ptlo, pthi;
	Boxnode() {}
	Boxnode(Point<DIM> mylo, Point<DIM> myhi, Int mymom, Int myd1,
		Int myd2, Int myptlo, Int mypthi) :
	    Box<DIM>(mylo, myhi), mom(mymom), dau1(myd1), dau2(myd2),
		ptlo(myptlo), pthi(mypthi) {}
};
Int selecti(const Int k, Int *indx, Int n, Doub *arr)
{
	Int i,ia,ir,j,l,mid;
	Doub a;

	l=0;
	ir=n-1;
	for (;;) {
		if (ir <= l+1) {
			if (ir == l+1 && arr[indx[ir]] < arr[indx[l]])
				SWAP(indx[l],indx[ir]);
			return indx[k];
		} else {
			mid=(l+ir) >> 1;
			SWAP(indx[mid],indx[l+1]);
			if (arr[indx[l]] > arr[indx[ir]]) SWAP(indx[l],indx[ir]);
			if (arr[indx[l+1]] > arr[indx[ir]]) SWAP(indx[l+1],indx[ir]);
			if (arr[indx[l]] > arr[indx[l+1]]) SWAP(indx[l],indx[l+1]);
			i=l+1;
			j=ir;
			ia = indx[l+1];
			a=arr[ia];
			for (;;) {
				do i++; while (arr[indx[i]] < a);
				do j--; while (arr[indx[j]] > a);
				if (j < i) break;
				SWAP(indx[i],indx[j]);
			}
			indx[l+1]=indx[j];
			indx[j]=ia;
			if (j >= k) ir=j-1;
			if (j <= k) l=i;
		}
	}
}
template<Int DIM> struct KDtree {
	static const Doub BIG;
	Int nboxes, npts;
	vector< Point<DIM> > &ptss;
	Boxnode<DIM> *boxes;
	VecInt ptindx, rptindx;
	Doub *coords;
	KDtree(vector< Point<DIM> > &pts);
	~KDtree() {delete [] boxes;}
	Doub disti(Int jpt, Int kpt);
	Int locate(Point<DIM> pt);
	Int locate(Int jpt);
	Int nearest(Int jpt);
	Int nearest(Point<DIM> pt);
	void nnearest(Int jpt, Int *nn, Doub *dn, Int n);
	static void sift_down(Doub *heap, Int *ndx, Int nn);
	Int locatenear(Point<DIM> pt, Doub r, Int *list, Int nmax);
};

template<Int DIM> const Doub KDtree<DIM>::BIG(1.0e99);
template<Int DIM> KDtree<DIM>::KDtree(vector< Point<DIM> > &pts) :
ptss(pts), npts(pts.size()), ptindx(npts), rptindx(npts) {
	Int ntmp,m,k,kk,j,nowtask,jbox,np,tmom,tdim,ptlo,pthi;
	Int *hp;
	Doub *cp;
	Int taskmom[50], taskdim[50];
	for (k=0; k<npts; k++) ptindx[k] = k;
	m = 1;
	for (ntmp = npts; ntmp; ntmp >>= 1) {
		m <<= 1;
	}
	nboxes = 2*npts - (m >> 1);
	if (m < nboxes) nboxes = m;
	nboxes--;
	boxes = new Boxnode<DIM>[nboxes];
	coords = new Doub[DIM*npts];
	for (j=0, kk=0; j<DIM; j++, kk += npts) {
		for (k=0; k<npts; k++) coords[kk+k] = pts[k].x[j];
	}
	Point<DIM> lo(-BIG,-BIG,-BIG), hi(BIG,BIG,BIG);
	boxes[0] = Boxnode<DIM>(lo, hi, 0, 0, 0, 0, npts-1);
	jbox = 0;
	taskmom[1] = 0;
	taskdim[1] = 0;
	nowtask = 1;
	while (nowtask) {
		tmom = taskmom[nowtask];
		tdim = taskdim[nowtask--];
		ptlo = boxes[tmom].ptlo;
		pthi = boxes[tmom].pthi;
		hp = &ptindx[ptlo];
		cp = &coords[tdim*npts];
		np = pthi - ptlo + 1;
		kk = (np-1)/2;
		(void) selecti(kk,hp,np,cp);
		hi = boxes[tmom].hi;
		lo = boxes[tmom].lo;
		hi.x[tdim] = lo.x[tdim] = coords[tdim*npts + hp[kk]];
		boxes[++jbox] = Boxnode<DIM>(boxes[tmom].lo,hi,tmom,0,0,ptlo,ptlo+kk);
		boxes[++jbox] = Boxnode<DIM>(lo,boxes[tmom].hi,tmom,0,0,ptlo+kk+1,pthi);
		boxes[tmom].dau1 = jbox-1;
		boxes[tmom].dau2 = jbox;
		if (kk > 1) {
			taskmom[++nowtask] = jbox-1;
			taskdim[nowtask] = (tdim+1) % DIM;
		}
		if (np - kk > 3) {
			taskmom[++nowtask] = jbox;
			taskdim[nowtask] = (tdim+1) % DIM;
		}
	}
	for (j=0; j<npts; j++) rptindx[ptindx[j]] = j;
	delete [] coords;
}
template<Int DIM> Doub KDtree<DIM>::disti(Int jpt, Int kpt) {
	if (jpt == kpt) return BIG;
	else return dist(ptss[jpt], ptss[kpt]);
}
template<Int DIM> Int KDtree<DIM>::locate(Point<DIM> pt) {
	Int nb,d1,jdim;
	nb = jdim = 0;
	while (boxes[nb].dau1) {
		d1 = boxes[nb].dau1;
		if (pt.x[jdim] <= boxes[d1].hi.x[jdim]) nb=d1;
		else nb=boxes[nb].dau2;
		jdim = ++jdim % DIM;
	}
	return nb;
}
template<Int DIM> Int KDtree<DIM>::locate(Int jpt) {
	Int nb,d1,jh;
	jh = rptindx[jpt];
	nb = 0;
	while (boxes[nb].dau1) {
		d1 = boxes[nb].dau1;
		if (jh <= boxes[d1].pthi) nb=d1;
		else nb = boxes[nb].dau2;
	}
	return nb;
}
template<Int DIM> Int KDtree<DIM>::nearest(Point<DIM> pt) {
	Int i,k,nrst,ntask;
	Int task[50];
	Doub dnrst = BIG, d;
	k = locate(pt);
	for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
		d = dist(ptss[ptindx[i]],pt);
		if (d < dnrst) {
			nrst = ptindx[i];
			dnrst = d;
		}
	}
	task[1] = 0;
	ntask = 1;
	while (ntask) {
		k = task[ntask--];
		if (dist(boxes[k],pt) < dnrst) {
			if (boxes[k].dau1) {
				task[++ntask] = boxes[k].dau1;
				task[++ntask] = boxes[k].dau2;
			} else {
				for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
					d = dist(ptss[ptindx[i]],pt);
					if (d < dnrst) {
						nrst = ptindx[i];
						dnrst = d;
					}
				}
			}
		}
	}
	return nrst;
}
template<Int DIM> void KDtree<DIM>::nnearest(Int jpt, Int *nn, Doub *dn, Int n)
{
	Int i,k,ntask,kp;
	Int task[50];
	Doub d;
	if (n > npts-1) throw("too many neighbors requested");
	for (i=0; i<n; i++) dn[i] = BIG;
	kp = boxes[locate(jpt)].mom;
	while (boxes[kp].pthi - boxes[kp].ptlo < n) kp = boxes[kp].mom;
	for (i=boxes[kp].ptlo; i<=boxes[kp].pthi; i++) {
		if (jpt == ptindx[i]) continue;
		d = disti(ptindx[i],jpt);
		if (d < dn[0]) {
			dn[0] = d;
			nn[0] = ptindx[i];
			if (n>1) sift_down(dn,nn,n);
		}
	}
	task[1] = 0;
	ntask = 1;
	while (ntask) {
		k = task[ntask--];
		if (k == kp) continue;
		if (dist(boxes[k],ptss[jpt]) < dn[0]) {
			if (boxes[k].dau1) {
				task[++ntask] = boxes[k].dau1;
				task[++ntask] = boxes[k].dau2;
			} else {
				for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
					d = disti(ptindx[i],jpt);
					if (d < dn[0]) {
						dn[0] = d;
						nn[0] = ptindx[i];
						if (n>1) sift_down(dn,nn,n);
					}
				}
			}
		}
	}
	return;
}
template<Int DIM> void KDtree<DIM>::sift_down(Doub *heap, Int *ndx, Int nn) {
	Int n = nn - 1;
	Int j,jold,ia;
	Doub a;
	a = heap[0];
	ia = ndx[0];
	jold = 0;
	j = 1;
	while (j <= n) {
		if (j < n && heap[j] < heap[j+1]) j++;
		if (a >= heap[j]) break;
		heap[jold] = heap[j];
		ndx[jold] = ndx[j];
		jold = j;
		j = 2*j + 1;
	}
	heap[jold] = a;
	ndx[jold] = ia;
}
template<Int DIM>
Int KDtree<DIM>::locatenear(Point<DIM> pt, Doub r, Int *list, Int nmax) {
	Int k,i,nb,nbold,nret,ntask,jdim,d1,d2;
	Int task[50];
	nb = jdim = nret = 0;
	if (r < 0.0) throw("radius must be nonnegative");
	while (boxes[nb].dau1) {
		nbold = nb;
		d1 = boxes[nb].dau1;
		d2 = boxes[nb].dau2;
		if (pt.x[jdim] + r <= boxes[d1].hi.x[jdim]) nb = d1;
		else if (pt.x[jdim] - r >= boxes[d2].lo.x[jdim]) nb = d2;
		jdim = ++jdim % DIM;
		if (nb == nbold) break;
	}
	task[1] = nb;
	ntask = 1;
	while (ntask) {
		k = task[ntask--];
		if (dist(boxes[k],pt) > r) continue;
		if (boxes[k].dau1) {
			task[++ntask] = boxes[k].dau1;
			task[++ntask] = boxes[k].dau2;
		} else {
			for (i=boxes[k].ptlo; i<=boxes[k].pthi; i++) {
				if (dist(ptss[ptindx[i]],pt) <= r && nret < nmax)
					list[nret++] = ptindx[i];
				if (nret == nmax) return nmax;
			}
		}
	}
	return nret;
}
