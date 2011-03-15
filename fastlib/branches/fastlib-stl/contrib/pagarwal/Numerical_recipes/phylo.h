struct Phylagglomnode {
	Int mo,ldau,rdau,nel;
	Doub modist,dep,seq;
};

struct Phylagglom{
	Int n, root, fsroot;
	Doub seqmax, depmax;
	vector<Phylagglomnode> t;
	virtual void premin(MatDoub &d, VecInt &nextp) = 0;
	virtual Doub dminfn(MatDoub &d, Int i, Int j) = 0;
	virtual Doub dbranchfn(MatDoub &d, Int i, Int j) = 0;
	virtual Doub dnewfn(MatDoub &d, Int k, Int i, Int j, Int ni, Int nj) = 0;
	virtual void drootbranchfn(MatDoub &d, Int i, Int j, Int ni, Int nj,
	Doub &bi, Doub &bj) = 0;
	Int comancestor(Int leafa, Int leafb);
	Phylagglom(const MatDoub &dist, Int fsr = -1)
		: n(dist.nrows()), fsroot(fsr), t(2*n-1) {}

	void makethetree(const MatDoub &dist) {
		Int i, j, k, imin, jmin, ncurr, node, ntask;
		Doub dd, dmin;
		MatDoub d(dist);
		VecInt tp(n), nextp(n), prevp(n), tasklist(2*n+1);
		VecDoub tmp(n);
		for (i=0;i<n;i++) {
			nextp[i] = i+1;
			prevp[i] = i-1;
			tp[i] = i;
			t[i].ldau = t[i].rdau = -1;
			t[i].nel = 1;
		}		
		prevp[0] = nextp[n-1] = -1;
		ncurr = n;
		for (node = n; node < 2*n-2; node++) {
			premin(d,nextp);
			dmin = 9.99e99;
			for (i=0; i>=0; i=nextp[i]) {
				if (tp[i] == fsroot) continue;
				for (j=nextp[i]; j>=0; j=nextp[j]) {
					if (tp[j] == fsroot) continue;
					if ((dd = dminfn(d,i,j)) < dmin) {
						dmin = dd;
						imin = i; jmin = j;
					}
				}
			}
			i = imin; j = jmin;
			t[tp[i]].mo = t[tp[j]].mo = node;
			t[tp[i]].modist = dbranchfn(d,i,j);
			t[tp[j]].modist = dbranchfn(d,j,i);
			t[node].ldau = tp[i];
			t[node].rdau = tp[j];
			t[node].nel = t[tp[i]].nel + t[tp[j]].nel;
			for (k=0; k>=0; k=nextp[k]) {
				tmp[k] = dnewfn(d,k,i,j,t[tp[i]].nel,t[tp[j]].nel);
			}
			for (k=0; k>=0; k=nextp[k]) d[i][k] = d[k][i] = tmp[k];
			tp[i] = node;
			if (prevp[j] >= 0) nextp[prevp[j]] = nextp[j];
			if (nextp[j] >= 0) prevp[nextp[j]] = prevp[j];
			ncurr--;
		}
		i = 0; j = nextp[0];
		root = node;
		t[tp[i]].mo = t[tp[j]].mo = t[root].mo = root;
		drootbranchfn(d,i,j,t[tp[i]].nel,t[tp[j]].nel,
			t[tp[i]].modist,t[tp[j]].modist);
		t[root].ldau = tp[i];
		t[root].rdau = tp[j];
		t[root].modist = t[root].dep = 0.;
		t[root].nel = t[tp[i]].nel + t[tp[j]].nel;
		ntask = 0;
		seqmax = depmax = 0.;
		tasklist[ntask++] = root;
		while (ntask > 0) {
			i = tasklist[--ntask];
			if (i >= 0) {
				t[i].dep = t[t[i].mo].dep + t[i].modist;
				if (t[i].dep > depmax) depmax = t[i].dep;
				if (t[i].ldau < 0) {
					t[i].seq = seqmax++;
				} else {
					tasklist[ntask++] = -i-1;
					tasklist[ntask++] = t[i].ldau;
					tasklist[ntask++] = t[i].rdau;
				}
			} else {
				i = -i-1;
				t[i].seq = 0.5*(t[t[i].ldau].seq + t[t[i].rdau].seq);
			}
		}
	}
};
struct Phylo_wpgma : Phylagglom {
	void premin(MatDoub &d, VecInt &nextp) {}
	Doub dminfn(MatDoub &d, Int i, Int j) {return d[i][j];}
	Doub dbranchfn(MatDoub &d, Int i, Int j) {return 0.5*d[i][j];}
	Doub dnewfn(MatDoub &d, Int k, Int i, Int j, Int ni, Int nj) {
		return 0.5*(d[i][k]+d[j][k]);}
	void drootbranchfn(MatDoub &d, Int i, Int j, Int ni, Int nj,
		Doub &bi, Doub &bj) {bi = bj = 0.5*d[i][j];}
	Phylo_wpgma(const MatDoub &dist) : Phylagglom(dist)
		{makethetree(dist);}
};
struct Phylo_upgma : Phylagglom {
	void premin(MatDoub &d, VecInt &nextp) {}
	Doub dminfn(MatDoub &d, Int i, Int j) {return d[i][j];}
	Doub dbranchfn(MatDoub &d, Int i, Int j) {return 0.5*d[i][j];}
	Doub dnewfn(MatDoub &d, Int k, Int i, Int j, Int ni, Int nj) {
		return (ni*d[i][k] + nj*d[j][k]) / (ni+nj);}
	void drootbranchfn(MatDoub &d, Int i, Int j, Int ni, Int nj,
		Doub &bi, Doub &bj) {bi = bj = 0.5*d[i][j];}
	Phylo_upgma(const MatDoub &dist) : Phylagglom(dist)
		{makethetree(dist);}
};
struct Phylo_slc : Phylagglom {
	void premin(MatDoub &d, VecInt &nextp) {}
	Doub dminfn(MatDoub &d, Int i, Int j) {return d[i][j];}
	Doub dbranchfn(MatDoub &d, Int i, Int j) {return 0.5*d[i][j];}
	Doub dnewfn(MatDoub &d, Int k, Int i, Int j, Int ni, Int nj) {
		return MIN(d[i][k],d[j][k]);}
	void drootbranchfn(MatDoub &d, Int i, Int j, Int ni, Int nj,
		Doub &bi, Doub &bj) {bi = bj = 0.5*d[i][j];}
	Phylo_slc(const MatDoub &dist) : Phylagglom(dist)
		{makethetree(dist);}
};

struct Phylo_clc : Phylagglom {
	void premin(MatDoub &d, VecInt &nextp) {}
	Doub dminfn(MatDoub &d, Int i, Int j) {return d[i][j];}
	Doub dbranchfn(MatDoub &d, Int i, Int j) {return 0.5*d[i][j];}
	Doub dnewfn(MatDoub &d, Int k, Int i, Int j, Int ni, Int nj) {
		return MAX(d[i][k],d[j][k]);}
	void drootbranchfn(MatDoub &d, Int i, Int j, Int ni, Int nj,
		Doub &bi, Doub &bj) {bi = bj = 0.5*d[i][j];}
	Phylo_clc(const MatDoub &dist) : Phylagglom(dist)
		{makethetree(dist);}
};
struct Phylo_nj : Phylagglom {
	VecDoub u;
	void premin(MatDoub &d, VecInt &nextp) {
		Int i,j,ncurr = 0;
		Doub sum;
		for (i=0; i>=0; i=nextp[i]) ncurr++;
		for (i=0; i>=0; i=nextp[i]) {
			sum = 0.;
			for (j=0; j>=0; j=nextp[j]) if (i != j) sum += d[i][j];
			u[i] = sum/(ncurr-2);
		}
	}
	Doub dminfn(MatDoub &d, Int i, Int j) {
		return d[i][j] - u[i] - u[j];
	}
	Doub dbranchfn(MatDoub &d, Int i, Int j) {
		return 0.5*(d[i][j]+u[i]-u[j]);
	}
	Doub dnewfn(MatDoub &d, Int k, Int i, Int j, Int ni, Int nj) {
		return 0.5*(d[i][k] + d[j][k] - d[i][j]);
	}
	void drootbranchfn(MatDoub &d, Int i, Int j, Int ni, Int nj,
	Doub &bi, Doub &bj) {
		bi = d[i][j]*(nj - 1 + 1.e-15)/(ni + nj -2 + 2.e-15);
		bj = d[i][j]*(ni - 1 + 1.e-15)/(ni + nj -2 + 2.e-15);
	}
	Phylo_nj(const MatDoub &dist, Int fsr = -1)
		: Phylagglom(dist,fsr), u(n) {makethetree(dist);}
};
Int Phylagglom::comancestor(Int leafa, Int leafb) {
	Int i, j;
	for (i = leafa; i != root; i = t[i].mo) {
		for (j = leafb; j != root; j = t[j].mo) if (i == j) break;
		if (i == j) break;
	}
	return i;
}
void newick(Phylagglom &p, MatChar str, char *filename) {
	FILE *OUT = fopen(filename,"wb");
	Int i, s, ntask = 0, n = p.n, root = p.root;
	VecInt tasklist(2*n+1);
	tasklist[ntask++] = (1 << 16) + root;
	while (ntask-- > 0) {
		s = tasklist[ntask] >> 16;
		i = tasklist[ntask] & 0xffff;
		if (s == 1 || s == 2) {
			tasklist[ntask++] = ((s+2) << 16) + p.t[i].mo;
			if (p.t[i].ldau >= 0) {
				fprintf(OUT,"(");
				tasklist[ntask++] = (2 << 16) + p.t[i].rdau;			
				tasklist[ntask++] = (1 << 16) + p.t[i].ldau;			
			}
			else fprintf(OUT,"%s:%f",&str[i][0],p.t[i].modist);	
		}
		else if (s == 3) {if (ntask > 0) fprintf(OUT,",\n");}
		else if (s == 4) {
			if (i == root) fprintf(OUT,");\n");
			else fprintf(OUT,"):%f",p.t[i].modist);
		}
	}
	fclose(OUT);
}
void phyl2ps(char *filename, Phylagglom &ph, MatChar str, Int extend,
	Doub xl, Doub xr, Doub yt, Doub yb) {
	Int i,j;
	Doub id,jd,xi,yi,xj,yj,seqmax,depmax;
	FILE *OUT = fopen(filename,"wb");
	fprintf(OUT,"%%!PS\n/Courier findfont 8 scalefont setfont\n");
	seqmax = ph.seqmax;
	depmax = ph.depmax;
	for (i=0; i<2*(ph.n)-1; i++) {
		j = ph.t[i].mo;
		id = ph.t[i].dep;
		jd = ph.t[j].dep;
		xi = xl + (xr-xl)*id/depmax;
		yi = yt - (yt-yb)*(ph.t[i].seq+0.5)/seqmax;
		xj = xl + (xr-xl)*jd/depmax;
		yj = yt - (yt-yb)*(ph.t[j].seq+0.5)/seqmax;
		fprintf(OUT,"%f %f moveto %f %f lineto %f %f lineto 0 setgray stroke\n",
			xj,yj,xj,yi,xi,yi);
		if (extend) {
			if (i < ph.n) {
				fprintf(OUT,"%f %f moveto %f %f lineto 0.7 setgray stroke\n",
					xi,yi,xr,yi);
				fprintf(OUT,"%f %f moveto (%s (%02d)) 0 setgray show\n",
					xr+3.,yi-2.,&str[i][0],i);
			}
		} else {
			if (i < ph.n) fprintf(OUT,"%f %f moveto (%s (%02d)) 0 setgray show\n",
				xi+3.,yi-2.,&str[i][0],i);
		}
	}
	fprintf(OUT,"showpage\n\004");
	fclose(OUT);
}
