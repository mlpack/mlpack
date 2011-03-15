struct Mglin {
	Int n,ng;
	MatDoub *uj,*uj1;
	NRvector<NRmatrix<Doub> *> rho;

	Mglin(MatDoub_IO &u, const Int ncycle) : n(u.nrows()), ng(0)
	{
		Int nn=n;
		while (nn >>= 1) ng++;
		if ((n-1) != (1 << ng))
			throw("n-1 must be a power of 2 in mglin.");
		nn=n;
		Int ngrid=ng-1;
		rho.resize(ng);
		rho[ngrid] = new MatDoub(nn,nn);
		*rho[ngrid]=u;
		while (nn > 3) {
			nn=nn/2+1;
			rho[--ngrid]=new MatDoub(nn,nn);
			rstrct(*rho[ngrid],*rho[ngrid+1]);
		}
		nn=3;
		uj=new MatDoub(nn,nn);
		slvsml(*uj,*rho[0]);
		for (Int j=1;j<ng;j++) {
			nn=2*nn-1;
			uj1=uj;
			uj=new MatDoub(nn,nn);
			interp(*uj,*uj1);
			delete uj1;
			for (Int jcycle=0;jcycle<ncycle;jcycle++)
				mg(j,*uj,*rho[j]);
		}
		u = *uj;
	}

	~Mglin()
	{
		if (uj != NULL) delete uj;
		for (Int j=0;j<ng;j++)
			if (rho[j] != NULL) delete rho[j];
	}

	void interp(MatDoub_O &uf, MatDoub_I &uc)
	{
		Int nf=uf.nrows();
		Int nc=nf/2+1;
		for (Int jc=0;jc<nc;jc++)
			for (Int ic=0;ic<nc;ic++) uf[2*ic][2*jc]=uc[ic][jc];
		for (Int jf=0;jf<nf;jf+=2)
			for (Int iif=1;iif<nf-1;iif+=2)
				uf[iif][jf]=0.5*(uf[iif+1][jf]+uf[iif-1][jf]);
		for (Int jf=1;jf<nf-1;jf+=2)
			for (Int iif=0;iif<nf;iif++)
				uf[iif][jf]=0.5*(uf[iif][jf+1]+uf[iif][jf-1]);
	}

	void addint(MatDoub_O &uf, MatDoub_I &uc, MatDoub_O &res)
	{
		Int nf=uf.nrows();
		interp(res,uc);
		for (Int j=0;j<nf;j++)
			for (Int i=0;i<nf;i++)
				uf[i][j] += res[i][j];
	}

	void slvsml(MatDoub_O &u, MatDoub_I &rhs)
	{
		Doub h=0.5;
		for (Int i=0;i<3;i++)
			for (Int j=0;j<3;j++)
				u[i][j]=0.0;
		u[1][1] = -h*h*rhs[1][1]/4.0;
	}

	void relax(MatDoub_IO &u, MatDoub_I &rhs)
	{
		Int n=u.nrows();
		Doub h=1.0/(n-1);
		Doub h2=h*h;
		for (Int ipass=0,jsw=1;ipass<2;ipass++,jsw=3-jsw) {
			for (Int j=1,isw=jsw;j<n-1;j++,isw=3-isw)
				for (Int i=isw;i<n-1;i+=2)
					u[i][j]=0.25*(u[i+1][j]+u[i-1][j]+u[i][j+1]
						+u[i][j-1]-h2*rhs[i][j]);
		}
	}

	void resid(MatDoub_O &res, MatDoub_I &u, MatDoub_I &rhs)
	{
		Int n=u.nrows();
		Doub h=1.0/(n-1);
		Doub h2i=1.0/(h*h);
		for (Int j=1;j<n-1;j++)
			for (Int i=1;i<n-1;i++)
				res[i][j] = -h2i*(u[i+1][j]+u[i-1][j]+u[i][j+1]
					+u[i][j-1]-4.0*u[i][j])+rhs[i][j];
		for (Int i=0;i<n;i++)
			res[i][0]=res[i][n-1]=res[0][i]=res[n-1][i]=0.0;
	}

	void rstrct(MatDoub_O &uc, MatDoub_I &uf)
	{
		Int nc=uc.nrows();
		Int ncc=2*nc-2;
		for (Int jf=2,jc=1;jc<nc-1;jc++,jf+=2) {
			for (Int iif=2,ic=1;ic<nc-1;ic++,iif+=2) {
				uc[ic][jc]=0.5*uf[iif][jf]+0.125*(uf[iif+1][jf]+uf[iif-1][jf]
					+uf[iif][jf+1]+uf[iif][jf-1]);
			}
		}
		for (Int jc=0,ic=0;ic<nc;ic++,jc+=2) {
			uc[ic][0]=uf[jc][0];
			uc[ic][nc-1]=uf[jc][ncc];
		}
		for (Int jc=0,ic=0;ic<nc;ic++,jc+=2) {
			uc[0][ic]=uf[0][jc];
			uc[nc-1][ic]=uf[ncc][jc];
		}
	}

	void mg(Int j, MatDoub_IO &u, MatDoub_I &rhs)
	{
		const Int NPRE=1,NPOST=1;
		Int nf=u.nrows();
		Int nc=(nf+1)/2;
		if (j == 0)
			slvsml(u,rhs);
		else {
			MatDoub res(nc,nc),v(nc,nc,0.0),temp(nf,nf);
			for (Int jpre=0;jpre<NPRE;jpre++)
				relax(u,rhs);
			resid(temp,u,rhs);
			rstrct(res,temp);
			mg(j-1,v,res);
			addint(u,v,temp);
			for (Int jpost=0;jpost<NPOST;jpost++)
				relax(u,rhs);
		}
	}
};
