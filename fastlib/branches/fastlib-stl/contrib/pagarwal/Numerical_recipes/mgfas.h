struct Mgfas {
	Int n,ng;
	MatDoub *uj,*uj1;
	NRvector<NRmatrix<Doub> *> rho;

	Mgfas(MatDoub_IO &u, const Int maxcyc) : n(u.nrows()), ng(0)
	{
		Int nn=n;
		while (nn >>= 1) ng++;
		if ((n-1) != (1 << ng))
			throw("n-1 must be a power of 2 in mgfas.");
		nn=n;
		Int ngrid=ng-1;
		rho.resize(ng);
		rho[ngrid]=new MatDoub(nn,nn);
		*rho[ngrid]=u;
		while (nn > 3) {
			nn=nn/2+1;
			rho[--ngrid]=new MatDoub(nn,nn);
			rstrct(*rho[ngrid],*rho[ngrid+1]);
		}
		nn=3;
		uj=new MatDoub(nn,nn);
		slvsm2(*uj,*rho[0]);
		for (Int j=1;j<ng;j++) {
			nn=2*nn-1;
			uj1=uj;
			uj=new MatDoub(nn,nn);
			MatDoub temp(nn,nn);
			interp(*uj,*uj1);
			delete uj1;
			for (Int jcycle=0;jcycle<maxcyc;jcycle++) {
				Doub trerr=1.0;
				mg(j,*uj,temp,rho,trerr);
				lop(temp,*uj);
				matsub(temp,*rho[j],temp);
				Doub res=anorm2(temp);
				if (res < trerr) break;
			}
		}
		u = *uj;
	}
	
	~Mgfas()
	{
		if (uj != NULL) delete uj;
		for (Int j=0;j<ng;j++)
			if (rho[j] != NULL) delete rho[j];
	}
	
	void matadd(MatDoub_I &a, MatDoub_I &b, MatDoub_O &c)
	{
		Int n=a.nrows();
		for (Int j=0;j<n;j++)
			for (Int i=0;i<n;i++)
				c[i][j]=a[i][j]+b[i][j];
	}
	
	void matsub(MatDoub_I &a, MatDoub_I &b, MatDoub_O &c)
	{
		Int n=a.nrows();
		for (Int j=0;j<n;j++)
			for (Int i=0;i<n;i++)
				c[i][j]=a[i][j]-b[i][j];
	}
	
	void slvsm2(MatDoub_O &u, MatDoub_I &rhs)
	{
		Doub h=0.5;
		for (Int i=0;i<3;i++)
			for (Int j=0;j<3;j++)
				u[i][j]=0.0;
		Doub fact=2.0/(h*h);
		Doub disc=sqrt(fact*fact+rhs[1][1]);
		u[1][1]= -rhs[1][1]/(fact+disc);
	}
	
	void relax2(MatDoub_IO &u, MatDoub_I &rhs)
	{
		Int n=u.nrows();
		Int jsw=1;
		Doub h=1.0/(n-1);
		Doub h2i=1.0/(h*h);
		Doub foh2 = -4.0*h2i;
		for (Int ipass=0;ipass<2;ipass++,jsw=3-jsw) {
			Int isw=jsw;
			for (Int j=1;j<n-1;j++,isw=3-isw) {
				for (Int i=isw;i<n-1;i+=2) {
					Doub res=h2i*(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]-
						4.0*u[i][j])+u[i][j]*u[i][j]-rhs[i][j];
					u[i][j] -= res/(foh2+2.0*u[i][j]);
				}
			}
		}
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
	
	void lop(MatDoub_O &out, MatDoub_I &u)
	{
		Int n=u.nrows();
		Doub h=1.0/(n-1);
		Doub h2i=1.0/(h*h);
		for (Int j=1;j<n-1;j++)
			for (Int i=1;i<n-1;i++)
				out[i][j]=h2i*(u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]-
					4.0*u[i][j])+u[i][j]*u[i][j];
		for (Int i=0;i<n;i++)
			out[i][0]=out[i][n-1]=out[0][i]=out[n-1][i]=0.0;
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
	
	Doub anorm2(MatDoub_I &a)
	{
		Doub sum=0.0;
		Int n=a.nrows();
		for (Int j=0;j<n;j++)
			for (Int i=0;i<n;i++)
				sum += a[i][j]*a[i][j];
		return sqrt(sum)/n;
	}
	
	void mg(const Int j, MatDoub_IO &u, MatDoub_I &rhs,
		NRvector<NRmatrix<Doub> *> &rho, Doub &trerr)
	{
		const Int NPRE=1,NPOST=1;
		const Doub ALPHA=0.33;
		Doub dum=-1.0;
		Int nf=u.nrows();
		Int nc=(nf+1)/2;
		MatDoub temp(nf,nf);
		if (j == 0) {
			matadd(rhs,*rho[j],temp);
			slvsm2(u,temp);
		} else {
			MatDoub v(nc,nc),ut(nc,nc),tau(nc,nc),tempc(nc,nc);
			for (Int jpre=0;jpre<NPRE;jpre++) {
				if (trerr < 0.0) {
					matadd(rhs,*rho[j],temp);
					relax2(u,temp);
				}
				else
					relax2(u,*rho[j]);
			}
			rstrct(ut,u);
			v=ut;
			lop(tau,ut);
			lop(temp,u);
			if (trerr < 0.0)
				matsub(temp,rhs,temp);
			rstrct(tempc,temp);
			matsub(tau,tempc,tau);
			if (trerr > 0.0)
				trerr=ALPHA*anorm2(tau);
			mg(j-1,v,tau,rho,dum);
			matsub(v,ut,tempc);
			interp(temp,tempc);
			matadd(u,temp,u);
			for (Int jpost=0;jpost<NPOST;jpost++) {
				if (trerr < 0.0) {
					matadd(rhs,*rho[j],temp);
					relax2(u,temp);
				}
				else
					relax2(u,*rho[j]);
			}
		}
	}
};
