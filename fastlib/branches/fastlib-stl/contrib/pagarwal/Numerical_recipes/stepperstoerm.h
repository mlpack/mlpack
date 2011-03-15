template <class D>
struct StepperStoerm : StepperBS<D> {
	using StepperBS<D>::x; using StepperBS<D>::xold; using StepperBS<D>::y;
	using StepperBS<D>::dydx; using StepperBS<D>::dense; using StepperBS<D>::n;
	using StepperBS<D>::KMAXX; using StepperBS<D>::IMAXX; using StepperBS<D>::nseq;
	using StepperBS<D>::cost; using StepperBS<D>::mu; using StepperBS<D>::errfac;
	using StepperBS<D>::ysave; using StepperBS<D>::fsave;
	using StepperBS<D>::dens; using StepperBS<D>::neqn;
	MatDoub ysavep;
	StepperStoerm(VecDoub_IO &yy, VecDoub_IO &dydxx, Doub &xx,
		const Doub atol, const Doub rtol, bool dens);
	void dy(VecDoub_I &y, const Doub htot, const Int k, VecDoub_O &yend,
		Int &ipt,D &derivs);
	void prepare_dense(const Doub h,VecDoub_I &dydxnew, VecDoub_I &ysav,
		VecDoub_I &scale, const Int k, Doub &error);
	Doub dense_out(const Int i,const Doub x,const Doub h);
	void dense_interp(const Int n, VecDoub_IO &y, const Int imit);
};
template <class D>
StepperStoerm<D>::StepperStoerm(VecDoub_IO &yy, VecDoub_IO &dydxx, Doub &xx,
	const Doub atoll,const Doub rtoll, bool dens)
	: StepperBS<D>(yy,dydxx,xx,atoll,rtoll,dens),ysavep(IMAXX,n/2) {
	neqn=n/2;
	cost[0]=nseq[0]/2+1;
	for (Int k=0;k<KMAXX;k++)
		cost[k+1]=cost[k]+nseq[k+1]/2;
	for (Int i=0; i<2*IMAXX+1; i++) {
		Int ip7=i+7;
		Doub fac=1.5/ip7;
		errfac[i]=fac*fac*fac;
		Doub e = 0.5*sqrt(Doub(i+1)/ip7);
		for (Int j=0; j<=i; j++) {
			errfac[i] *= e/(j+1);
		}
	}
}
template <class D>
void StepperStoerm<D>::prepare_dense(const Doub h,VecDoub_I &dydxnew,
	VecDoub_I &ysav,VecDoub_I &scale,const Int k, Doub &error) {
	Doub h2=h*h;
	mu=MAX(1,2*k-3);
	for (Int i=0; i<neqn; i++) {
		dens[i]=ysav[i];
		dens[neqn+i]=h*ysav[neqn+i];
		dens[2*neqn+i]=h2*dydx[i];
		dens[3*neqn+i]=y[i];
		dens[4*neqn+i]=h*y[neqn+i];
		dens[5*neqn+i]=h2*dydxnew[i];
	}
	for (Int j=1; j<=k; j++) {
		Doub dblenj=nseq[j];
		for (Int l=j; l>=1; l--) {
			Doub factor=SQR(dblenj/nseq[l-1])-1.0;
			for (Int i=0; i<neqn; i++) {
				ysave[l-1][i]=ysave[l][i]+(ysave[l][i]-ysave[l-1][i])/factor;
				ysavep[l-1][i]=ysavep[l][i]+(ysavep[l][i]-ysavep[l-1][i])/factor;
			}
		}
	}
	for (Int i=0; i<neqn; i++) {
		dens[6*neqn+i]=ysave[0][i];
		dens[7*neqn+i]=h*ysavep[0][i];
	}
	for (Int kmi=2; kmi<=mu; kmi++) {
		Int kbeg=(kmi-2)/2;
		if (kmi == 2*kbeg+2) {
			if (kmi == 2) {
				for (Int i=0; i<neqn; i++)
					ysave[0][i]=0.5*(dydxnew[i]+fsave[0][i]);
				kbeg=1;
			}
			for (Int kk=kbeg; kk<=k; kk++) {
				Doub facnj=0.5*pow(nseq[kk]/2.0,kmi-2);
				Int ipt=kk*kk+kk+kmi/2-2;
				for (Int i=0; i<neqn; i++)
					ysave[kk][i]=(fsave[ipt][i]+fsave[ipt+1][i])*facnj;
			}
		} else {
			for (Int kk=kbeg; kk<=k; kk++) {
				Doub facnj=pow(nseq[kk]/2.0,kmi-2);
				Int ipt=kk*kk+kk+kbeg;
				for (Int i=0; i<neqn; i++)
					ysave[kk][i]=fsave[ipt][i]*facnj;
			}
		}
		for (Int j=kbeg+1; j<=k; j++) {
			Doub dblenj=nseq[j];
			for (Int l=j; l>=kbeg+1; l--) {
				Doub factor=SQR(dblenj/nseq[l-1])-1.0;
				for (Int i=0; i<neqn; i++)
					ysave[l-1][i]=ysave[l][i]+
						(ysave[l][i]-ysave[l-1][i])/factor;
			}
		}
		for (Int i=0; i<neqn; i++)
			dens[(kmi+6)*neqn+i]=ysave[kbeg][i]*h2;
		if (kmi == mu) continue;
		for (Int kk=(kmi-1)/2; kk<=k; kk++) {
			Int lbeg=kk*kk+kmi-2;
			Int lend=SQR(kk+1)-1;
			if (kmi == 2) lbeg++;
			for (Int l=lend; l>=lbeg; l--)
				for (Int i=0; i<neqn; i++)
					fsave[l][i]=fsave[l][i]-fsave[l-1][i];
			if (kmi == 2) {
				Int l=lbeg-1;
				for (Int i=0; i<neqn; i++)
					fsave[l][i]=fsave[l][i]-dydx[i];
			}
		}
	}
	dense_interp(neqn,dens,mu);
	error=0.0;
	if (mu >= 1) {
		for (Int i=0; i<neqn; i++)
			error += SQR(dens[(mu+6)*neqn+i]/scale[i]);
		error=sqrt(error/neqn)*errfac[mu-1];
	}
}
template <class D>
Doub StepperStoerm<D>::dense_out(const Int i,const Doub x,const Doub h) {
	Doub theta=(x-xold)/h;
	Doub theta1=1.0-theta;
	Int neqn=n/2;
	if (i>=neqn) throw("no dense output for y' in StepperStoerm");
	Doub yinterp=dens[i]+theta*(dens[neqn+i]+theta1*(dens[2*neqn+i]+
		theta*(dens[3*neqn+i]+theta1*(dens[4*neqn+i]*theta+
		dens[5*neqn+i]*theta1))));
	if (mu<0)
		return yinterp;
	Doub theta05=theta-0.5;
	Doub t4=theta*theta1;
	Doub c=dens[neqn*(mu+6)+i];
	for (Int j=mu;j>0; j--)
		c=dens[neqn*(j+5)+i]+c*theta05/j;
	yinterp += t4*t4*t4*c;
	return yinterp;
}
template <class D>
void StepperStoerm<D>::dense_interp(const Int n, VecDoub_IO &y, const Int imit) {
	Doub y0,y1,yp0,yp1,ypp0,ypp1,ydiff,ah,bh,ch,dh,eh,fh,gh,abh,gfh,gmf,
		ph0,ph1,ph2,ph3,ph4,ph5,fc1,fc2,fc3;
	VecDoub a(41);
	for (Int i=0; i<n; i++) {
		y0=y[i];
		y1=y[3*n+i];
		yp0=y[n+i];
		yp1=y[4*n+i];
		ypp0=y[2*n+i]/2.0;
		ypp1=y[5*n+i]/2.0;
		ydiff=y1-y0;
		ah=ydiff-yp0;
		bh=yp1-ydiff;
		ch=ah-ypp0;
		dh=bh-ah;
		eh=ypp1-bh;
		fh=dh-ch;
		gh=eh-dh;
		y[n+i]=ydiff;
		y[2*n+i]=-ah;
		y[3*n+i]=-dh;
		y[4*n+i]=gh;
		y[5*n+i]=fh;
		if (imit < 0) continue;
		abh=ah+bh;
		gfh=gh+fh;
		gmf=gh-fh;
		ph0=0.5*(y0+y1+0.25*(-abh+0.25*gfh));
		ph1=ydiff+0.25*(ah-bh+0.25*gmf);
		ph2=abh-0.5*gfh;
		ph3=6.0*(bh-ah)-3.0*gmf;
		ph4=12.0*gfh;
		ph5=120.0*gmf;
		if (imit >= 1) {
			a[1]=64.0*(y[7*n+i]-ph1);
			if (imit >= 3) {
				a[3]=64.0*(y[9*n+i]-ph3+a[1]*9.0/8.0);
				if (imit >= 5) {
					a[5]=64.0*(y[11*n+i]-ph5+a[3]*15.0/4.0-a[1]*90.0);
					for (Int im=7; im <=imit; im+=2) {
						fc1=im*(im-1)*3.0/16.0;
						fc2=fc1*(im-2)*(im-3)*4.0;
						fc3=im*(im-1)*(im-2)*(im-3)*(im-4)*(im-5);
						a[im]=64.0*(y[(im+6)*n+i]+fc1*a[im-2]-fc2*a[im-4]+fc3*a[im-6]);
					}
				}
			}
		}
		a[0]=64.0*(y[6*n+i]-ph0);
		if (imit >= 2) {
			a[2]=64.0*(y[n*8+i]-ph2+a[0]*3.0/8.0);
			if (imit >= 4) {
				a[4]=64.0*(y[n*10+i]-ph4+a[2]*9.0/4.0-a[0]*18.0);
				for (Int im=6; im<=imit; im+=2) {
					fc1=im*(im-1)*3.0/16.0;
					fc2=fc1*(im-2)*(im-3)*4.0;
					fc3=im*(im-1)*(im-2)*(im-3)*(im-4)*(im-5);
					a[im]=64.0*(y[n*(im+6)+i]+a[im-2]*fc1-a[im-4]*fc2+a[im-6]*fc3);
				}
			}
		}
		for (Int im=0; im<=imit; im++)
			y[n*(im+6)+i]=a[im];
	}
}
template <class D>
void StepperStoerm<D>::dy(VecDoub_I &y, const Doub htot, const Int k,
	VecDoub_O &yend, Int &ipt, D &derivs) {
	VecDoub ytemp(n);
	Int nstep=nseq[k];
	Doub h=htot/nstep;
	Doub h2=2.0*h;
	for (Int i=0;i<neqn;i++) {
		ytemp[i]=y[i];
		Int ni=neqn+i;
		ytemp[ni]=y[ni]+h*dydx[i];
	}
	Doub xnew=x;
	Int nstp2=nstep/2;
	for (Int nn=1;nn<=nstp2;nn++) {
		if (dense && nn == (nstp2+1)/2) {
			for (Int i=0;i<neqn;i++) {
				ysavep[k][i]=ytemp[neqn+i];
				ysave[k][i]=ytemp[i]+h*ytemp[neqn+i];
			}
		}
		for (Int i=0;i<neqn;i++)
			ytemp[i] += h2*ytemp[neqn+i];
		xnew += h2;
		derivs(xnew,ytemp,yend);
		if (dense && abs(nn-(nstp2+1)/2) < k+1) {
			ipt++;
			for (Int i=0;i<neqn;i++)
				fsave[ipt][i]=yend[i];
		}
		if (nn != nstp2) {
			for (Int i=0;i<neqn;i++)
				ytemp[neqn+i] += h2*yend[i];
		}
	}
	for (Int i=0;i<neqn;i++) {
		Int ni=neqn+i;
		yend[ni]=ytemp[ni]+h*yend[i];
		yend[i]=ytemp[i];
	}
}
