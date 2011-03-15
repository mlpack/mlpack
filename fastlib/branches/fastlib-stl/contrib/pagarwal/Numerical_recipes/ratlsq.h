Ratfn ratlsq(Doub fn(const Doub), const Doub a, const Doub b, const Int mm,
	const Int kk, Doub &dev)
{
	const Int NPFAC=8,MAXIT=5;
	const Doub BIG=1.0e99,PIO2=1.570796326794896619;
	Int i,it,j,ncof=mm+kk+1,npt=NPFAC*ncof;
	Doub devmax,e,hth,power,sum;
	VecDoub bb(npt),coff(ncof),ee(npt),fs(npt),wt(npt),xs(npt);
	MatDoub u(npt,ncof);
	Ratfn ratbest(coff,mm+1,kk+1);
	dev=BIG;
	for (i=0;i<npt;i++) {
		if (i < (npt/2)-1) {
			hth=PIO2*i/(npt-1.0);
			xs[i]=a+(b-a)*SQR(sin(hth));
		} else {
			hth=PIO2*(npt-i)/(npt-1.0);
			xs[i]=b-(b-a)*SQR(sin(hth));
		}
		fs[i]=fn(xs[i]);
		wt[i]=1.0;
		ee[i]=1.0;
	}
	e=0.0;
	for (it=0;it<MAXIT;it++) {
		for (i=0;i<npt;i++) {
			power=wt[i];
			bb[i]=power*(fs[i]+SIGN(e,ee[i]));
			for (j=0;j<mm+1;j++) {
				u[i][j]=power;
				power *= xs[i];
			}
			power = -bb[i];
			for (j=mm+1;j<ncof;j++) {
				power *= xs[i];
				u[i][j]=power;
			}
		}
		SVD svd(u);
		svd.solve(bb,coff);
		devmax=sum=0.0;
		Ratfn rat(coff,mm+1,kk+1);
		for (j=0;j<npt;j++) {
			ee[j]=rat(xs[j])-fs[j];
			wt[j]=abs(ee[j]);
			sum += wt[j];
			if (wt[j] > devmax) devmax=wt[j];
		}
		e=sum/npt;
		if (devmax <= dev) {
			ratbest = rat;
			dev=devmax;
		}
		cout << " ratlsq iteration= " << it;
		cout << "  max error= " << setw(10) << devmax << endl;
	}
	return ratbest;
}
