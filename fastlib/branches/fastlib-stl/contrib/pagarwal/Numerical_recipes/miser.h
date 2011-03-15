void miser(Doub func(VecDoub_I &), VecDoub_I &regn, const Int npts,
	const Doub dith, Doub &ave, Doub &var) {
	const Int MNPT=15, MNBS=60;
	const Doub PFAC=0.1, TINY=1.0e-30, BIG=1.0e30;
	static Int iran=0;
	Int j,jb,n,ndim,npre,nptl,nptr;
	Doub avel,varl,fracl,fval,rgl,rgm,rgr,s,sigl,siglb,sigr,sigrb;
	Doub sum,sumb,summ,summ2;

	ndim=regn.size()/2;
	VecDoub pt(ndim);
	if (npts < MNBS) {
		summ=summ2=0.0;
		for (n=0;n<npts;n++) {
			ranpt(pt,regn);
			fval=func(pt);
			summ += fval;
			summ2 += fval * fval;
		}
		ave=summ/npts;
		var=MAX(TINY,(summ2-summ*summ/npts)/(npts*npts));
	} else {
		VecDoub rmid(ndim);
		npre=MAX(Int(npts*PFAC),Int(MNPT));
		VecDoub fmaxl(ndim),fmaxr(ndim),fminl(ndim),fminr(ndim);
		for (j=0;j<ndim;j++) {
			iran=(iran*2661+36979) % 175000;
			s=SIGN(dith,Doub(iran-87500));
			rmid[j]=(0.5+s)*regn[j]+(0.5-s)*regn[ndim+j];
			fminl[j]=fminr[j]=BIG;
			fmaxl[j]=fmaxr[j]=(-BIG);
		}
		for (n=0;n<npre;n++) {
			ranpt(pt,regn);
			fval=func(pt);
			for (j=0;j<ndim;j++) {
				if (pt[j]<=rmid[j]) {
					fminl[j]=MIN(fminl[j],fval);
					fmaxl[j]=MAX(fmaxl[j],fval);
				} else {
					fminr[j]=MIN(fminr[j],fval);
					fmaxr[j]=MAX(fmaxr[j],fval);
				}
			}
		}
		sumb=BIG;
		jb= -1;
		siglb=sigrb=1.0;
		for (j=0;j<ndim;j++) {
			if (fmaxl[j] > fminl[j] && fmaxr[j] > fminr[j]) {
				sigl=MAX(TINY,pow(fmaxl[j]-fminl[j],2.0/3.0));
				sigr=MAX(TINY,pow(fmaxr[j]-fminr[j],2.0/3.0));
				sum=sigl+sigr;
				if (sum<=sumb) {
					sumb=sum;
					jb=j;
					siglb=sigl;
					sigrb=sigr;
				}
			}
		}
		if (jb == -1) jb=(ndim*iran)/175000;
		rgl=regn[jb];
		rgm=rmid[jb];
		rgr=regn[ndim+jb];
		fracl=abs((rgm-rgl)/(rgr-rgl));
		nptl=Int(MNPT+(npts-npre-2*MNPT)*fracl*siglb
			/(fracl*siglb+(1.0-fracl)*sigrb));
		nptr=npts-npre-nptl;
		VecDoub regn_temp(2*ndim);
		for (j=0;j<ndim;j++) {
			regn_temp[j]=regn[j];
			regn_temp[ndim+j]=regn[ndim+j];
		}
		regn_temp[ndim+jb]=rmid[jb];
		miser(func,regn_temp,nptl,dith,avel,varl);
		regn_temp[jb]=rmid[jb];
		regn_temp[ndim+jb]=regn[ndim+jb];
		miser(func,regn_temp,nptr,dith,ave,var);
		ave=fracl*avel+(1-fracl)*ave;
		var=fracl*fracl*varl+(1-fracl)*(1-fracl)*var;
	}
}
