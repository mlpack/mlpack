void ttest(VecDoub_I &data1, VecDoub_I &data2, Doub &t, Doub &prob)
{
	Beta beta;
	Doub var1,var2,svar,df,ave1,ave2;
	Int n1=data1.size(), n2=data2.size();
	avevar(data1,ave1,var1);
	avevar(data2,ave2,var2);
	df=n1+n2-2;
	svar=((n1-1)*var1+(n2-1)*var2)/df;
	t=(ave1-ave2)/sqrt(svar*(1.0/n1+1.0/n2));
	prob=beta.betai(0.5*df,0.5,df/(df+t*t));
}
void tutest(VecDoub_I &data1, VecDoub_I &data2, Doub &t, Doub &prob) {
	Beta beta;
	Doub var1,var2,df,ave1,ave2;
	Int n1=data1.size(), n2=data2.size();
	avevar(data1,ave1,var1);
	avevar(data2,ave2,var2);
	t=(ave1-ave2)/sqrt(var1/n1+var2/n2);
	df=SQR(var1/n1+var2/n2)/(SQR(var1/n1)/(n1-1)+SQR(var2/n2)/(n2-1));
	prob=beta.betai(0.5*df,0.5,df/(df+SQR(t)));
}
void tptest(VecDoub_I &data1, VecDoub_I &data2, Doub &t, Doub &prob) {
	Beta beta;
	Int j, n=data1.size();
	Doub var1,var2,ave1,ave2,sd,df,cov=0.0;
	avevar(data1,ave1,var1);
	avevar(data2,ave2,var2);
	for (j=0;j<n;j++) cov += (data1[j]-ave1)*(data2[j]-ave2);
	cov /= (df=n-1);
	sd=sqrt((var1+var2-2.0*cov)/n);
	t=(ave1-ave2)/sd;
	prob=beta.betai(0.5*df,0.5,df/(df+t*t));
}
void ftest(VecDoub_I &data1, VecDoub_I &data2, Doub &f, Doub &prob) {
	Beta beta;
	Doub var1,var2,ave1,ave2,df1,df2;
	Int n1=data1.size(), n2=data2.size();
	avevar(data1,ave1,var1);
	avevar(data2,ave2,var2);
	if (var1 > var2) {
		f=var1/var2;
		df1=n1-1;
		df2=n2-1;
	} else {
		f=var2/var1;
		df1=n2-1;
		df2=n1-1;
	}
	prob = 2.0*beta.betai(0.5*df2,0.5*df1,df2/(df2+df1*f));
	if (prob > 1.0) prob=2.-prob;
}
void chsone(VecDoub_I &bins, VecDoub_I &ebins, Doub &df,
	Doub &chsq, Doub &prob, const Int knstrn=1) {
	Gamma gam;
	Int j,nbins=bins.size();
	Doub temp;
	df=nbins-knstrn;
	chsq=0.0;
	for (j=0;j<nbins;j++) {
		if (ebins[j]<0.0 || (ebins[j]==0. && bins[j]>0.))
			throw("Bad expected number in chsone");
		if (ebins[j]==0.0 && bins[j]==0.0) {
			--df;
		} else {
			temp=bins[j]-ebins[j];
			chsq += temp*temp/ebins[j];
		}
	}
	prob=gam.gammq(0.5*df,0.5*chsq);
}
void chstwo(VecDoub_I &bins1, VecDoub_I &bins2, Doub &df,
	Doub &chsq, Doub &prob, const Int knstrn=1) {
	Gamma gam;
	Int j,nbins=bins1.size();
	Doub temp;
	df=nbins-knstrn;
	chsq=0.0;
	for (j=0;j<nbins;j++)
		if (bins1[j] == 0.0 && bins2[j] == 0.0)
			--df;
		else {
			temp=bins1[j]-bins2[j];
			chsq += temp*temp/(bins1[j]+bins2[j]);
		}
	prob=gam.gammq(0.5*df,0.5*chsq);
}
void cntab(MatInt_I &nn, Doub &chisq, Doub &df, Doub &prob, Doub &cramrv,
	Doub &ccc)
{
	const Doub TINY=1.0e-30;
	Gamma gam;
	Int i,j,nnj,nni,minij,ni=nn.nrows(),nj=nn.ncols();
	Doub sum=0.0,expctd,temp;
	VecDoub sumi(ni),sumj(nj);
	nni=ni;
	nnj=nj;
	for (i=0;i<ni;i++) {
		sumi[i]=0.0;
		for (j=0;j<nj;j++) {
			sumi[i] += nn[i][j];
			sum += nn[i][j];
		}
		if (sumi[i] == 0.0) --nni;
	}
	for (j=0;j<nj;j++) {
		sumj[j]=0.0;
		for (i=0;i<ni;i++) sumj[j] += nn[i][j];
		if (sumj[j] == 0.0) --nnj;
	}
	df=nni*nnj-nni-nnj+1;
	chisq=0.0;
	for (i=0;i<ni;i++) {
		for (j=0;j<nj;j++) {
			expctd=sumj[j]*sumi[i]/sum;
			temp=nn[i][j]-expctd;
			chisq += temp*temp/(expctd+TINY);
		}
	}
	prob=gam.gammq(0.5*df,0.5*chisq);
	minij = nni < nnj ? nni-1 : nnj-1;
	cramrv=sqrt(chisq/(sum*minij));
	ccc=sqrt(chisq/(chisq+sum));
}
void pearsn(VecDoub_I &x, VecDoub_I &y, Doub &r, Doub &prob, Doub &z)
{
	const Doub TINY=1.0e-20;
	Beta beta;
	Int j,n=x.size();
	Doub yt,xt,t,df;
	Doub syy=0.0,sxy=0.0,sxx=0.0,ay=0.0,ax=0.0;
	for (j=0;j<n;j++) {
		ax += x[j];
		ay += y[j];
	}
	ax /= n;
	ay /= n;
	for (j=0;j<n;j++) {
		xt=x[j]-ax;
		yt=y[j]-ay;
		sxx += xt*xt;
		syy += yt*yt;
		sxy += xt*yt;
	}
	r=sxy/(sqrt(sxx*syy)+TINY);
	z=0.5*log((1.0+r+TINY)/(1.0-r+TINY));
	df=n-2;
	t=r*sqrt(df/((1.0-r+TINY)*(1.0+r+TINY)));
	prob=beta.betai(0.5*df,0.5,df/(df+t*t));
	// prob=erfcc(abs(z*sqrt(n-1.0))/1.4142136);
}
void crank(VecDoub_IO &w, Doub &s)
{
	Int j=1,ji,jt,n=w.size();
	Doub t,rank;
	s=0.0;
	while (j < n) {
		if (w[j] != w[j-1]) {
			w[j-1]=j;
			++j;
		} else {
			for (jt=j+1;jt<=n && w[jt-1]==w[j-1];jt++);
			rank=0.5*(j+jt-1);
			for (ji=j;ji<=(jt-1);ji++)
				w[ji-1]=rank;
			t=jt-j;
			s += (t*t*t-t);
			j=jt;
		}
	}
	if (j == n) w[n-1]=n;
}
void spear(VecDoub_I &data1, VecDoub_I &data2, Doub &d, Doub &zd, Doub &probd,
	Doub &rs, Doub &probrs)
{
	Beta bet;
	Int j,n=data1.size();
	Doub vard,t,sg,sf,fac,en3n,en,df,aved;
	VecDoub wksp1(n),wksp2(n);
	for (j=0;j<n;j++) {
		wksp1[j]=data1[j];
		wksp2[j]=data2[j];
	}
	sort2(wksp1,wksp2);
	crank(wksp1,sf);
	sort2(wksp2,wksp1);
	crank(wksp2,sg);
	d=0.0;
	for (j=0;j<n;j++)
		d += SQR(wksp1[j]-wksp2[j]);
	en=n;
	en3n=en*en*en-en;
	aved=en3n/6.0-(sf+sg)/12.0;
	fac=(1.0-sf/en3n)*(1.0-sg/en3n);
	vard=((en-1.0)*en*en*SQR(en+1.0)/36.0)*fac;
	zd=(d-aved)/sqrt(vard);
	probd=erfcc(abs(zd)/1.4142136);
	rs=(1.0-(6.0/en3n)*(d+(sf+sg)/12.0))/sqrt(fac);
	fac=(rs+1.0)*(1.0-rs);
	if (fac > 0.0) {
		t=rs*sqrt((en-2.0)/fac);
		df=en-2.0;
		probrs=bet.betai(0.5*df,0.5,df/(df+t*t));
	} else
		probrs=0.0;
}
void kendl1(VecDoub_I &data1, VecDoub_I &data2, Doub &tau, Doub &z, Doub &prob)
{
	Int is=0,j,k,n2=0,n1=0,n=data1.size();
	Doub svar,aa,a2,a1;
	for (j=0;j<n-1;j++) {
		for (k=j+1;k<n;k++) {
			a1=data1[j]-data1[k];
			a2=data2[j]-data2[k];
			aa=a1*a2;
			if (aa != 0.0) {
				++n1;
				++n2;
				aa > 0.0 ? ++is : --is;
			} else {
				if (a1 != 0.0) ++n1;
				if (a2 != 0.0) ++n2;
			}
		}
	}
	tau=is/(sqrt(Doub(n1))*sqrt(Doub(n2)));
	svar=(4.0*n+10.0)/(9.0*n*(n-1.0));
	z=tau/sqrt(svar);
	prob=erfcc(abs(z)/1.4142136);
}
void kendl2(MatDoub_I &tab, Doub &tau, Doub &z, Doub &prob)
{
	Int k,l,nn,mm,m2,m1,lj,li,kj,ki,i=tab.nrows(),j=tab.ncols();
	Doub svar,s=0.0,points,pairs,en2=0.0,en1=0.0;
	nn=i*j;
	points=tab[i-1][j-1];
	for (k=0;k<=nn-2;k++) {
		ki=(k/j);
		kj=k-j*ki;
		points += tab[ki][kj];
		for (l=k+1;l<=nn-1;l++) {
			li=l/j;
			lj=l-j*li;
			mm=(m1=li-ki)*(m2=lj-kj);
			pairs=tab[ki][kj]*tab[li][lj];
			if (mm != 0) {
				en1 += pairs;
				en2 += pairs;
				s += (mm > 0 ? pairs : -pairs);
			} else {
				if (m1 != 0) en1 += pairs;
				if (m2 != 0) en2 += pairs;
			}
		}
	}
	tau=s/sqrt(en1*en2);
	svar=(4.0*points+10.0)/(9.0*points*(points-1.0));
	z=tau/sqrt(svar);
	prob=erfcc(abs(z)/1.4142136);
}
