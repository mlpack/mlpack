void memcof(VecDoub_I &data, Doub &xms, VecDoub_O &d) {
	Int k,j,i,n=data.size(),m=d.size();
	Doub p=0.0;
	VecDoub wk1(n),wk2(n),wkm(m);
	for (j=0;j<n;j++) p += SQR(data[j]);
	xms=p/n;
	wk1[0]=data[0];
	wk2[n-2]=data[n-1];
	for (j=1;j<n-1;j++) {
		wk1[j]=data[j];
		wk2[j-1]=data[j];
	}
	for (k=0;k<m;k++) {
		Doub num=0.0,denom=0.0;
		for (j=0;j<(n-k-1);j++) {
			num += (wk1[j]*wk2[j]);
			denom += (SQR(wk1[j])+SQR(wk2[j]));
		}
		d[k]=2.0*num/denom;
		xms *= (1.0-SQR(d[k]));
		for (i=0;i<k;i++)
			d[i]=wkm[i]-d[k]*wkm[k-1-i];
		if (k == m-1)
			return;
		for (i=0;i<=k;i++) wkm[i]=d[i];
		for (j=0;j<(n-k-2);j++) {
			wk1[j] -= (wkm[k]*wk2[j]);
			wk2[j]=wk2[j+1]-wkm[k]*wk1[j+1];
		}
	}
	throw("never get here in memcof");
}
void fixrts(VecDoub_IO &d) {
	Bool polish=true;
	Int i,j,m=d.size();
	VecComplex a(m+1),roots(m);
	a[m]=1.0;
	for (j=0;j<m;j++)
		a[j]= -d[m-1-j];
	zroots(a,roots,polish);
	for (j=0;j<m;j++)
		if (abs(roots[j]) > 1.0)
			roots[j]=1.0/conj(roots[j]);
	a[0]= -roots[0];
	a[1]=1.0;
	for (j=1;j<m;j++) {
		a[j+1]=1.0;
		for (i=j;i>=1;i--)
			a[i]=a[i-1]-roots[j]*a[i];
		a[0]= -roots[j]*a[0];
	}
	for (j=0;j<m;j++)
		d[m-1-j] = -real(a[j]);
}
void predic(VecDoub_I &data, VecDoub_I &d, VecDoub_O &future) {
	Int k,j,ndata=data.size(),m=d.size(),nfut=future.size();
	Doub sum,discrp;
	VecDoub reg(m);
	for (j=0;j<m;j++) reg[j]=data[ndata-1-j];
	for (j=0;j<nfut;j++) {
		discrp=0.0;
		sum=discrp;
		for (k=0;k<m;k++) sum += d[k]*reg[k];
		for (k=m-1;k>=1;k--) reg[k]=reg[k-1];
		future[j]=reg[0]=sum;
	}
}
Doub evlmem(const Doub fdt, VecDoub_I &d, const Doub xms)
{
	Int i;
	Doub sumr=1.0,sumi=0.0,wr=1.0,wi=0.0,wpr,wpi,wtemp,theta;

	Int m=d.size();
	theta=6.28318530717959*fdt;
	wpr=cos(theta);
	wpi=sin(theta);
	for (i=0;i<m;i++) {
		wr=(wtemp=wr)*wpr-wi*wpi;
		wi=wi*wpr+wtemp*wpi;
		sumr -= d[i]*wr;
		sumi -= d[i]*wi;
	}
	return xms/(sumr*sumr+sumi*sumi);
}
