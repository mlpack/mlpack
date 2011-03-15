Int fpoly_np = 10;

VecDoub fpoly(const Doub x) {
	Int j;
	VecDoub p(fpoly_np);
	p[0]=1.0;
	for (j=1;j<fpoly_np;j++) p[j]=p[j-1]*x;
	return p;
}
Int fleg_nl = 10;

VecDoub fleg(const Doub x) {
	Int j;
	Doub twox,f2,f1,d;
	VecDoub pl(fleg_nl);
	pl[0]=1.;
	pl[1]=x;
	if (fleg_nl > 2) {
		twox=2.*x;
		f2=x;
		d=1.;
		for (j=2;j<fleg_nl;j++) {
			f1=d++;
			f2+=twox;
			pl[j]=(f2*pl[j-1]-f1*pl[j-2])/d;
		}
	}
	return pl;
}
void fgauss(const Doub x, VecDoub_I &a, Doub &y, VecDoub_O &dyda) {
	Int i,na=a.size();
	Doub fac,ex,arg;
	y=0.;
	for (i=0;i<na-1;i+=3) {
		arg=(x-a[i+1])/a[i+2];
		ex=exp(-SQR(arg));
		fac=a[i]*ex*2.*arg;
		y += a[i]*ex;
		dyda[i]=ex;
		dyda[i+1]=fac/a[i+2];
		dyda[i+2]=fac*arg/a[i+2];
	}
}
