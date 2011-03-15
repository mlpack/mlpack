void laguer(VecComplex_I &a, Complex &x, Int &its) {
	const Int MR=8,MT=10,MAXIT=MT*MR;
	const Doub EPS=numeric_limits<Doub>::epsilon();
	static const Doub frac[MR+1]=
		{0.0,0.5,0.25,0.75,0.13,0.38,0.62,0.88,1.0};
	Complex dx,x1,b,d,f,g,h,sq,gp,gm,g2;
	Int m=a.size()-1;
	for (Int iter=1;iter<=MAXIT;iter++) {
		its=iter;
		b=a[m];
		Doub err=abs(b);
		d=f=0.0;
		Doub abx=abs(x);
		for (Int j=m-1;j>=0;j--) {
			f=x*f+d;
			d=x*d+b;
			b=x*b+a[j];
			err=abs(b)+abx*err;
		}
		err *= EPS;
		if (abs(b) <= err) return;
		g=d/b;
		g2=g*g;
		h=g2-2.0*f/b;
		sq=sqrt(Doub(m-1)*(Doub(m)*h-g2));
		gp=g+sq;
		gm=g-sq;
		Doub abp=abs(gp);
		Doub abm=abs(gm);
		if (abp < abm) gp=gm;
		dx=MAX(abp,abm) > 0.0 ? Doub(m)/gp : polar(1+abx,Doub(iter));
		x1=x-dx;
		if (x == x1) return;
		if (iter % MT != 0) x=x1;
		else x -= frac[iter/MT]*dx;
	}
	throw("too many iterations in laguer");
}
void zroots(VecComplex_I &a, VecComplex_O &roots, const Bool &polish)
{
	const Doub EPS=1.0e-14;
	Int i,its;
	Complex x,b,c;
	Int m=a.size()-1;
	VecComplex ad(m+1);
	for (Int j=0;j<=m;j++) ad[j]=a[j];
	for (Int j=m-1;j>=0;j--) {
		x=0.0;
		VecComplex ad_v(j+2);
		for (Int jj=0;jj<j+2;jj++) ad_v[jj]=ad[jj];
		laguer(ad_v,x,its);
		if (abs(imag(x)) <= 2.0*EPS*abs(real(x)))
			x=Complex(real(x),0.0);
		roots[j]=x;
		b=ad[j+1];
		for (Int jj=j;jj>=0;jj--) {
			c=ad[jj];
			ad[jj]=b;
			b=x*b+c;
		}
	}
	if (polish)
		for (Int j=0;j<m;j++)
			laguer(a,roots[j],its);
	for (Int j=1;j<m;j++) {
		x=roots[j];
		for (i=j-1;i>=0;i--) {
			if (real(roots[i]) <= real(x)) break;
			roots[i+1]=roots[i];
		}
		roots[i+1]=x;
	}
}
