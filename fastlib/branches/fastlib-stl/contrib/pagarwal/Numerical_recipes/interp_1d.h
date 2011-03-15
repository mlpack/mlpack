struct Base_interp
{
	Int n, mm, jsav, cor, dj;
	const Doub *xx, *yy;
	Base_interp(VecDoub_I &x, const Doub *y, Int m)
		: n(x.size()), mm(m), jsav(0), cor(0), xx(&x[0]), yy(y) {
		dj = MIN(1,(int)pow((Doub)n,0.25));
	}

	Doub interp(Doub x) {
		Int jlo = cor ? hunt(x) : locate(x);
		return rawinterp(jlo,x);
	}

	Int locate(const Doub x);
	Int hunt(const Doub x);
	
	Doub virtual rawinterp(Int jlo, Doub x) = 0;

};
Int Base_interp::locate(const Doub x)
{
	Int ju,jm,jl;
	if (n < 2 || mm < 2 || mm > n) throw("locate size error");
	Bool ascnd=(xx[n-1] >= xx[0]);
	jl=0;
	ju=n-1;
	while (ju-jl > 1) {
		jm = (ju+jl) >> 1;
		if (x >= xx[jm] == ascnd)
			jl=jm;
		else
			ju=jm;
	}
	cor = abs(jl-jsav) > dj ? 0 : 1;
	jsav = jl;
	return MAX(0,MIN(n-mm,jl-((mm-2)>>1)));
}
Int Base_interp::hunt(const Doub x)
{
	Int jl=jsav, jm, ju, inc=1;
	if (n < 2 || mm < 2 || mm > n) throw("hunt size error");
	Bool ascnd=(xx[n-1] >= xx[0]);
	if (jl < 0 || jl > n-1) {
		jl=0;
		ju=n-1;
	} else {
		if (x >= xx[jl] == ascnd) {
			for (;;) {
				ju = jl + inc;
				if (ju >= n-1) { ju = n-1; break;}
				else if (x < xx[ju] == ascnd) break;
				else {
					jl = ju;
					inc += inc;
				}
			}
		} else {
			ju = jl;
			for (;;) {
				jl = jl - inc;
				if (jl <= 0) { jl = 0; break;}
				else if (x >= xx[jl] == ascnd) break;
				else {
					ju = jl;
					inc += inc;
				}
			}
		}
	}
	while (ju-jl > 1) {
		jm = (ju+jl) >> 1;
		if (x >= xx[jm] == ascnd)
			jl=jm;
		else
			ju=jm;
	}
	cor = abs(jl-jsav) > dj ? 0 : 1;
	jsav = jl;
	return MAX(0,MIN(n-mm,jl-((mm-2)>>1)));
}
struct Poly_interp : Base_interp
{
	Doub dy;
	Poly_interp(VecDoub_I &xv, VecDoub_I &yv, Int m)
		: Base_interp(xv,&yv[0],m), dy(0.) {}
	Doub rawinterp(Int jl, Doub x);
};

Doub Poly_interp::rawinterp(Int jl, Doub x)
{
	Int i,m,ns=0;
	Doub y,den,dif,dift,ho,hp,w;
	const Doub *xa = &xx[jl], *ya = &yy[jl];
	VecDoub c(mm),d(mm);
	dif=abs(x-xa[0]);
	for (i=0;i<mm;i++) {
		if ((dift=abs(x-xa[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	y=ya[ns--];
	for (m=1;m<mm;m++) {
		for (i=0;i<mm-m;i++) {
			ho=xa[i]-x;
			hp=xa[i+m]-x;
			w=c[i+1]-d[i];
			if ((den=ho-hp) == 0.0) throw("Poly_interp error");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		y += (dy=(2*(ns+1) < (mm-m) ? c[ns+1] : d[ns--]));
	}
	return y;
}
struct Rat_interp : Base_interp
{
	Doub dy;
	Rat_interp(VecDoub_I &xv, VecDoub_I &yv, Int m)
		: Base_interp(xv,&yv[0],m), dy(0.) {}
	Doub rawinterp(Int jl, Doub x);
};

Doub Rat_interp::rawinterp(Int jl, Doub x)
{
	const Doub TINY=1.0e-99;
	Int m,i,ns=0;
	Doub y,w,t,hh,h,dd;
	const Doub *xa = &xx[jl], *ya = &yy[jl];
	VecDoub c(mm),d(mm);
	hh=abs(x-xa[0]);
	for (i=0;i<mm;i++) {
		h=abs(x-xa[i]);
		if (h == 0.0) {
			dy=0.0;
			return ya[i];
		} else if (h < hh) {
			ns=i;
			hh=h;
		}
		c[i]=ya[i];
		d[i]=ya[i]+TINY;
	}
	y=ya[ns--];
	for (m=1;m<mm;m++) {
		for (i=0;i<mm-m;i++) {
			w=c[i+1]-d[i];
			h=xa[i+m]-x;
			t=(xa[i]-x)*d[i]/h;
			dd=t-c[i+1];
			if (dd == 0.0) throw("Error in routine ratint");
			dd=w/dd;
			d[i]=c[i+1]*dd;
			c[i]=t*dd;
		}
		y += (dy=(2*(ns+1) < (mm-m) ? c[ns+1] : d[ns--]));
	}
	return y;
}
struct Spline_interp : Base_interp
{
	VecDoub y2;
	
	Spline_interp(VecDoub_I &xv, VecDoub_I &yv, Doub yp1=1.e99, Doub ypn=1.e99)
	: Base_interp(xv,&yv[0],2), y2(xv.size())
	{sety2(&xv[0],&yv[0],yp1,ypn);}

	Spline_interp(VecDoub_I &xv, const Doub *yv, Doub yp1=1.e99, Doub ypn=1.e99)
	: Base_interp(xv,yv,2), y2(xv.size())
	{sety2(&xv[0],yv,yp1,ypn);}

	void sety2(const Doub *xv, const Doub *yv, Doub yp1, Doub ypn);
	Doub rawinterp(Int jl, Doub xv);
};
void Spline_interp::sety2(const Doub *xv, const Doub *yv, Doub yp1, Doub ypn)
{
	Int i,k;
	Doub p,qn,sig,un;
	Int n=y2.size();
	VecDoub u(n-1);
	if (yp1 > 0.99e99)
		y2[0]=u[0]=0.0;
	else {
		y2[0] = -0.5;
		u[0]=(3.0/(xv[1]-xv[0]))*((yv[1]-yv[0])/(xv[1]-xv[0])-yp1);
	}
	for (i=1;i<n-1;i++) {
		sig=(xv[i]-xv[i-1])/(xv[i+1]-xv[i-1]);
		p=sig*y2[i-1]+2.0;
		y2[i]=(sig-1.0)/p;
		u[i]=(yv[i+1]-yv[i])/(xv[i+1]-xv[i]) - (yv[i]-yv[i-1])/(xv[i]-xv[i-1]);
		u[i]=(6.0*u[i]/(xv[i+1]-xv[i-1])-sig*u[i-1])/p;
	}
	if (ypn > 0.99e99)
		qn=un=0.0;
	else {
		qn=0.5;
		un=(3.0/(xv[n-1]-xv[n-2]))*(ypn-(yv[n-1]-yv[n-2])/(xv[n-1]-xv[n-2]));
	}
	y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0);
	for (k=n-2;k>=0;k--)
		y2[k]=y2[k]*y2[k+1]+u[k];
}
Doub Spline_interp::rawinterp(Int jl, Doub x)
{
	Int klo=jl,khi=jl+1;
	Doub y,h,b,a;
	h=xx[khi]-xx[klo];
	if (h == 0.0) throw("Bad input to routine splint");
	a=(xx[khi]-x)/h;
	b=(x-xx[klo])/h;
	y=a*yy[klo]+b*yy[khi]+((a*a*a-a)*y2[klo]
		+(b*b*b-b)*y2[khi])*(h*h)/6.0;
	return y;
}
struct BaryRat_interp : Base_interp
{
	VecDoub w;
	Int d;
	BaryRat_interp(VecDoub_I &xv, VecDoub_I &yv, Int dd);
	Doub rawinterp(Int jl, Doub x);
	Doub interp(Doub x);
};

BaryRat_interp::BaryRat_interp(VecDoub_I &xv, VecDoub_I &yv, Int dd)
		: Base_interp(xv,&yv[0],xv.size()), w(n), d(dd)
{
	if (n<=d) throw("d too large for number of points in BaryRat_interp");
	for (Int k=0;k<n;k++) {
		Int imin=MAX(k-d,0);
		Int imax = k >= n-d ? n-d-1 : k;
		Doub temp = imin & 1 ? -1.0 : 1.0;
		Doub sum=0.0;
		for (Int i=imin;i<=imax;i++) {
			Int jmax=MIN(i+d,n-1);
			Doub term=1.0;
			for (Int j=i;j<=jmax;j++) {
				if (j==k) continue;
				term *= (xx[k]-xx[j]);
			}
			term=temp/term;
			temp=-temp;
			sum += term;
		}
		w[k]=sum;
	}
}
Doub BaryRat_interp::rawinterp(Int jl, Doub x)
{
	Doub num=0,den=0;
	for (Int i=0;i<n;i++) {
		Doub h=x-xx[i];
		if (h == 0.0) {
			return yy[i];
		} else {
			Doub temp=w[i]/h;
			num += temp*yy[i];
			den += temp;
		}
	}
	return num/den;
}
Doub BaryRat_interp::interp(Doub x) {
	return rawinterp(1,x);
}
