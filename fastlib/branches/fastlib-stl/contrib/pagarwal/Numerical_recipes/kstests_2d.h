void quadct(const Doub x, const Doub y, VecDoub_I &xx, VecDoub_I &yy, Doub &fa,
	Doub &fb, Doub &fc, Doub &fd)
{
	Int k,na,nb,nc,nd,nn=xx.size();
	Doub ff;
	na=nb=nc=nd=0;
	for (k=0;k<nn;k++) {
		if (yy[k] == y && xx[k] == x) continue;
		if (yy[k] > y)
			xx[k] > x ? ++na : ++nb;
		else
			xx[k] > x ? ++nd : ++nc;
	}
	ff=1.0/nn;
	fa=ff*na;
	fb=ff*nb;
	fc=ff*nc;
	fd=ff*nd;
}
void ks2d1s(VecDoub_I &x1, VecDoub_I &y1, void quadvl(const Doub, const Doub,
	Doub &, Doub &, Doub &, Doub &), Doub &d1, Doub &prob)
{
	Int j,n1=x1.size();
	Doub dum,dumm,fa,fb,fc,fd,ga,gb,gc,gd,r1,rr,sqen;
	KSdist ks;
	d1=0.0;
	for (j=0;j<n1;j++) {
		quadct(x1[j],y1[j],x1,y1,fa,fb,fc,fd);
		quadvl(x1[j],y1[j],ga,gb,gc,gd);
		if (fa > ga) fa += 1.0/n1;
		if (fb > gb) fb += 1.0/n1;
		if (fc > gc) fc += 1.0/n1;
		if (fd > gd) fd += 1.0/n1;
		d1=MAX(d1,abs(fa-ga));
		d1=MAX(d1,abs(fb-gb));
		d1=MAX(d1,abs(fc-gc));
		d1=MAX(d1,abs(fd-gd));
	}
	pearsn(x1,y1,r1,dum,dumm);
	sqen=sqrt(Doub(n1));
	rr=sqrt(1.0-r1*r1);
	prob=ks.qks(d1*sqen/(1.0+rr*(0.25-0.75/sqen)));
}
void ks2d2s(VecDoub_I &x1, VecDoub_I &y1, VecDoub_I &x2, VecDoub_I &y2, Doub &d,
	Doub &prob)
{
	Int j,n1=x1.size(),n2=x2.size();
	Doub d1,d2,dum,dumm,fa,fb,fc,fd,ga,gb,gc,gd,r1,r2,rr,sqen;
	KSdist ks;
	d1=0.0;
	for (j=0;j<n1;j++) {
		quadct(x1[j],y1[j],x1,y1,fa,fb,fc,fd);
		quadct(x1[j],y1[j],x2,y2,ga,gb,gc,gd);
		if (fa > ga) fa += 1.0/n1;
		if (fb > gb) fb += 1.0/n1;
		if (fc > gc) fc += 1.0/n1;
		if (fd > gd) fd += 1.0/n1;
		d1=MAX(d1,abs(fa-ga));
		d1=MAX(d1,abs(fb-gb));
		d1=MAX(d1,abs(fc-gc));
		d1=MAX(d1,abs(fd-gd));
	}
	d2=0.0;
	for (j=0;j<n2;j++) {
		quadct(x2[j],y2[j],x1,y1,fa,fb,fc,fd);
		quadct(x2[j],y2[j],x2,y2,ga,gb,gc,gd);
		if (ga > fa) ga += 1.0/n1;
		if (gb > fb) gb += 1.0/n1;
		if (gc > fc) gc += 1.0/n1;
		if (gd > fd) gd += 1.0/n1;
		d2=MAX(d2,abs(fa-ga));
		d2=MAX(d2,abs(fb-gb));
		d2=MAX(d2,abs(fc-gc));
		d2=MAX(d2,abs(fd-gd));
	}
	d=0.5*(d1+d2);
	sqen=sqrt(n1*n2/Doub(n1+n2));
	pearsn(x1,y1,r1,dum,dumm);
	pearsn(x2,y2,r2,dum,dumm);
	rr=sqrt(1.0-0.5*(r1*r1+r2*r2));
	prob=ks.qks(d*sqen/(1.0+rr*(0.25-0.75/sqen)));
}
