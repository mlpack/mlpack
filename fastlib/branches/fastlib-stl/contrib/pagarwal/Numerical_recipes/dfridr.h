template<class T>
Doub dfridr(T &func, const Doub x, const Doub h, Doub &err)
{
	const Int ntab=10;
	const Doub con=1.4, con2=(con*con);
	const Doub big=numeric_limits<Doub>::max();
	const Doub safe=2.0;
	Int i,j;
	Doub errt,fac,hh,ans;
	MatDoub a(ntab,ntab);
	if (h == 0.0) throw("h must be nonzero in dfridr.");
	hh=h;
	a[0][0]=(func(x+hh)-func(x-hh))/(2.0*hh);
	err=big;
	for (i=1;i<ntab;i++) {
		hh /= con;
		a[0][i]=(func(x+hh)-func(x-hh))/(2.0*hh);
		fac=con2;
		for (j=1;j<=i;j++) {
			a[j][i]=(a[j-1][i]*fac-a[j-1][i-1])/(fac-1.0);
			fac=con2*fac;
			errt=MAX(abs(a[j][i]-a[j-1][i]),abs(a[j][i]-a[j-1][i-1]));
			if (errt <= err) {
				err=errt;
				ans=a[j][i];
			}
		}
		if (abs(a[i][i]-a[i-1][i-1]) >= safe*err) break;
	}
	return ans;
}
