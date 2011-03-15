void rk4(VecDoub_I &y, VecDoub_I &dydx, const Doub x, const Doub h,
	VecDoub_O &yout, void derivs(const Doub, VecDoub_I &, VecDoub_O &))
{
	Int n=y.size();
	VecDoub dym(n),dyt(n),yt(n);
	Doub hh=h*0.5;
	Doub h6=h/6.0;
	Doub xh=x+hh;
	for (Int i=0;i<n;i++) yt[i]=y[i]+hh*dydx[i];
	derivs(xh,yt,dyt);
	for (Int i=0;i<n;i++) yt[i]=y[i]+hh*dyt[i];
	derivs(xh,yt,dym);
	for (Int i=0;i<n;i++) {
		yt[i]=y[i]+h*dym[i];
		dym[i] += dyt[i];
	}
	derivs(x+h,yt,dyt);
	for (Int i=0;i<n;i++)
		yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}
