struct Fitab {
	Int ndata;
	Doub a, b, siga, sigb, chi2, q, sigdat;
	VecDoub_I &x, &y, &sig;

	Fitab(VecDoub_I &xx, VecDoub_I &yy, VecDoub_I &ssig)
	: ndata(xx.size()), x(xx), y(yy), sig(ssig), chi2(0.), q(1.), sigdat(0.)  {
		Gamma gam;
		Int i;
		Doub ss=0.,sx=0.,sy=0.,st2=0.,t,wt,sxoss;
		b=0.0;
		for (i=0;i<ndata;i++) {
			wt=1.0/SQR(sig[i]);
			ss += wt;
			sx += x[i]*wt;
			sy += y[i]*wt;
		}
		sxoss=sx/ss;
		for (i=0;i<ndata;i++) {
			t=(x[i]-sxoss)/sig[i];
			st2 += t*t;
			b += t*y[i]/sig[i];
		}
		b /= st2;
		a=(sy-sx*b)/ss;
		siga=sqrt((1.0+sx*sx/(ss*st2))/ss);
		sigb=sqrt(1.0/st2);
		for (i=0;i<ndata;i++) chi2 += SQR((y[i]-a-b*x[i])/sig[i]);
		if (ndata>2) q=gam.gammq(0.5*(ndata-2),0.5*chi2);
	}

	Fitab(VecDoub_I &xx, VecDoub_I &yy)
	: ndata(xx.size()), x(xx), y(yy), sig(xx), chi2(0.), q(1.), sigdat(0.) {
		Int i;
		Doub ss,sx=0.,sy=0.,st2=0.,t,sxoss;
		b=0.0;
		for (i=0;i<ndata;i++) {
			sx += x[i];
			sy += y[i];
		}
		ss=ndata;
		sxoss=sx/ss;
		for (i=0;i<ndata;i++) {
			t=x[i]-sxoss;
			st2 += t*t;
			b += t*y[i];
		}
		b /= st2;
		a=(sy-sx*b)/ss;
		siga=sqrt((1.0+sx*sx/(ss*st2))/ss);
		sigb=sqrt(1.0/st2);
		for (i=0;i<ndata;i++) chi2 += SQR(y[i]-a-b*x[i]);
		if (ndata > 2) sigdat=sqrt(chi2/(ndata-2));
		siga *= sigdat;
		sigb *= sigdat;
	}
};
