struct Fermi {
	Doub kk,etaa,thetaa;
	Doub operator() (const Doub t);
	Doub operator() (const Doub x, const Doub del);
	Doub val(const Doub k, const Doub eta, const Doub theta);
};

Doub Fermi::operator() (const Doub t) {
	Doub x;
	x=exp(t-exp(-t));
	return x*(1.0+exp(-t))*pow(x,kk)*sqrt(1.0+thetaa*0.5*x)/
		(exp(x-etaa)+1.0);
}

Doub Fermi::operator() (const Doub x, const Doub del) {
	if (x < 1.0)
		return pow(del,kk)*sqrt(1.0+thetaa*0.5*x)/(exp(x-etaa)+1.0);
	else
		return pow(x,kk)*sqrt(1.0+thetaa*0.5*x)/(exp(x-etaa)+1.0);
}

Doub Fermi::val(const Doub k, const Doub eta, const Doub theta)
{
	const Doub EPS=3.0e-9;
	const Int NMAX=11;
	Doub a,aa,b,bb,hmax,olds,sum;
	kk=k;
	etaa=eta;
	thetaa=theta;
	if (eta <= 15.0) {
		a=-4.5;
		b=5.0;
		Trapzd<Fermi> s(*this,a,b);
		for (Int i=1;i<=NMAX;i++) {
			sum=s.next();
			if (i > 3)
				if (abs(sum-olds) <= EPS*abs(olds))
					return sum;
			olds=sum;
		}
	}
	else {
		a=0.0;
		b=eta;
		aa=eta;
		bb=eta+60.0;
		hmax=4.3;
		DErule<Fermi> s(*this,a,b,hmax);
		DErule<Fermi> ss(*this,aa,bb,hmax);
		for (Int i=1;i<=NMAX;i++) {
			sum=s.next()+ss.next();
			if (i > 3)
				if (abs(sum-olds) <= EPS*abs(olds))
					return sum;
			olds=sum;
		}
	}
	throw("no convergence in fermi");
	return 0.0;
}
