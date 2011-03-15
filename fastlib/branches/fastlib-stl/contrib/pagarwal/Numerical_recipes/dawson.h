Doub dawson(const Doub x) {
	static const Int NMAX=6;
	static VecDoub c(NMAX);
	static Bool init = true;
	static const Doub H=0.4, A1=2.0/3.0, A2=0.4, A3=2.0/7.0;
	Int i,n0;
	Doub d1,d2,e1,e2,sum,x2,xp,xx,ans;
	if (init) {
		init=false;
		for (i=0;i<NMAX;i++) c[i]=exp(-SQR((2.0*i+1.0)*H));
	}
	if (abs(x) < 0.2) {
		x2=x*x;
		ans=x*(1.0-A1*x2*(1.0-A2*x2*(1.0-A3*x2)));
	} else {
		xx=abs(x);
		n0=2*Int(0.5*xx/H+0.5);
		xp=xx-n0*H;
		e1=exp(2.0*xp*H);
		e2=e1*e1;
		d1=n0+1;
		d2=d1-2.0;
		sum=0.0;
		for (i=0;i<NMAX;i++,d1+=2.0,d2-=2.0,e1*=e2)
			sum += c[i]*(e1/d1+1.0/(d2*e1));
		ans=0.5641895835*SIGN(exp(-xp*xp),x)*sum;
	}
	return ans;
}
