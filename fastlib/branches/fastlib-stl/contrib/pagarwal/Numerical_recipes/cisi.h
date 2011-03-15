Complex cisi(const Doub x) {
	static const Int MAXIT=100;
	static const Doub EULER=0.577215664901533, PIBY2=1.570796326794897,
		TMIN=2.0, EPS=numeric_limits<Doub>::epsilon(),
		FPMIN=numeric_limits<Doub>::min()*4.0,
		BIG=numeric_limits<Doub>::max()*EPS;
	Int i,k;
	Bool odd;
	Doub a,err,fact,sign,sum,sumc,sums,t,term;
	Complex h,b,c,d,del,cs;
	if ((t=abs(x)) == 0.0) return -BIG;
	if (t > TMIN) {
		b=Complex(1.0,t);
		c=Complex(BIG,0.0);
		d=h=1.0/b;
		for (i=1;i<MAXIT;i++) {
			a= -i*i;
			b += 2.0;
			d=1.0/(a*d+b);
			c=b+a/c;
			del=c*d;
			h *= del;
			if (abs(real(del)-1.0)+abs(imag(del)) <= EPS) break;
		}
		if (i >= MAXIT) throw("cf failed in cisi");
		h=Complex(cos(t),-sin(t))*h;
		cs= -conj(h)+Complex(0.0,PIBY2);
	} else {
		if (t < sqrt(FPMIN)) {
			sumc=0.0;
			sums=t;
		} else {
			sum=sums=sumc=0.0;
			sign=fact=1.0;
			odd=true;
			for (k=1;k<=MAXIT;k++) {
				fact *= t/k;
				term=fact/k;
				sum += sign*term;
				err=term/abs(sum);
				if (odd) {
					sign = -sign;
					sums=sum;
					sum=sumc;
				} else {
					sumc=sum;
					sum=sums;
				}
				if (err < EPS) break;
				odd=!odd;
			}
			if (k > MAXIT) throw("maxits exceeded in cisi");
		}
		cs=Complex(sumc+log(t)+EULER,sums);
	}
	if (x < 0.0) cs = conj(cs);
	return cs;
}
