Complex frenel(const Doub x) {
	static const Int MAXIT=100;
	static const Doub PI=3.141592653589793238, PIBY2=(PI/2.0), XMIN=1.5,
		EPS=numeric_limits<Doub>::epsilon(),
		FPMIN=numeric_limits<Doub>::min(),
		BIG=numeric_limits<Doub>::max()*EPS;
	Bool odd;
	Int k,n;
	Doub a,ax,fact,pix2,sign,sum,sumc,sums,term,test;
	Complex b,cc,d,h,del,cs;
	if ((ax=abs(x)) < sqrt(FPMIN)) {
		cs=ax;
	} else if (ax <= XMIN) {
		sum=sums=0.0;
		sumc=ax;
		sign=1.0;
		fact=PIBY2*ax*ax;
		odd=true;
		term=ax;
		n=3;
		for (k=1;k<=MAXIT;k++) {
			term *= fact/k;
			sum += sign*term/n;
			test=abs(sum)*EPS;
			if (odd) {
				sign = -sign;
				sums=sum;
				sum=sumc;
			} else {
				sumc=sum;
				sum=sums;
			}
			if (term < test) break;
			odd=!odd;
			n += 2;
		}
		if (k > MAXIT) throw("series failed in frenel");
		cs=Complex(sumc,sums);
	} else {
		pix2=PI*ax*ax;
		b=Complex(1.0,-pix2);
		cc=BIG;
		d=h=1.0/b;
		n = -1;
		for (k=2;k<=MAXIT;k++) {
			n += 2;
			a = -n*(n+1);
			b += 4.0;
			d=1.0/(a*d+b);
			cc=b+a/cc;
			del=cc*d;
			h *= del;
			if (abs(real(del)-1.0)+abs(imag(del)) <= EPS) break;
		}
		if (k > MAXIT) throw("cf failed in frenel");
		h *= Complex(ax,-ax);
		cs=Complex(0.5,0.5)
			*(1.0-Complex(cos(0.5*pix2),sin(0.5*pix2))*h);
	}
	if (x < 0.0) cs = -cs;
	return cs;
}
