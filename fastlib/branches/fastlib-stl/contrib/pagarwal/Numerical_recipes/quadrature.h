struct Quadrature{
	Int n;
	virtual Doub next() = 0;
};
template<class T>
struct Trapzd : Quadrature {
	Doub a,b,s;
	T &func;
	Trapzd() {};
	Trapzd(T &funcc, const Doub aa, const Doub bb) :
		func(funcc), a(aa), b(bb) {n=0;}
	Doub next() {
		Doub x,tnm,sum,del;
		Int it,j;
		n++;
		if (n == 1) {
			return (s=0.5*(b-a)*(func(a)+func(b)));
		} else {
			for (it=1,j=1;j<n-1;j++) it <<= 1;
			tnm=it;
			del=(b-a)/tnm;
			x=a+0.5*del;
			for (sum=0.0,j=0;j<it;j++,x+=del) sum += func(x);
			s=0.5*(s+(b-a)*sum/tnm);
			return s;
		}
	}
};
template<class T>
Doub qtrap(T &func, const Doub a, const Doub b, const Doub eps=1.0e-10) {
	const Int JMAX=20;
	Doub s,olds=0.0;
	Trapzd<T> t(func,a,b);
	for (Int j=0;j<JMAX;j++) {
		s=t.next();
		if (j > 5)
			if (abs(s-olds) < eps*abs(olds) ||
				(s == 0.0 && olds == 0.0)) return s;
		olds=s;
	}
	throw("Too many steps in routine qtrap");
}
template<class T>
Doub qsimp(T &func, const Doub a, const Doub b, const Doub eps=1.0e-10) {
	const Int JMAX=20;
	Doub s,st,ost=0.0,os=0.0;
	Trapzd<T> t(func,a,b);
	for (Int j=0;j<JMAX;j++) {
		st=t.next();
		s=(4.0*st-ost)/3.0;
		if (j > 5)
			if (abs(s-os) < eps*abs(os) ||
				(s == 0.0 && os == 0.0)) return s;
		os=s;
		ost=st;
	}
	throw("Too many steps in routine qsimp");
}
template <class T>
struct Midpnt : Quadrature {
	Doub a,b,s;
	T &funk;
	Midpnt(T &funcc, const Doub aa, const Doub bb) :
		funk(funcc), a(aa), b(bb) {n=0;}
	Doub next(){
		Int it,j;
		Doub x,tnm,sum,del,ddel;
		n++;
		if (n == 1) {
			return (s=(b-a)*func(0.5*(a+b)));
		} else {
			for(it=1,j=1;j<n-1;j++) it *= 3;
			tnm=it;
			del=(b-a)/(3.0*tnm);
			ddel=del+del;
			x=a+0.5*del;
			sum=0.0;
			for (j=0;j<it;j++) {
				sum += func(x);
				x += ddel;
				sum += func(x);
				x += del;
			}
			s=(s+(b-a)*sum/tnm)/3.0;
			return s;
		}
	}
	virtual Doub func(const Doub x) {return funk(x);}
};
template <class T>
struct Midinf : Midpnt<T>{
	Doub func(const Doub x) {
		return Midpnt<T>::funk(1.0/x)/(x*x);
	}
	Midinf(T &funcc, const Doub aa, const Doub bb) :
		Midpnt<T>(funcc, aa, bb) {
		Midpnt<T>::a=1.0/bb;
		Midpnt<T>::b=1.0/aa;
	}
};
template <class T>
struct Midsql : Midpnt<T>{
	Doub aorig;
	Doub func(const Doub x) {
		return 2.0*x*Midpnt<T>::funk(aorig+x*x);
	}
	Midsql(T &funcc, const Doub aa, const Doub bb) :
		Midpnt<T>(funcc, aa, bb), aorig(aa) {
		Midpnt<T>::a=0;
		Midpnt<T>::b=sqrt(bb-aa);
	}
};
template <class T>
struct Midsqu : Midpnt<T>{
	Doub borig;
	Doub func(const Doub x) {
		return 2.0*x*Midpnt<T>::funk(borig-x*x);
	}
	Midsqu(T &funcc, const Doub aa, const Doub bb) :
		Midpnt<T>(funcc, aa, bb), borig(bb) {
		Midpnt<T>::a=0;
		Midpnt<T>::b=sqrt(bb-aa);
	}
};
template <class T>
struct Midexp : Midpnt<T>{
	Doub func(const Doub x) {
		return Midpnt<T>::funk(-log(x))/x;
	}
	Midexp(T &funcc, const Doub aa, const Doub bb) :
		Midpnt<T>(funcc, aa, bb) {
		Midpnt<T>::a=0.0;
		Midpnt<T>::b=exp(-aa);
	}
};
