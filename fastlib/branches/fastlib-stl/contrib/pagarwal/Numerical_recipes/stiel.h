struct Stiel {
	struct pp {
		Stiel *st;
		Doub operator() (const Doub x, const Doub del)
		{
			Doub pval=st->p(x);
			return pval*(st->wt1)(x,del)*pval;
		}
		Doub operator() (const Doub t)
		{
			Doub x=(st->fx)(t);
			Doub pval=st->p(x);
			return pval*(st->wt2)(x)*(st->fdxdt)(t)*pval;
		}
	};
	struct ppx {
		Stiel *st;
		Doub operator() (const Doub x, const Doub del)
		{
			return st->ppfunc(x,del)*x;
		}
		Doub operator() (const Doub t)
		{
			return st->ppfunc(t)*(st->fx)(t);
		}
	};
	Int j,n;
	Doub aa,bb,hmax;
	Doub (*wt1)(const Doub x, const Doub del);
	Doub (*wt2)(const Doub x);
	Doub (*fx)(const Doub t);
	Doub (*fdxdt)(const Doub t);
	VecDoub a,b;
	Quadrature *s1,*s2;
	Doub p(const Doub x);
	pp ppfunc;
	ppx ppxfunc;
	Stiel(Int nn, Doub aaa, Doub bbb, Doub hmaxx, Doub wwt1(Doub,Doub));
	Stiel(Int nn, Doub aaa, Doub bbb, Doub wwt2(Doub), Doub ffx(Doub),
		Doub ffdxdt(Doub));
	Doub quad(Quadrature *s);
	void get_weights(VecDoub_O &x, VecDoub_O &w);
};
Doub Stiel::p(const Doub x)
{
	Doub pval,pj,pjm1;
	if (j == 0)
		return 1.0;
	else {
		pjm1=0.0;
		pj=1.0;
		for (Int i=0;i<j;i++) {
			pval=(x-a[i])*pj-b[i]*pjm1;
			pjm1=pj;
			pj=pval;
		}
	}
	return pval;
}

Stiel::Stiel(Int nn, Doub aaa, Doub bbb, Doub hmaxx, Doub wwt1(Doub,Doub)) :
	n(nn), aa(aaa), bb(bbb), hmax(hmaxx), wt1(wwt1), a(nn), b(nn) {
	ppfunc.st=this;
	ppxfunc.st=this;
	s1=new DErule<pp>(ppfunc,aa,bb,hmax);
	s2=new DErule<ppx>(ppxfunc,aa,bb,hmax);
}
Stiel::Stiel(Int nn, Doub aaa, Doub bbb, Doub wwt2(Doub), Doub ffx(Doub),
	Doub ffdxdt(Doub)) : n(nn), aa(aaa), bb(bbb), a(nn), b(nn), wt2(wwt2),
	fx(ffx), fdxdt(ffdxdt) {
	ppfunc.st=this;
	ppxfunc.st=this;
	s1=new Trapzd<pp>(ppfunc,aa,bb);
	s2=new Trapzd<ppx>(ppxfunc,aa,bb);
}

Doub Stiel::quad(Quadrature *s)
{
	const Doub EPS=3.0e-11, MACHEPS=numeric_limits<Doub>::epsilon();
	const Int NMAX=11;
	Doub olds,sum;
	s->n=0;
	for (Int i=1;i<=NMAX;i++) {
		sum=s->next();
		if (i > 3)
			if (abs(sum-olds) <= EPS*abs(olds))
				return sum;
		if (i == NMAX)
			if (abs(sum) <= MACHEPS && abs(olds) <= MACHEPS)
				return 0.0;		
		olds=sum;
	}
	throw("no convergence in quad");
	return 0.0;
}

void Stiel::get_weights(VecDoub_O &x, VecDoub_O &w)
{
	Doub amu0,c,oldc=1.0;
	if (n != x.size()) throw("bad array size in Stiel");
	for (Int i=0;i<n;i++) {
		j=i;
		c=quad(s1);
		b[i]=c/oldc;
		a[i]=quad(s2)/c;
		oldc=c;
	}
	amu0=b[0];
	gaucof(a,b,amu0,x,w);
}

