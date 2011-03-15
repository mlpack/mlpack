template <class T>
Bool zbrac(T &func, Doub &x1, Doub &x2)
{
	const Int NTRY=50;
	const Doub FACTOR=1.6;
	if (x1 == x2) throw("Bad initial range in zbrac");
	Doub f1=func(x1);
	Doub f2=func(x2);
	for (Int j=0;j<NTRY;j++) {
		if (f1*f2 < 0.0) return true;
		if (abs(f1) < abs(f2))
			f1=func(x1 += FACTOR*(x1-x2));
		else
			f2=func(x2 += FACTOR*(x2-x1));
	}
	return false;
}
template <class T>
void zbrak(T &fx, const Doub x1, const Doub x2, const Int n, VecDoub_O &xb1,
	VecDoub_O &xb2, Int &nroot)
{
	Int nb=20;
	xb1.resize(nb);
	xb2.resize(nb);
	nroot=0;
	Doub dx=(x2-x1)/n;
	Doub x=x1;
	Doub fp=fx(x1);
	for (Int i=0;i<n;i++) {
		Doub fc=fx(x += dx);
		if (fc*fp <= 0.0) {
			xb1[nroot]=x-dx;
			xb2[nroot++]=x;
			if(nroot == nb) {
				VecDoub tempvec1(xb1),tempvec2(xb2);
				xb1.resize(2*nb);
				xb2.resize(2*nb);
				for (Int j=0; j<nb; j++) {
					xb1[j]=tempvec1[j];
					xb2[j]=tempvec2[j];
				}
				nb *= 2;
			}
		}
		fp=fc;
	}
}
template <class T>
Doub rtbis(T &func, const Doub x1, const Doub x2, const Doub xacc) {
	const Int JMAX=50;
	Doub dx,xmid,rtb;
	Doub f=func(x1);
	Doub fmid=func(x2);
	if (f*fmid >= 0.0) throw("Root must be bracketed for bisection in rtbis");
	rtb = f < 0.0 ? (dx=x2-x1,x1) : (dx=x1-x2,x2);
	for (Int j=0;j<JMAX;j++) {
		fmid=func(xmid=rtb+(dx *= 0.5));
		if (fmid <= 0.0) rtb=xmid;
		if (abs(dx) < xacc || fmid == 0.0) return rtb;
	}
	throw("Too many bisections in rtbis");
}
template <class T>
Doub rtflsp(T &func, const Doub x1, const Doub x2, const Doub xacc) {
	const Int MAXIT=30;
	Doub xl,xh,del;
	Doub fl=func(x1);
	Doub fh=func(x2);
	if (fl*fh > 0.0) throw("Root must be bracketed in rtflsp");
	if (fl < 0.0) {
		xl=x1;
		xh=x2;
	} else {
		xl=x2;
		xh=x1;
		SWAP(fl,fh);
	}
	Doub dx=xh-xl;
	for (Int j=0;j<MAXIT;j++) {
		Doub rtf=xl+dx*fl/(fl-fh);
		Doub f=func(rtf);
		if (f < 0.0) {
			del=xl-rtf;
			xl=rtf;
			fl=f;
		} else {
			del=xh-rtf;
			xh=rtf;
			fh=f;
		}
		dx=xh-xl;
		if (abs(del) < xacc || f == 0.0) return rtf;
	}
	throw("Maximum number of iterations exceeded in rtflsp");
}
template <class T>
Doub rtsec(T &func, const Doub x1, const Doub x2, const Doub xacc) {
	const Int MAXIT=30;
	Doub xl,rts;
	Doub fl=func(x1);
	Doub f=func(x2);
	if (abs(fl) < abs(f)) {
		rts=x1;
		xl=x2;
		SWAP(fl,f);
	} else {
		xl=x1;
		rts=x2;
	}
	for (Int j=0;j<MAXIT;j++) {
		Doub dx=(xl-rts)*f/(f-fl);
		xl=rts;
		fl=f;
		rts += dx;
		f=func(rts);
		if (abs(dx) < xacc || f == 0.0) return rts;
	}
	throw("Maximum number of iterations exceeded in rtsec");
}
template <class T>
Doub zriddr(T &func, const Doub x1, const Doub x2, const Doub xacc) {
	const Int MAXIT=60;
	Doub fl=func(x1);
	Doub fh=func(x2);
	if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0)) {
		Doub xl=x1;
		Doub xh=x2;
		Doub ans=-9.99e99;
		for (Int j=0;j<MAXIT;j++) {
			Doub xm=0.5*(xl+xh);
			Doub fm=func(xm);
			Doub s=sqrt(fm*fm-fl*fh);
			if (s == 0.0) return ans;
			Doub xnew=xm+(xm-xl)*((fl >= fh ? 1.0 : -1.0)*fm/s);
			if (abs(xnew-ans) <= xacc) return ans;
			ans=xnew;
			Doub fnew=func(ans);
			if (fnew == 0.0) return ans;
			if (SIGN(fm,fnew) != fm) {
				xl=xm;
				fl=fm;
				xh=ans;
				fh=fnew;
			} else if (SIGN(fl,fnew) != fl) {
				xh=ans;
				fh=fnew;
			} else if (SIGN(fh,fnew) != fh) {
				xl=ans;
				fl=fnew;
			} else throw("never get here.");
			if (abs(xh-xl) <= xacc) return ans;
		}
		throw("zriddr exceed maximum iterations");
	}
	else {
		if (fl == 0.0) return x1;
		if (fh == 0.0) return x2;
		throw("root must be bracketed in zriddr.");
	}
}
template <class T>
Doub zbrent(T &func, const Doub x1, const Doub x2, const Doub tol)
{
	const Int ITMAX=100;
	const Doub EPS=numeric_limits<Doub>::epsilon();
	Doub a=x1,b=x2,c=x2,d,e,fa=func(a),fb=func(b),fc,p,q,r,s,tol1,xm;
	if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
		throw("Root must be bracketed in zbrent");
	fc=fb;
	for (Int iter=0;iter<ITMAX;iter++) {
		if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
			c=a;
			fc=fa;
			e=d=b-a;
		}
		if (abs(fc) < abs(fb)) {
			a=b;
			b=c;
			c=a;
			fa=fb;
			fb=fc;
			fc=fa;
		}
		tol1=2.0*EPS*abs(b)+0.5*tol;
		xm=0.5*(c-b);
		if (abs(xm) <= tol1 || fb == 0.0) return b;
		if (abs(e) >= tol1 && abs(fa) > abs(fb)) {
			s=fb/fa;
			if (a == c) {
				p=2.0*xm*s;
				q=1.0-s;
			} else {
				q=fa/fc;
				r=fb/fc;
				p=s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0));
				q=(q-1.0)*(r-1.0)*(s-1.0);
			}
			if (p > 0.0) q = -q;
			p=abs(p);
			Doub min1=3.0*xm*q-abs(tol1*q);
			Doub min2=abs(e*q);
			if (2.0*p < (min1 < min2 ? min1 : min2)) {
				e=d;
				d=p/q;
			} else {
				d=xm;
				e=d;
			}
		} else {
			d=xm;
			e=d;
		}
		a=b;
		fa=fb;
		if (abs(d) > tol1)
			b += d;
		else
			b += SIGN(tol1,xm);
			fb=func(b);
	}
	throw("Maximum number of iterations exceeded in zbrent");
}
template <class T>
Doub rtnewt(T &funcd, const Doub x1, const Doub x2, const Doub xacc) {
	const Int JMAX=20;
	Doub rtn=0.5*(x1+x2);
	for (Int j=0;j<JMAX;j++) {
		Doub f=funcd(rtn);
		Doub df=funcd.df(rtn);
		Doub dx=f/df;
		rtn -= dx;
		if ((x1-rtn)*(rtn-x2) < 0.0)
			throw("Jumped out of brackets in rtnewt");
		if (abs(dx) < xacc) return rtn;
	}
	throw("Maximum number of iterations exceeded in rtnewt");
}
template <class T>
Doub rtsafe(T &funcd, const Doub x1, const Doub x2, const Doub xacc) {
	const Int MAXIT=100;
	Doub xh,xl;
	Doub fl=funcd(x1);
	Doub fh=funcd(x2);
	if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0))
		throw("Root must be bracketed in rtsafe");
	if (fl == 0.0) return x1;
	if (fh == 0.0) return x2;
	if (fl < 0.0) {
		xl=x1;
		xh=x2;
	} else {
		xh=x1;
		xl=x2;
	}
	Doub rts=0.5*(x1+x2);
	Doub dxold=abs(x2-x1);
	Doub dx=dxold;
	Doub f=funcd(rts);
	Doub df=funcd.df(rts);
	for (Int j=0;j<MAXIT;j++) {
		if ((((rts-xh)*df-f)*((rts-xl)*df-f) > 0.0)
			|| (abs(2.0*f) > abs(dxold*df))) {
			dxold=dx;
			dx=0.5*(xh-xl);
			rts=xl+dx;
			if (xl == rts) return rts;
		} else {
			dxold=dx;
			dx=f/df;
			Doub temp=rts;
			rts -= dx;
			if (temp == rts) return rts;
		}
		if (abs(dx) < xacc) return rts;
		Doub f=funcd(rts);
		Doub df=funcd.df(rts);
		if (f < 0.0)
			xl=rts;
		else
			xh=rts;
	}
	throw("Maximum number of iterations exceeded in rtsafe");
}
