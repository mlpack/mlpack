struct MParith {

	void mpadd(VecUchar_O &w, VecUchar_I &u, VecUchar_I &v) {
		Int j,n=u.size(),m=v.size(),p=w.size();
		Int n_min=MIN(n,m),p_min=MIN(n_min,p-1);
		Uint ireg=0;
		for (j=p_min-1;j>=0;j--) {
			ireg=u[j]+v[j]+hibyte(ireg);
			w[j+1]=lobyte(ireg);
		}
		w[0]=hibyte(ireg);
		if (p > p_min+1)
			for (j=p_min+1;j<p;j++) w[j]=0;
	}

	void mpsub(Int &is, VecUchar_O &w, VecUchar_I &u, VecUchar_I &v) {
		Int j,n=u.size(),m=v.size(),p=w.size();
		Int n_min=MIN(n,m),p_min=MIN(n_min,p-1);
		Uint ireg=256;
		for (j=p_min-1;j>=0;j--) {
			ireg=255+u[j]-v[j]+hibyte(ireg);
			w[j]=lobyte(ireg);
		}
		is=hibyte(ireg)-1;
		if (p > p_min)
			for (j=p_min;j<p;j++) w[j]=0;
	}

	void mpsad(VecUchar_O &w, VecUchar_I &u, const Int iv) {
		Int j,n=u.size(),p=w.size();
		Uint ireg=256*iv;
		for (j=n-1;j>=0;j--) {
			ireg=u[j]+hibyte(ireg);
			if (j+1 < p) w[j+1]=lobyte(ireg);
		}
		w[0]=hibyte(ireg);
		for (j=n+1;j<p;j++) w[j]=0;
	}

	void mpsmu(VecUchar_O &w, VecUchar_I &u, const Int iv) {
		Int j,n=u.size(),p=w.size();
		Uint ireg=0;
		for (j=n-1;j>=0;j--) {
			ireg=u[j]*iv+hibyte(ireg);
			if (j < p-1) w[j+1]=lobyte(ireg);
		}
		w[0]=hibyte(ireg);
		for (j=n+1;j<p;j++) w[j]=0;
	}

	void mpsdv(VecUchar_O &w, VecUchar_I &u, const Int iv, Int &ir) {
		Int i,j,n=u.size(),p=w.size(),p_min=MIN(n,p);
		ir=0;
		for (j=0;j<p_min;j++) {
			i=256*ir+u[j];
			w[j]=Uchar(i/iv);
			ir=i % iv;
		}
		if (p > p_min)
			for (j=p_min;j<p;j++) w[j]=0;
	}

	void mpneg(VecUchar_IO &u) {
		Int j,n=u.size();
		Uint ireg=256;
		for (j=n-1;j>=0;j--) {
			ireg=255-u[j]+hibyte(ireg);
			u[j]=lobyte(ireg);
		}
	}

	void mpmov(VecUchar_O &u, VecUchar_I &v) {
		Int j,n=u.size(),m=v.size(),n_min=MIN(n,m);
		for (j=0;j<n_min;j++) u[j]=v[j];
		if (n > n_min)
			for(j=n_min;j<n-1;j++) u[j]=0;
	}

	void mplsh(VecUchar_IO &u) {
		Int j,n=u.size();
		for (j=0;j<n-1;j++) u[j]=u[j+1];
		u[n-1]=0;
	}

	Uchar lobyte(Uint x) {return (x & 0xff);}
	Uchar hibyte(Uint x) {return ((x >> 8) & 0xff);}

	void mpmul(VecUchar_O &w, VecUchar_I &u, VecUchar_I &v);
	void mpinv(VecUchar_O &u, VecUchar_I &v);
	void mpdiv(VecUchar_O &q, VecUchar_O &r, VecUchar_I &u, VecUchar_I &v);
	void mpsqrt(VecUchar_O &w, VecUchar_O &u, VecUchar_I &v);
	void mp2dfr(VecUchar_IO &a, string &s);
	string mppi(const Int np);
};
void MParith::mpmul(VecUchar_O &w, VecUchar_I &u, VecUchar_I &v) {
	const Doub RX=256.0;
	Int j,nn=1,n=u.size(),m=v.size(),p=w.size(),n_max=MAX(m,n);
	Doub cy,t;
	while (nn < n_max) nn <<= 1;
	nn <<= 1;
	VecDoub a(nn,0.0),b(nn,0.0);
	for (j=0;j<n;j++) a[j]=u[j];
	for (j=0;j<m;j++) b[j]=v[j];
	realft(a,1);
	realft(b,1);
	b[0] *= a[0];
	b[1] *= a[1];
	for (j=2;j<nn;j+=2) {
		b[j]=(t=b[j])*a[j]-b[j+1]*a[j+1];
		b[j+1]=t*a[j+1]+b[j+1]*a[j];
	}
	realft(b,-1);
	cy=0.0;
	for (j=nn-1;j>=0;j--) {
		t=b[j]/(nn >> 1)+cy+0.5;
		cy=Uint(t/RX);
		b[j]=t-cy*RX;
	}
	if (cy >= RX) throw("cannot happen in mpmul");
	for (j=0;j<p;j++) w[j]=0;
	w[0]=Uchar(cy);
	for (j=1;j<MIN(n+m,p);j++) w[j]=Uchar(b[j-1]);
}
void MParith::mpinv(VecUchar_O &u, VecUchar_I &v) {
	const Int MF=4;
	const Doub BI=1.0/256.0;
	Int i,j,n=u.size(),m=v.size(),mm=MIN(MF,m);
	Doub fu,fv=Doub(v[mm-1]);
	VecUchar s(n+m),r(2*n+m);
	for (j=mm-2;j>=0;j--) {
		fv *= BI;
		fv += v[j];
	}
	fu=1.0/fv;
	for (j=0;j<n;j++) {
		i=Int(fu);
		u[j]=Uchar(i);
		fu=256.0*(fu-i);
	}
	for (;;) {
		mpmul(s,u,v);
		mplsh(s);
		mpneg(s);
		s[0] += Uchar(2);
		mpmul(r,s,u);
		mplsh(r);
		mpmov(u,r);
		for (j=1;j<n-1;j++)
			if (s[j] != 0) break;
		if (j==n-1) return;
	}
}
void MParith::mpdiv(VecUchar_O &q, VecUchar_O &r, VecUchar_I &u, VecUchar_I &v) {
	const Int MACC=1;
	Int i,is,mm,n=u.size(),m=v.size(),p=r.size(),n_min=MIN(m,p);
	if (m > n) throw("Divisor longer than dividend in mpdiv");
	mm=m+MACC;
	VecUchar s(mm),rr(mm),ss(mm+1),qq(n-m+1),t(n);
	mpinv(s,v);
	mpmul(rr,s,u);
	mpsad(ss,rr,1);
	mplsh(ss);
	mplsh(ss);
	mpmov(qq,ss);
	mpmov(q,qq);
	mpmul(t,qq,v);
	mplsh(t);
	mpsub(is,t,u,t);
	if (is != 0) throw("MACC too small in mpdiv");
	for (i=0;i<n_min;i++) r[i]=t[i+n-m];
	if (p>m) for (i=m;i<p;i++) r[i]=0;
}
void MParith::mpsqrt(VecUchar_O &w, VecUchar_O &u, VecUchar_I &v) {
	const Int MF=3;
	const Doub BI=1.0/256.0;
	Int i,ir,j,n=u.size(),m=v.size(),mm=MIN(m,MF);
	VecUchar r(2*n),x(n+m),s(2*n+m),t(3*n+m);
	Doub fu,fv=Doub(v[mm-1]);
	for (j=mm-2;j>=0;j--) {
		fv *= BI;
		fv += v[j];
	}
	fu=1.0/sqrt(fv);
	for (j=0;j<n;j++) {
		i=Int(fu);
		u[j]=Uchar(i);
		fu=256.0*(fu-i);
	}
	for (;;) {
		mpmul(r,u,u);
		mplsh(r);
		mpmul(s,r,v);
		mplsh(s);
		mpneg(s);
		s[0] += Uchar(3);
		mpsdv(s,s,2,ir);
		for (j=1;j<n-1;j++) {
			if (s[j] != 0) {
				mpmul(t,s,u);
				mplsh(t);
				mpmov(u,t);
				break;
			}
		}
		if (j<n-1) continue;
		mpmul(x,u,v);
		mplsh(x);
		mpmov(w,x);
		return;
	}
}
void MParith::mp2dfr(VecUchar_IO &a, string &s)
{
	const Uint IAZ=48;
	char buffer[4];
	Int j,m;

	Int n=a.size();
	m=Int(2.408*n);
	sprintf(buffer,"%d",a[0]);
	s=buffer;
	s += '.';
	mplsh(a);
	for (j=0;j<m;j++) {
		mpsmu(a,a,10);
		s += a[0]+IAZ;
		mplsh(a);
	}
}
string MParith::mppi(const Int np) {
	const Uint IAOFF=48,MACC=2;
	Int ir,j,n=np+MACC;
	Uchar mm;
	string s;
	VecUchar x(n),y(n),sx(n),sxi(n),z(n),t(n),pi(n),ss(2*n),tt(2*n);
	t[0]=2;
	for (j=1;j<n;j++) t[j]=0;
	mpsqrt(x,x,t);
	mpadd(pi,t,x);
	mplsh(pi);
	mpsqrt(sx,sxi,x);
	mpmov(y,sx);
	for (;;) {
		mpadd(z,sx,sxi);
		mplsh(z);
		mpsdv(x,z,2,ir);
		mpsqrt(sx,sxi,x);
		mpmul(tt,y,sx);
		mplsh(tt);
		mpadd(tt,tt,sxi);
		mplsh(tt);
		x[0]++;
		y[0]++;
		mpinv(ss,y);
		mpmul(y,tt,ss);
		mplsh(y);
		mpmul(tt,x,ss);
		mplsh(tt);
		mpmul(ss,pi,tt);
		mplsh(ss);
		mpmov(pi,ss);
		mm=tt[0]-1;
		for (j=1;j < n-1;j++)
			if (tt[j] != mm) break;
		if (j == n-1) {
			mp2dfr(pi,s);
			s.erase(Int(2.408*np),s.length());
			return s;
		}
	}
}
