struct Unsymmeig {
	Int n;
	MatDoub a,zz;
	VecComplex wri;
	VecDoub scale;
	VecInt perm;
	Bool yesvecs,hessen;

	Unsymmeig(MatDoub_I &aa, Bool yesvec=true, Bool hessenb=false) :
		n(aa.nrows()), a(aa), zz(n,n,0.0), wri(n), scale(n,1.0), perm(n),
		yesvecs(yesvec), hessen(hessenb)
	{
		balance();
		if (!hessen) elmhes();
		if (yesvecs) {
			for (Int i=0;i<n;i++)
				zz[i][i]=1.0;
			if (!hessen) eltran();
			hqr2();
			balbak();
			sortvecs();
		} else {
			hqr();
			sort();
		}
	}
	void balance();
	void elmhes();
	void eltran();
	void hqr();
	void hqr2();
	void balbak();
	void sort();
	void sortvecs();
};
void Unsymmeig::balance()
{
	const Doub RADIX = numeric_limits<Doub>::radix;
	Bool done=false;
	Doub sqrdx=RADIX*RADIX;
	while (!done) {
		done=true;
		for (Int i=0;i<n;i++) {
			Doub r=0.0,c=0.0;
			for (Int j=0;j<n;j++)
				if (j != i) {
					c += abs(a[j][i]);
					r += abs(a[i][j]);
				}
			if (c != 0.0 && r != 0.0) {
				Doub g=r/RADIX;
				Doub f=1.0;
				Doub s=c+r;
				while (c<g) {
					f *= RADIX;
					c *= sqrdx;
				}
				g=r*RADIX;
				while (c>g) {
					f /= RADIX;
					c /= sqrdx;
				}
				if ((c+r)/f < 0.95*s) {
					done=false;
					g=1.0/f;
					scale[i] *= f;
					for (Int j=0;j<n;j++) a[i][j] *= g;
					for (Int j=0;j<n;j++) a[j][i] *= f;
				}
			}
		}
	}
}
void Unsymmeig::balbak()
{
	for (Int i=0;i<n;i++)
		for (Int j=0;j<n;j++)
			zz[i][j] *= scale[i];
}
void Unsymmeig::elmhes()
{
	for (Int m=1;m<n-1;m++) {
		Doub x=0.0;
		Int i=m;
		for (Int j=m;j<n;j++) {
			if (abs(a[j][m-1]) > abs(x)) {
				x=a[j][m-1];
				i=j;
			}
		}
		perm[m]=i;
		if (i != m) {
			for (Int j=m-1;j<n;j++) SWAP(a[i][j],a[m][j]);
			for (Int j=0;j<n;j++) SWAP(a[j][i],a[j][m]);
		}
		if (x != 0.0) {
			for (i=m+1;i<n;i++) {
				Doub y=a[i][m-1];
				if (y != 0.0) {
					y /= x;
					a[i][m-1]=y;
					for (Int j=m;j<n;j++) a[i][j] -= y*a[m][j];
					for (Int j=0;j<n;j++) a[j][m] += y*a[j][i];
				}
			}
		}
	}
}
void Unsymmeig::eltran()
{
	for (Int mp=n-2;mp>0;mp--) {
		for (Int k=mp+1;k<n;k++)
			zz[k][mp]=a[k][mp-1];
		Int i=perm[mp];
		if (i != mp) {
			for (Int j=mp;j<n;j++) {
				zz[mp][j]=zz[i][j];
				zz[i][j]=0.0;
			}
			zz[i][mp]=1.0;
		}
	}
}
void Unsymmeig::hqr()
{
	Int nn,m,l,k,j,its,i,mmin;
	Doub z,y,x,w,v,u,t,s,r,q,p,anorm=0.0;

	const Doub EPS=numeric_limits<Doub>::epsilon();
	for (i=0;i<n;i++)
		for (j=MAX(i-1,0);j<n;j++)
			anorm += abs(a[i][j]);
	nn=n-1;
	t=0.0;
	while (nn >= 0) {
		its=0;
		do {
			for (l=nn;l>0;l--) {
				s=abs(a[l-1][l-1])+abs(a[l][l]);
				if (s == 0.0) s=anorm;
				if (abs(a[l][l-1]) <= EPS*s) {
					a[l][l-1] = 0.0;
					break;
				}
			}
			x=a[nn][nn];
			if (l == nn) {
				wri[nn--]=x+t;
			} else {
				y=a[nn-1][nn-1];
				w=a[nn][nn-1]*a[nn-1][nn];
				if (l == nn-1) {
					p=0.5*(y-x);
					q=p*p+w;
					z=sqrt(abs(q));
					x += t;
					if (q >= 0.0) {
						z=p+SIGN(z,p);
						wri[nn-1]=wri[nn]=x+z;
						if (z != 0.0) wri[nn]=x-w/z;
					} else {
						wri[nn]=Complex(x+p,-z);
						wri[nn-1]=conj(wri[nn]);
					}
					nn -= 2;
				} else {
					if (its == 30) throw("Too many iterations in hqr");
					if (its == 10 || its == 20) {
						t += x;
						for (i=0;i<nn+1;i++) a[i][i] -= x;
						s=abs(a[nn][nn-1])+abs(a[nn-1][nn-2]);
						y=x=0.75*s;
						w = -0.4375*s*s;
					}
					++its;
					for (m=nn-2;m>=l;m--) {
						z=a[m][m];
						r=x-z;
						s=y-z;
						p=(r*s-w)/a[m+1][m]+a[m][m+1];
						q=a[m+1][m+1]-z-r-s;
						r=a[m+2][m+1];
						s=abs(p)+abs(q)+abs(r);
						p /= s;
						q /= s;
						r /= s;
						if (m == l) break;
						u=abs(a[m][m-1])*(abs(q)+abs(r));
						v=abs(p)*(abs(a[m-1][m-1])+abs(z)+abs(a[m+1][m+1]));
						if (u <= EPS*v) break;
					}
					for (i=m;i<nn-1;i++) {
						a[i+2][i]=0.0;
						if (i != m) a[i+2][i-1]=0.0;
					}
					for (k=m;k<nn;k++) {
						if (k != m) {
							p=a[k][k-1];
							q=a[k+1][k-1];
							r=0.0;
							if (k+1 != nn) r=a[k+2][k-1];
							if ((x=abs(p)+abs(q)+abs(r)) != 0.0) {
								p /= x;
								q /= x;
								r /= x;
							}
						}
						if ((s=SIGN(sqrt(p*p+q*q+r*r),p)) != 0.0) {
							if (k == m) {
								if (l != m)
								a[k][k-1] = -a[k][k-1];
							} else
								a[k][k-1] = -s*x;
							p += s;
							x=p/s;
							y=q/s;
							z=r/s;
							q /= p;
							r /= p;
							for (j=k;j<nn+1;j++) {
								p=a[k][j]+q*a[k+1][j];
								if (k+1 != nn) {
									p += r*a[k+2][j];
									a[k+2][j] -= p*z;
								}
								a[k+1][j] -= p*y;
								a[k][j] -= p*x;
							}
							mmin = nn < k+3 ? nn : k+3;
							for (i=l;i<mmin+1;i++) {
								p=x*a[i][k]+y*a[i][k+1];
								if (k+1 != nn) {
									p += z*a[i][k+2];
									a[i][k+2] -= p*r;
								}
								a[i][k+1] -= p*q;
								a[i][k] -= p;
							}
						}
					}
				}
			}
		} while (l+1 < nn);
	}
}
void Unsymmeig::hqr2()
{
	Int nn,m,l,k,j,its,i,mmin,na;
	Doub z,y,x,w,v,u,t,s,r,q,p,anorm=0.0,ra,sa,vr,vi;

	const Doub EPS=numeric_limits<Doub>::epsilon();
	for (i=0;i<n;i++)
		for (j=MAX(i-1,0);j<n;j++)
			anorm += abs(a[i][j]);
	nn=n-1;
	t=0.0;
	while (nn >= 0) {
		its=0;
		do {
			for (l=nn;l>0;l--) {
				s=abs(a[l-1][l-1])+abs(a[l][l]);
				if (s == 0.0) s=anorm;
				if (abs(a[l][l-1]) <= EPS*s) {
					a[l][l-1] = 0.0;
					break;
				}
			}
			x=a[nn][nn];
			if (l == nn) {
				wri[nn]=a[nn][nn]=x+t;
				nn--;
			} else {
				y=a[nn-1][nn-1];
				w=a[nn][nn-1]*a[nn-1][nn];
				if (l == nn-1) {
					p=0.5*(y-x);
					q=p*p+w;
					z=sqrt(abs(q));
					x += t;
					a[nn][nn]=x;
					a[nn-1][nn-1]=y+t;
					if (q >= 0.0) {
						z=p+SIGN(z,p);
						wri[nn-1]=wri[nn]=x+z;
						if (z != 0.0) wri[nn]=x-w/z;
						x=a[nn][nn-1];
						s=abs(x)+abs(z);
						p=x/s;
						q=z/s;
						r=sqrt(p*p+q*q);
						p /= r;
						q /= r;
						for (j=nn-1;j<n;j++) {
							z=a[nn-1][j];
							a[nn-1][j]=q*z+p*a[nn][j];
							a[nn][j]=q*a[nn][j]-p*z;
						}
						for (i=0;i<=nn;i++) {
							z=a[i][nn-1];
							a[i][nn-1]=q*z+p*a[i][nn];
							a[i][nn]=q*a[i][nn]-p*z;
						}
						for (i=0;i<n;i++) {
							z=zz[i][nn-1];
							zz[i][nn-1]=q*z+p*zz[i][nn];
							zz[i][nn]=q*zz[i][nn]-p*z;
						}
					} else {
						wri[nn]=Complex(x+p,-z);
						wri[nn-1]=conj(wri[nn]);
					}
					nn -= 2;
				} else {
					if (its == 30) throw("Too many iterations in hqr");
					if (its == 10 || its == 20) {
						t += x;
						for (i=0;i<nn+1;i++) a[i][i] -= x;
						s=abs(a[nn][nn-1])+abs(a[nn-1][nn-2]);
						y=x=0.75*s;
						w = -0.4375*s*s;
					}
					++its;
					for (m=nn-2;m>=l;m--) {
						z=a[m][m];
						r=x-z;
						s=y-z;
						p=(r*s-w)/a[m+1][m]+a[m][m+1];
						q=a[m+1][m+1]-z-r-s;
						r=a[m+2][m+1];
						s=abs(p)+abs(q)+abs(r);
						p /= s;
						q /= s;
						r /= s;
						if (m == l) break;
						u=abs(a[m][m-1])*(abs(q)+abs(r));
						v=abs(p)*(abs(a[m-1][m-1])+abs(z)+abs(a[m+1][m+1]));
						if (u <= EPS*v) break;
					}
					for (i=m;i<nn-1;i++) {
						a[i+2][i]=0.0;
						if (i != m) a[i+2][i-1]=0.0;
					}
					for (k=m;k<nn;k++) {
						if (k != m) {
							p=a[k][k-1];
							q=a[k+1][k-1];
							r=0.0;
							if (k+1 != nn) r=a[k+2][k-1];
							if ((x=abs(p)+abs(q)+abs(r)) != 0.0) {
								p /= x;
								q /= x;
								r /= x;
							}
						}
						if ((s=SIGN(sqrt(p*p+q*q+r*r),p)) != 0.0) {
							if (k == m) {
								if (l != m)
								a[k][k-1] = -a[k][k-1];
							} else
								a[k][k-1] = -s*x;
							p += s;
							x=p/s;
							y=q/s;
							z=r/s;
							q /= p;
							r /= p;
							for (j=k;j<n;j++) {
								p=a[k][j]+q*a[k+1][j];
								if (k+1 != nn) {
									p += r*a[k+2][j];
									a[k+2][j] -= p*z;
								}
								a[k+1][j] -= p*y;
								a[k][j] -= p*x;
							}
							mmin = nn < k+3 ? nn : k+3;
							for (i=0;i<mmin+1;i++) {
								p=x*a[i][k]+y*a[i][k+1];
								if (k+1 != nn) {
									p += z*a[i][k+2];
									a[i][k+2] -= p*r;
								}
								a[i][k+1] -= p*q;
								a[i][k] -= p;
							}
							for (i=0; i<n; i++) {
								p=x*zz[i][k]+y*zz[i][k+1];
								if (k+1 != nn) {
									p += z*zz[i][k+2];
									zz[i][k+2] -= p*r;
								}
								zz[i][k+1] -= p*q;
								zz[i][k] -= p;
							}
						}
					}
				}
			}
		} while (l+1 < nn);
	}
	if (anorm != 0.0) {
		for (nn=n-1;nn>=0;nn--) {
			p=real(wri[nn]);
			q=imag(wri[nn]);
			na=nn-1;
			if (q == 0.0) {
				m=nn;
				a[nn][nn]=1.0;
				for (i=nn-1;i>=0;i--) {
					w=a[i][i]-p;
					r=0.0;
					for (j=m;j<=nn;j++)
						r += a[i][j]*a[j][nn];
					if (imag(wri[i]) < 0.0) {
						z=w;
						s=r;
					} else {
						m=i;
						
						if (imag(wri[i]) == 0.0) {
							t=w;
							if (t == 0.0)
								t=EPS*anorm;
							a[i][nn]=-r/t;
						} else {
							x=a[i][i+1];
							y=a[i+1][i];
							q=SQR(real(wri[i])-p)+SQR(imag(wri[i]));
							t=(x*s-z*r)/q;
							a[i][nn]=t;
							if (abs(x) > abs(z))
								a[i+1][nn]=(-r-w*t)/x;
							else
								a[i+1][nn]=(-s-y*t)/z;
						}
						t=abs(a[i][nn]);
						if (EPS*t*t > 1)
							for (j=i;j<=nn;j++)
								a[j][nn] /= t;
					}
				}
			} else if (q < 0.0) {
				m=na;
				if (abs(a[nn][na]) > abs(a[na][nn])) {
					a[na][na]=q/a[nn][na];
					a[na][nn]=-(a[nn][nn]-p)/a[nn][na];
				} else {
					Complex temp=Complex(0.0,-a[na][nn])/Complex(a[na][na]-p,q);
					a[na][na]=real(temp);
					a[na][nn]=imag(temp);
				}
				a[nn][na]=0.0;
				a[nn][nn]=1.0;
				for (i=nn-2;i>=0;i--) {
					w=a[i][i]-p;
					ra=sa=0.0;
					for (j=m;j<=nn;j++) {
						ra += a[i][j]*a[j][na];
						sa += a[i][j]*a[j][nn];
					}
					if (imag(wri[i]) < 0.0) {
						z=w;
						r=ra;
						s=sa;
					} else {
						m=i;
						if (imag(wri[i]) == 0.0) {
							Complex temp = Complex(-ra,-sa)/Complex(w,q);
							a[i][na]=real(temp);
							a[i][nn]=imag(temp);
						} else {
							x=a[i][i+1];
							y=a[i+1][i];
							vr=SQR(real(wri[i])-p)+SQR(imag(wri[i]))-q*q;
							vi=2.0*q*(real(wri[i])-p);
							if (vr == 0.0 && vi == 0.0)
								vr=EPS*anorm*(abs(w)+abs(q)+abs(x)+abs(y)+abs(z));
							Complex temp=Complex(x*r-z*ra+q*sa,x*s-z*sa-q*ra)/
								Complex(vr,vi);
							a[i][na]=real(temp);
							a[i][nn]=imag(temp);
							if (abs(x) > abs(z)+abs(q)) {
								a[i+1][na]=(-ra-w*a[i][na]+q*a[i][nn])/x;
								a[i+1][nn]=(-sa-w*a[i][nn]-q*a[i][na])/x;
							} else {
								Complex temp=Complex(-r-y*a[i][na],-s-y*a[i][nn])/
									Complex(z,q);
								a[i+1][na]=real(temp);
								a[i+1][nn]=imag(temp);
							}
						}
					}
					t=MAX(abs(a[i][na]),abs(a[i][nn]));
					if (EPS*t*t > 1)
						for (j=i;j<=nn;j++) {
							a[j][na] /= t;
							a[j][nn] /= t;
						}
				}
			}
		}
		for (j=n-1;j>=0;j--)
			for (i=0;i<n;i++) {
				z=0.0;
				for (k=0;k<=j;k++)
					z += zz[i][k]*a[k][j];
				zz[i][j]=z;
			}
	}
}
void Unsymmeig::sort()
{
	Int i;
	for (Int j=1;j<n;j++) {
		Complex x=wri[j];
		for (i=j-1;i>=0;i--) {
			if (real(wri[i]) >= real(x)) break;
			wri[i+1]=wri[i];
		}
		wri[i+1]=x;
	}
}
void Unsymmeig::sortvecs()
{
	Int i;
	VecDoub temp(n);
	for (Int j=1;j<n;j++) {
		Complex x=wri[j];
		for (Int k=0;k<n;k++)
			temp[k]=zz[k][j];
		for (i=j-1;i>=0;i--) {
			if (real(wri[i]) >= real(x)) break;
			wri[i+1]=wri[i];
			for (Int k=0;k<n;k++)
				zz[k][i+1]=zz[k][i];
		}
		wri[i+1]=x;
		for (Int k=0;k<n;k++)
			zz[k][i+1]=temp[k];
	}
}
