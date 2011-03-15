void gauleg(const Doub x1, const Doub x2, VecDoub_O &x, VecDoub_O &w)
{
	const Doub EPS=1.0e-14;
	Doub z1,z,xm,xl,pp,p3,p2,p1;
	Int n=x.size();
	Int m=(n+1)/2;
	xm=0.5*(x2+x1);
	xl=0.5*(x2-x1);
	for (Int i=0;i<m;i++) {
		z=cos(3.141592654*(i+0.75)/(n+0.5));
		do {
			p1=1.0;
			p2=0.0;
			for (Int j=0;j<n;j++) {
				p3=p2;
				p2=p1;
				p1=((2.0*j+1.0)*z*p2-j*p3)/(j+1);
			}
			pp=n*(z*p1-p2)/(z*z-1.0);
			z1=z;
			z=z1-p1/pp;
		} while (abs(z-z1) > EPS);
		x[i]=xm-xl*z;
		x[n-1-i]=xm+xl*z;
		w[i]=2.0*xl/((1.0-z*z)*pp*pp);
		w[n-1-i]=w[i];
	}
}
void gaulag(VecDoub_O &x, VecDoub_O &w, const Doub alf)
{
	const Int MAXIT=10;
	const Doub EPS=1.0e-14;
	Int i,its,j;
	Doub ai,p1,p2,p3,pp,z,z1;
	Int n=x.size();
	for (i=0;i<n;i++) {
		if (i == 0) {
			z=(1.0+alf)*(3.0+0.92*alf)/(1.0+2.4*n+1.8*alf);
		} else if (i == 1) {
			z += (15.0+6.25*alf)/(1.0+0.9*alf+2.5*n);
		} else {
			ai=i-1;
			z += ((1.0+2.55*ai)/(1.9*ai)+1.26*ai*alf/
				(1.0+3.5*ai))*(z-x[i-2])/(1.0+0.3*alf);
		}
		for (its=0;its<MAXIT;its++) {
			p1=1.0;
			p2=0.0;
			for (j=0;j<n;j++) {
				p3=p2;
				p2=p1;
				p1=((2*j+1+alf-z)*p2-(j+alf)*p3)/(j+1);
			}
			pp=(n*p1-(n+alf)*p2)/z;
			z1=z;
			z=z1-p1/pp;
			if (abs(z-z1) <= EPS) break;
		}
		if (its >= MAXIT) throw("too many iterations in gaulag");
		x[i]=z;
		w[i] = -exp(gammln(alf+n)-gammln(Doub(n)))/(pp*n*p2);
	}
}
void gauher(VecDoub_O &x, VecDoub_O &w)
{
	const Doub EPS=1.0e-14,PIM4=0.7511255444649425;
	const Int MAXIT=10;
	Int i,its,j,m;
	Doub p1,p2,p3,pp,z,z1;
	Int n=x.size();
	m=(n+1)/2;
	for (i=0;i<m;i++) {
		if (i == 0) {
			z=sqrt(Doub(2*n+1))-1.85575*pow(Doub(2*n+1),-0.16667);
		} else if (i == 1) {
			z -= 1.14*pow(Doub(n),0.426)/z;
		} else if (i == 2) {
			z=1.86*z-0.86*x[0];
		} else if (i == 3) {
			z=1.91*z-0.91*x[1];
		} else {
			z=2.0*z-x[i-2];
		}
		for (its=0;its<MAXIT;its++) {
			p1=PIM4;
			p2=0.0;
			for (j=0;j<n;j++) {
				p3=p2;
				p2=p1;
				p1=z*sqrt(2.0/(j+1))*p2-sqrt(Doub(j)/(j+1))*p3;
			}
			pp=sqrt(Doub(2*n))*p2;
			z1=z;
			z=z1-p1/pp;
			if (abs(z-z1) <= EPS) break;
		}
		if (its >= MAXIT) throw("too many iterations in gauher");
		x[i]=z;
		x[n-1-i] = -z;
		w[i]=2.0/(pp*pp);
		w[n-1-i]=w[i];
	}
}
void gaujac(VecDoub_O &x, VecDoub_O &w, const Doub alf, const Doub bet)
{
	const Int MAXIT=10;
	const Doub EPS=1.0e-14;
	Int i,its,j;
	Doub alfbet,an,bn,r1,r2,r3;
	Doub a,b,c,p1,p2,p3,pp,temp,z,z1;
	Int n=x.size();
	for (i=0;i<n;i++) {
		if (i == 0) {
			an=alf/n;
			bn=bet/n;
			r1=(1.0+alf)*(2.78/(4.0+n*n)+0.768*an/n);
			r2=1.0+1.48*an+0.96*bn+0.452*an*an+0.83*an*bn;
			z=1.0-r1/r2;
		} else if (i == 1) {
			r1=(4.1+alf)/((1.0+alf)*(1.0+0.156*alf));
			r2=1.0+0.06*(n-8.0)*(1.0+0.12*alf)/n;
			r3=1.0+0.012*bet*(1.0+0.25*abs(alf))/n;
			z -= (1.0-z)*r1*r2*r3;
		} else if (i == 2) {
			r1=(1.67+0.28*alf)/(1.0+0.37*alf);
			r2=1.0+0.22*(n-8.0)/n;
			r3=1.0+8.0*bet/((6.28+bet)*n*n);
			z -= (x[0]-z)*r1*r2*r3;
		} else if (i == n-2) {
			r1=(1.0+0.235*bet)/(0.766+0.119*bet);
			r2=1.0/(1.0+0.639*(n-4.0)/(1.0+0.71*(n-4.0)));
			r3=1.0/(1.0+20.0*alf/((7.5+alf)*n*n));
			z += (z-x[n-4])*r1*r2*r3;
		} else if (i == n-1) {
			r1=(1.0+0.37*bet)/(1.67+0.28*bet);
			r2=1.0/(1.0+0.22*(n-8.0)/n);
			r3=1.0/(1.0+8.0*alf/((6.28+alf)*n*n));
			z += (z-x[n-3])*r1*r2*r3;
		} else {
			z=3.0*x[i-1]-3.0*x[i-2]+x[i-3];
		}
		alfbet=alf+bet;
		for (its=1;its<=MAXIT;its++) {
			temp=2.0+alfbet;
			p1=(alf-bet+temp*z)/2.0;
			p2=1.0;
			for (j=2;j<=n;j++) {
				p3=p2;
				p2=p1;
				temp=2*j+alfbet;
				a=2*j*(j+alfbet)*(temp-2.0);
				b=(temp-1.0)*(alf*alf-bet*bet+temp*(temp-2.0)*z);
				c=2.0*(j-1+alf)*(j-1+bet)*temp;
				p1=(b*p2-c*p3)/a;
			}
			pp=(n*(alf-bet-temp*z)*p1+2.0*(n+alf)*(n+bet)*p2)/(temp*(1.0-z*z));
			z1=z;
			z=z1-p1/pp;
			if (abs(z-z1) <= EPS) break;
		}
		if (its > MAXIT) throw("too many iterations in gaujac");
		x[i]=z;
		w[i]=exp(gammln(alf+n)+gammln(bet+n)-gammln(n+1.0)-
			gammln(n+alfbet+1.0))*temp*pow(2.0,alfbet)/(pp*p2);
	}
}
