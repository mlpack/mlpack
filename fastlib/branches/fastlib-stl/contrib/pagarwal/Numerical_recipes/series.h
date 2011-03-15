struct Eulsum {
	VecDoub wksp;
	Int n,ncv;
	Bool cnvgd;
	Doub sum,eps,lastval,lasteps;

	Eulsum(Int nmax, Doub epss) : wksp(nmax), n(0), ncv(0),
		cnvgd(0), sum(0.), eps(epss), lastval(0.) {}

	Doub next(const Doub term)
	{
		Int j;
		Doub tmp,dum;
		if (n+1 > wksp.size()) throw("wksp too small in eulsum");
		if (n == 0) {
			sum=0.5*(wksp[n++]=term);
		} else {
			tmp=wksp[0];
			wksp[0]=term;
			for (j=1;j<n;j++) {
				dum=wksp[j];
				wksp[j]=0.5*(wksp[j-1]+tmp);
				tmp=dum;
			}
			wksp[n]=0.5*(wksp[n-1]+tmp);
			if (abs(wksp[n]) <= abs(wksp[n-1]))
				sum += (0.5*wksp[n++]);
			else
				sum += wksp[n];
		}
		lasteps = abs(sum-lastval);
		if (lasteps <= eps) ncv++;
		if (ncv >= 2) cnvgd = 1;
		return (lastval = sum);
	}
};
struct Epsalg {
	VecDoub e;
	Int n,ncv;
	Bool cnvgd;
	Doub eps,small,big,lastval,lasteps;
	
	Epsalg(Int nmax, Doub epss) : e(nmax), n(0), ncv(0),
	cnvgd(0), eps(epss), lastval(0.) {
		small = numeric_limits<Doub>::min()*10.0;
		big = numeric_limits<Doub>::max();
	}

	Doub next(Doub sum) {
		Doub diff,temp1,temp2,val;
		e[n]=sum;
		temp2=0.0;
		for (Int j=n; j>0; j--) {
			temp1=temp2;
			temp2=e[j-1];
			diff=e[j]-temp2;
			if (abs(diff) <= small)
				e[j-1]=big;
			else
				e[j-1]=temp1+1.0/diff;
		}
		n++;
		val = (n & 1) ? e[0] : e[1];
		if (abs(val) > 0.01*big) val = lastval;
		lasteps = abs(val-lastval);
		if (lasteps > eps) ncv = 0;
		else ncv++;
		if (ncv >= 3) cnvgd = 1;
		return (lastval = val);
	}
	
};
struct Levin {
	VecDoub numer,denom;
	Int n,ncv;
	Bool cnvgd;
	Doub small,big;
	Doub eps,lastval,lasteps;

	Levin(Int nmax, Doub epss) : numer(nmax), denom(nmax), n(0), ncv(0),
	cnvgd(0), eps(epss), lastval(0.) {
		small=numeric_limits<Doub>::min()*10.0;
		big=numeric_limits<Doub>::max();
	}

	Doub next(Doub sum, Doub omega, Doub beta=1.) {
		Int j;
		Doub fact,ratio,term,val;
		term=1.0/(beta+n);
		denom[n]=term/omega;
		numer[n]=sum*denom[n];
		if (n > 0) {
			ratio=(beta+n-1)*term;
			for (j=1;j<=n;j++) {
				fact=(n-j+beta)*term;
				numer[n-j]=numer[n-j+1]-fact*numer[n-j];
				denom[n-j]=denom[n-j+1]-fact*denom[n-j];
				term=term*ratio;
			}
		}
		n++;
		val = abs(denom[0]) < small ? lastval : numer[0]/denom[0];
		lasteps = abs(val-lastval);
		if (lasteps <= eps) ncv++;
		if (ncv >= 2) cnvgd = 1;
		return (lastval = val);
	}
};

