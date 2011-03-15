Doub invxlogx(Doub y) {
	const Doub ooe = 0.367879441171442322;
	Doub t,u,to=0.;
	if (y >= 0. || y <= -ooe) throw("no such inverse value");
 	if (y < -0.2) u = log(ooe-sqrt(2*ooe*(y+ooe)));
	else u = -10.;
	do {
		u += (t=(log(y/u)-u)*(u/(1.+u)));
		if (t < 1.e-8 && abs(t+to)<0.01*abs(t)) break;
		to = t;
	} while (abs(t/u) > 1.e-15);	
	return exp(u);
}
struct KSdist {
	Doub pks(Doub z) {
		if (z < 0.) throw("bad z in KSdist");
		if (z == 0.) return 0.;
		if (z < 1.18) {
			Doub y = exp(-1.23370055013616983/SQR(z));
			return 2.25675833419102515*sqrt(-log(y))
				*(y + pow(y,9) + pow(y,25) + pow(y,49));
		} else {
			Doub x = exp(-2.*SQR(z));
			return 1. - 2.*(x - pow(x,4) + pow(x,9));
		}
	}
	Doub qks(Doub z) {
		if (z < 0.) throw("bad z in KSdist");
		if (z == 0.) return 1.;
		if (z < 1.18) return 1.-pks(z);
		Doub x = exp(-2.*SQR(z));
		return 2.*(x - pow(x,4) + pow(x,9));
	}
	Doub invqks(Doub q) {
		Doub y,logy,yp,x,xp,f,ff,u,t;
		if (q <= 0. || q > 1.) throw("bad q in KSdist");
		if (q == 1.) return 0.;
		if (q > 0.3) {
			f = -0.392699081698724155*SQR(1.-q);
			y = invxlogx(f);
			do {
				yp = y;
				logy = log(y);
				ff = f/SQR(1.+ pow(y,4)+ pow(y,12));
				u = (y*logy-ff)/(1.+logy);
				y = y - (t=u/MAX(0.5,1.-0.5*u/(y*(1.+logy))));
			} while (abs(t/y)>1.e-15);
			return 1.57079632679489662/sqrt(-log(y));
		} else {
			x = 0.03;
			do {
				xp = x;
				x = 0.5*q+pow(x,4)-pow(x,9);
				if (x > 0.06) x += pow(x,16)-pow(x,25);
			} while (abs((xp-x)/x)>1.e-15);
			return sqrt(-0.5*log(x));
		}
	}
	Doub invpks(Doub p) {return invqks(1.-p);}
};
