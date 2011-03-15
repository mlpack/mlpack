Doub func(const Doub x)
{
	if (x == 0.0)
		return 0.0;
	else {
		Bessel bess;
		return x*bess.jnu(0.0,x)/(1.0+x*x);
	}
}

Int main_levex(void)
{
	const Doub PI=3.141592653589793;
	Int nterm=12;
	Doub beta=1.0,a=0.0,b=0.0,sum=0.0;
	Levin series(100,0.0);
	cout << setw(5) << "N" << setw(19) << "Sum (direct)" << setw(21)
		<< "Sum (Levin)" << endl;
	for (Int n=0; n<=nterm; n++) {
		b+=PI;
		Doub s=qromb(func,a,b,1.e-8);
		a=b;
		sum+=s;
		Doub omega=(beta+n)*s;
		Doub ans=series.next(sum,omega,beta);
		cout << setw(5) << n << fixed << setprecision(14) << setw(21)
			<< sum << setw(21) << ans << endl;
	}
	return 0;
}
