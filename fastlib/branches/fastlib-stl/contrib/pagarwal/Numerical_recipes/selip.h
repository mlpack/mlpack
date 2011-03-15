Doub selip(const Int k, VecDoub_I &arr) {
	const Int M=64;
	const Doub BIG=.99e99;
	Int i,j,jl,jm,ju,kk,mm,nlo,nxtmm,n=arr.size();
	Doub ahi,alo,sum;
	VecInt isel(M+2);
	VecDoub sel(M+2);
	if (k < 0 || k > n-1) throw("bad input to selip");
	kk=k;
	ahi=BIG;
	alo = -BIG;
	for (;;) {
		mm=nlo=0;
		sum=0.0;
		nxtmm=M+1;
		for (i=0;i<n;i++) {
			if (arr[i] >= alo && arr[i] <= ahi) {
				mm++;
				if (arr[i] == alo) nlo++;
				if (mm <= M) sel[mm-1]=arr[i];
				else if (mm == nxtmm) {
					nxtmm=mm+mm/M;
					sel[(i+2+mm+kk) % M]=arr[i];
				}
				sum += arr[i];
			}
		}
		if (kk < nlo) {
			return alo;
		}
		else if (mm < M+1) {
			shell(sel,mm);
			ahi = sel[kk];
			return ahi;
		}
		sel[M]=sum/mm;
		shell(sel,M+1);
		sel[M+1]=ahi;
		for (j=0;j<M+2;j++) isel[j]=0;
		for (i=0;i<n;i++) {
			if (arr[i] >= alo && arr[i] <= ahi) {
				jl=0;
				ju=M+2;
				while (ju-jl > 1) {
					jm=(ju+jl)/2;
					if (arr[i] >= sel[jm-1]) jl=jm;
					else ju=jm;
				}
				isel[ju-1]++;
			}
		}
		j=0;
		while (kk >= isel[j]) {
			alo=sel[j];
			kk -= isel[j++];
		}
		ahi=sel[j];
	}
}
