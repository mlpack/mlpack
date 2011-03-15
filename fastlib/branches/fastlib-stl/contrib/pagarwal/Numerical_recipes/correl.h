void correl(VecDoub_I &data1, VecDoub_I &data2, VecDoub_O &ans) {
	Int no2,i,n=data1.size();
	Doub tmp;
	VecDoub temp(n);
	for (i=0;i<n;i++) {
		ans[i]=data1[i];
		temp[i]=data2[i];
	}
	realft(ans,1);
	realft(temp,1);
	no2=n>>1;
	for (i=2;i<n;i+=2) {
		tmp=ans[i];
		ans[i]=(ans[i]*temp[i]+ans[i+1]*temp[i+1])/no2;
		ans[i+1]=(ans[i+1]*temp[i]-tmp*temp[i+1])/no2;
	}
	ans[0]=ans[0]*temp[0]/no2;
	ans[1]=ans[1]*temp[1]/no2;
	realft(ans,-1);
}
