extern "C" {
	#include "ldl.h"
	#include "amd.h"
}

struct NRldl {
	Doub Info [AMD_INFO];
	Int lnz,n,nz;
	VecInt PP,PPinv,PPattern,LLnz,LLp,PParent,FFlag,*LLi;
	VecDoub YY,DD,*LLx;
	Doub *Ax, *Lx, *B, *D, *X, *Y;
	Int *Ai, *Ap, *Li, *Lp, *P, *Pinv, *Flag,*Pattern, *Lnz, *Parent;
	NRldl(NRsparseMat &adat);
	void order();
	void factorize();
	void solve(VecDoub_O &y,VecDoub &rhs);
	~NRldl();
};
Doub dotprod(VecDoub_I &x, VecDoub_I &y)
{
	Doub sum=0.0;
	for (Int i=0;i<x.size();i++)
		sum += x[i]*y[i];
	return sum;
}

Int intpt(const NRsparseMat &a, VecDoub_I &b, VecDoub_I &c, VecDoub_O &x)
{
	const Int MAXITS=200;
	const Doub EPS=1.0e-6;
	const Doub SIGMA=0.9;
	const Doub DELTA=0.02;
	const Doub BIG=numeric_limits<Doub>::max();
	Int i,j,iter,status;
	Int m=a.nrows;
	Int n=a.ncols;
	VecDoub y(m),z(n),ax(m),aty(n),rp(m),rd(n),d(n),dx(n),dy(m),dz(n),
		rhs(m),tempm(m),tempn(n);
	NRsparseMat at=a.transpose();
	ADAT adat(a,at);
	NRldl solver(adat.ref());
	solver.order();
	Doub rpfact=1.0+sqrt(dotprod(b,b));
	Doub rdfact=1.0+sqrt(dotprod(c,c));
	for (j=0;j<n;j++) {
		x[j]=1000.0;
		z[j]=1000.0;
	}
	for (i=0;i<m;i++) {
		y[i]=1000.0;
	}
	Doub normrp_old=BIG;
	Doub normrd_old=BIG;
	cout << setw(4) << "iter" << setw(12) << "Primal obj." << setw(9) <<
		"||r_p||" << setw(13) << "Dual obj." << setw(11) << "||r_d||" <<
		setw(13) << "duality gap" << setw(16) << "normalized gap" << endl;
	cout << scientific << setprecision(4);
	for (iter=0;iter<MAXITS;iter++) {
		ax=a.ax(x);
		for (i=0;i<m;i++)
			rp[i]=ax[i]-b[i];
		Doub normrp=sqrt(dotprod(rp,rp))/rpfact;
		aty=at.ax(y);
		for (j=0;j<n;j++)
			rd[j]=aty[j]+z[j]-c[j];
		Doub normrd=sqrt(dotprod(rd,rd))/rdfact;
		Doub gamma=dotprod(x,z);
		Doub mu=DELTA*gamma/n;
		Doub primal_obj=dotprod(c,x);
		Doub dual_obj=dotprod(b,y);
		Doub gamma_norm=gamma/(1.0+abs(primal_obj));
	 	cout << setw(3) << iter << setw(12) << primal_obj << setw(12) <<
			normrp << setw(12) << dual_obj << setw(12) << normrd << setw(12)
			<< gamma << setw(12) << gamma_norm<<endl;
		if (normrp < EPS && normrd < EPS && gamma_norm < EPS)
			return status=0;
		if (normrp > 1000*normrp_old && normrp > EPS)
			return status=1;
		if (normrd > 1000*normrd_old && normrd > EPS)
			return status=2;
		for (j=0;j<n;j++)
			d[j]=x[j]/z[j];
		adat.updateD(d);
		solver.factorize();
		for (j=0;j<n;j++)
			tempn[j]=x[j]-mu/z[j]-d[j]*rd[j];
		tempm=a.ax(tempn);
		for (i=0;i<m;i++)
			rhs[i]=-rp[i]+tempm[i];
		solver.solve(dy,rhs);
		tempn=at.ax(dy);
		for (j=0;j<n;j++)
			dz[j]=-tempn[j]-rd[j];
		for (j=0;j<n;j++)
			dx[j]=-d[j]*dz[j]+mu/z[j]-x[j];
		Doub alpha_p=1.0;
		for (j=0;j<n;j++)
			if (x[j]+alpha_p*dx[j] < 0.0)
				alpha_p=-x[j]/dx[j];
		Doub alpha_d=1.0;
		for (j=0;j<n;j++)
			if (z[j]+alpha_d*dz[j] < 0.0)
				alpha_d=-z[j]/dz[j];
		alpha_p = MIN(alpha_p*SIGMA,1.0);
		alpha_d = MIN(alpha_d*SIGMA,1.0);
		for (j=0;j<n;j++) {
			x[j]+=alpha_p*dx[j];
			z[j]+=alpha_d*dz[j];
		}
		for (i=0;i<m;i++)
			y[i]+=alpha_d*dy[i];
		normrp_old=normrp;
		normrd_old=normrd;
	}
	return status=3;
}
NRldl::NRldl(NRsparseMat &adat) : n(adat.ncols), nz(adat.nvals),
	Ap(&adat.col_ptr[0]), Ai(&adat.row_ind[0]), Ax(&adat.val[0]),
	PP(n),PPinv(n),PPattern(n),LLnz(n),LLp(n+1),PParent(n),FFlag(n),
	YY(n),DD(n),Y(&YY[0]),D(&DD[0]),P(&PP[0]),Pinv(&PPinv[0]),
	Pattern(&PPattern[0]),Lnz(&LLnz[0]),Lp(&LLp[0]),Parent(&PParent[0]),
	Flag(&FFlag[0]) {}

void NRldl::order() {
	if (amd_order (n, Ap, Ai, P, (Doub *) NULL, Info) != AMD_OK)
		throw("call to AMD failed");
	amd_control ((Doub *) NULL);
	//amd_info (Info);
	ldl_symbolic (n, Ap, Ai, Lp, Parent, Lnz, Flag, P, Pinv);
	lnz = Lp [n];
	/* find # of nonzeros in L, and flop count for ldl_numeric */
	Doub flops = 0 ;
	for (Int j = 0 ; j < n ; j++)
		flops += ((Doub) Lnz [j]) * (Lnz [j] + 2) ;
	cout << "Nz in L: " << lnz << " Flop count: " << flops << endl;
	/* -------------------------------------------------------------- */
	/* allocate remainder of L, of size lnz */
	/* -------------------------------------------------------------- */
	LLi=new VecInt(lnz);
	LLx=new VecDoub(lnz);
	Li=&(*LLi)[0];
	Lx=&(*LLx)[0];
}

void NRldl::factorize() {
	/* -------------------------------------------------------------- */
	/* numeric factorization to get Li, Lx, and D */
	/* -------------------------------------------------------------- */
	Int dd = ldl_numeric (n, Ap, Ai, Ax, Lp, Parent, Lnz, Li, Lx, D,
		Y, Flag, Pattern, P, Pinv) ;
	if (dd != n)
		throw("Factorization failed since diagonal is zero.");
}

void NRldl::solve(VecDoub_O &y,VecDoub &rhs) {
	B=&rhs[0];
	X=&y[0];
	/* solve Ax=b */
	/* the factorization is LDL' = PAP' */
	ldl_perm (n, Y, B, P) ;             /* y = Pb */
	ldl_lsolve (n, Y, Lp, Li, Lx) ;     /* y = L\y */
	ldl_dsolve (n, Y, D) ;              /* y = D\y */
	ldl_ltsolve (n, Y, Lp, Li, Lx) ;    /* y = L'\y */
	ldl_permt (n, X, Y, P) ;            /* x = P'y */
}

NRldl::~NRldl() {
	delete LLx;
	delete LLi;
}
