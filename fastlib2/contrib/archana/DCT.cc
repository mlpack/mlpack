#include "stdio.h"
#include "math.h"
#include "fastlib/fastlib.h"
#include "dataset.h"

#define pi 180
int main()
{
	Matrix input_values,cosine_coeffreading CSV files in Cs;
	data::Load ("Randomval1.csv",&input_values);        /* Load the csv file generated using Matlab */
	cosine_coeffs.Init(1,input_values.n_cols()); /*Initialize the matrix - the matrix generated is 256 point dataset*/
	cosine_coeffs.SetZero();
	/*Computing DCT -2 where N=256*/
	for(index_t k=0;k<256;++i)
	{
		for(index_t n=0;n<256;++n)
		{
			cosine_coeffs.set(0,k)+=input_values(0,n)*cos((pi*(n+0.5)*k )/256);
		}
	}
	FILE *f1;
  	f1=fopen("Results.txt","w");
	Save(*f1,cosine_coeffs);
	fclose(f1);
	return 0;
}
	 
	
		
