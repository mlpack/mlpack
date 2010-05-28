#pragma once

/** imregister.h
 **
 **/

/** find a projective mapping M such that ||M(P)-Q||^2 --> min
 **/
int projective_register(const Matrix& P, const Matrix& Q, Vector* m);

/** compute gaussian smoothed intensity
 ** gaussian kernel width: sigma
 ** 2 versions, with/without outputing image gradient
 **/
double smooth_intensity(const Matrix& I, double x, double y, double sigma);

double smooth_intensity(const Matrix& I, double x, double y, double sigma,
			double* Ir, double* Ic);

void projective_map(const Vector& m, double r, double c, 
		    double* r_map, double* c_map, double* d_map);
