#pragma once

/** nmf.h
 **/

void nmf(const Matrix& V, const Matrix& Winit, const Matrix& Hinit,
	 index_t maxiter, Matrix* W_, Matrix* H_);
void prepare_for_nmf(Matrix& V);
