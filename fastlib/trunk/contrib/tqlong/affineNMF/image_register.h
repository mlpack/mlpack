#pragma once

#include "image_type.h"

void register_transform(const ImageType& I1, const ImageType& I2,
			const Transformation& tInit, Transformation& tOut);

void register_transform(const ImageType& I1, const ImageType& I2,
			Transformation& tOut);

void register_weights(const ImageType& X, const ArrayList<ImageType>& I,
		      const Vector& wInit, Vector& wOut);

void register_weights(const ImageType& X, const ArrayList<ImageType>& I,
		      Vector& wOut);

void register_basis(const ArrayList<ImageType>& X, 
		    const ArrayList<Transformation>& T, const ArrayList<Vector>& W,
		    const ArrayList<ImageType>& BInit, ArrayList<ImageType>& BOut);

void register_basis(const ArrayList<ImageType>& X, 
		    const ArrayList<Transformation>& T, const ArrayList<Vector>& W,
		    ArrayList<ImageType>& BOut);

void register_all(const ArrayList<ImageType>& X, ArrayList<Transformation>& T, 
		  ArrayList<Vector>& W, ArrayList<ImageType>& B);

void CalculateRecovery(const ArrayList<Transformation>& T, 
		       const ArrayList<Vector>& W, 
		       const ArrayList<ImageType>& B, 
		       ArrayList<ImageType>& XRecover);
