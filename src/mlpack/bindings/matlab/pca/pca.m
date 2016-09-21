function result = pca(dataPoints, varargin)
%Principal Components Analysis
%
%  This program performs principal components analysis on the given dataset.  It
%  will transform the data onto its principal components, optionally performing
%  dimensionality reduction by ignoring the principal components with the
%  smallest eigenvalues.
%
%Parameters:
% dataPoints        - (required) Matrix to perform PCA on.
% newDimensionality - (optional) Desired dimensionality of output dataset.  If 0,
%                                no dimensionality reduction is performed.
%                                Default value 0.
% scale             - (optional) If set, the data will be scaled before running
%                                PCA, such that the variance of each feature is
%                                1. Default value is false.

% a parser for the inputs
p = inputParser;
p.addParamValue('newDimensionality', 0, @isscalar);
p.addParamValue('scale', false, @(x) (x == true) || (x == false));

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack
result = mex_pca(dataPoints', parsed.newDimensionality, parsed.scale);
result = result';




