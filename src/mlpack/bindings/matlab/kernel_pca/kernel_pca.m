function result = kernel_pca(dataPoints, kernel, varargin)
%Kernel Principal Components Analysis
%
%  This program performs Kernel Principal Components Analysis (KPCA) on the
%  specified dataset with the specified kernel.  This will transform the data
%  onto the kernel principal components, and optionally reduce the dimensionality
%  by ignoring the kernel principal components with the smallest eigenvalues.
%
%  For the case where a linear kernel is used, this reduces to regular PCA.
%
%  The kernels that are supported are listed below:
%
%   * 'linear': the standard linear dot product (same as normal PCA):
%      K(x, y) = x^T y
%
%   * 'gaussian': a Gaussian kernel; requires bandwidth:
%      K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))
%
%   * 'polynomial': polynomial kernel; requires offset and degree:
%      K(x, y) = (x^T y + offset) ^ degree
%
%   * 'hyptan': hyperbolic tangent kernel; requires scale and offset:
%      K(x, y) = tanh(scale * (x^T y) + offset)
%
%   * 'laplacian': Laplacian kernel; requires bandwidth:
%      K(x, y) = exp(-(|| x - y ||) / bandwidth)
%
%   * 'cosine': cosine distance:
%      K(x, y) = 1 - (x^T y) / (|| x || * || y ||)
%
%  The parameters for each of the kernels should be specified with the options
%  bandwidth, kernel_scale, offset, or degree (or a combination of those
%  options).
%
%Parameters
% dataPoints         - (required) Input dataset to perform KPCA on.
% kernel             - (required) The kernel to use.
% new_dimensionality - (optional) If not 0, reduce the dimensionality of the
%                      dataset by ignoring the dimensions with the smallest
%                      eighenvalues.
% bandwidth          - (optional) Bandwidt, for gaussian or laplacian kernels.
%                      Default value is 1.
% degree             - (optional)  Degree of polynomial, for 'polynomial' kernel.
%                      Default value 1.
% kernel_scale       - (optional) Scale, for 'hyptan' kernel.  Default value 1.
% offset             - (optional) Offset, for 'hyptan' and 'polynomial' kernels.
%								       Default value is 1.
% scale              - (optional) If true, the data will be scaled before performing
%                      KPCA such that the variance of each feature is 1.

% a parser for the inputs
p = inputParser;
p.addParamValue('new_dimensionality', @isscalar);
p.addParamValue('offset', @isscalar);
p.addParamValue('kernel_scale', @isscalar);
p.addParamValue('bandwidth', @isscalar);
p.addParamValue('degree', @isscalar);
p.addParamValue('scale', false, @(x) (x == true) || (x == false));

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack. transposing to machine learning standards.
result = mex_kernel_pca(dataPoints', kernel, ...
	parsed.new_dimensionality, parsed.scale, ...
	parsed.degree, parsed.offset, ...
	parsed.kernel_scale, parsed.bandwidth);

result = result';

