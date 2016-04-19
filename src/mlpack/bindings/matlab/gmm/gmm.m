function result = gmm(dataPoints, varargin)
%Gaussian Mixture Model (GMM) Training
%
%  This program takes a parametric estimate of a Gaussian mixture model (GMM)
%  using the EM algorithm to find the maximum likelihood estimate.  The model is
%  saved to an XML file, which contains information about each Gaussian.
%
%Parameters:
% dataPoints- (required) Matrix containing the data on which the model will be fit
% seed      - (optional) Random seed.  If 0, 'std::time(NULL)' is used.
%					    Default value is 0.
% gaussians - (optional) Number of gaussians in the GMM. Default value is 1.

% a parser for the inputs
p = inputParser;
p.addParamValue('gaussians', 1, @isscalar);
p.addParamValue('seed', 0, @isscalar);

% parsing the varargin options
p.parse(varargin{:});
parsed = p.Results;

% interfacing with mlpack
result = mex_gmm(dataPoints', parsed.gaussians, parsed.seed);




