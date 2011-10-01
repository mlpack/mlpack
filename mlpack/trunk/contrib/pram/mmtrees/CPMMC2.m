function [Acc timeElapsed] = CPMMC2(MyData,correctLabel)
% Cutting Plane Maximum Margin Clustering Algorithm (CPMMC)
%
% Input :   MyData :        The data sample matrix, each column corresponds to a
%                           data sample
%           correctLabel :  The correct labeling vector; this vector will
%                           only be used while calculating the clustering
%                           accuracy
%
% Output :  Acc :           Clustering accuray
%           timeElapsed :   CPU-time in seconds of CPMMC
%
% Step 1 :  Initialization, set the values for parameters 'C, epsilon, l, omega_0, b_0, xi_0'
%           ps: to my experience, 'C', 'l' and 'omega_0' need to be fine
%           tuned
% Step 2 :  Start CPMMC main procedure, call function 'CCCP_MMC_dual' for solving the
%           CCCP problem
% Step 3 :  Select the most violated constraint and add it into Omega
%
% Author: Bin Zhao
% Created on: 10 July, 2007
% Last Updated on : 19 April, 2008

[nDim,nData] = size(MyData);

% Step 1: Initialization
C = 0.01; % same as in SVM
l = 10.0; % balance constraint
initVal = 0.03;

fprintf(1, 'calling the splitter on %d points....\n', nData);

[omega, b, timeElapsed] = CP_CCCP_split(MyData, C, l, initVal);

labelLearned = zeros(nData,1);
labelLearned = sign(omega'*MyData + b);
marginMagnitude = abs(omega'*MyData + b);
sumFTheta = sum(omega'*MyData + b);

count1 = 0;
for i = 1:nData
    if(labelLearned(i) == correctLabel(i))
        count1 = count1 + 1;
    end;
end;
Acc = count1 / nData;

outMarginInd = find(marginMagnitude >= 1);
display(sprintf('Accuracy: %f', Acc));
display(sprintf('sumY:%d, outMargin:%d, %f', ...
		sum(labelLearned), size(outMarginInd, 2),...
		sumFTheta));
