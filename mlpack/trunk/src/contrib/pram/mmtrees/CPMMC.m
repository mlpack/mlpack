function [Acc timeElapsed] = CPMMC(MyData,correctLabel)
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
epsilon = 10; % epsilon controls the precision
W = []; % constraint set
l = 10.0; % balance constraint
flagReturn = 0;

omega_0 = 0.003 * ones(nDim,1); % initial value
b_0 = 0;
xi_0 = 0.5;
% Initialization done

% Find the most violated constraint in the original problem
constraint = zeros(nData,1);
tic
for i = 1:nData
    CountViolate = abs(omega_0'*MyData(:,i)+b_0);
    if(CountViolate<1)
        constraint(i) = 1;
    else
        constraint(i) = 0;
    end; % end of if
end; % end of for
% Add the most violated constraint to W
W = [W;constraint'];

%         run = 0;

while(flagReturn == 0)
    % Step 2: Solve the non-convex optimization problem via CCCP
    [omega,b,xi] = CCCP_MMC_dual2(omega_0,b_0,xi_0,C,W,l,MyData);

    % Step 3: Find the most violated constraint in the original problem
    constraint = zeros(nData,1);
    SumQuit = 0;
    for i = 1:nData
        CountViolate = abs(omega'*MyData(:,i)+b);
        if(CountViolate<1)
            constraint(i) = 1;
            SumQuit = SumQuit + constraint(i) - constraint(i)*CountViolate;
        else
            constraint(i) = 0;
        end; % end of if
    end; % end of for
    SumQuit = SumQuit / nData;

    % Step 4: Decide whether quit the 'while' iteration
    if(SumQuit <= xi * (1 + epsilon))
%    if(SumQuit <= xi  + epsilon)
        flagReturn = 1;
    else
        W = [W;constraint'];
        omega_0 = omega;
        b_0 = b;
        xi_0 = xi;
    end;
    
    if mod(size(W,1), 100) == 0 
      display(size(W));
    end
    
end; % end of while
timeElapsed = toc

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
Acc = count1 / nData

outMarginInd = find(marginMagnitude >= 1);
display(sprintf('sumY:%d, outMargin:%d, %f', ...
		sum(labelLearned), size(outMarginInd, 2),...
		sumFTheta));
