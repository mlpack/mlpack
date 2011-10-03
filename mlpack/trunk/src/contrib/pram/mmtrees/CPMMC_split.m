function [omega, b, timeElapsed] = CPMMC_split(MyData, C, l, ...
					       initVal)

% Cutting Plane Maximum Margin Clustering Algorithm (CPMMC)
%
% Input :   MyData :        The data sample matrix
% Output :  timeElapsed :   CPU-time in seconds of CPMMC


[nDim,nData] = size(MyData);

% Step 1: Initialization
% C = 0.01; % same as in SVM
epsilon = 10; % epsilon controls the precision
W = []; % constraint set
flagReturn = 0;

omega_0 = initVal * ones(nDim,1); % initial value
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
    [omega,b,xi, fail] = CCCP_MMC_dual(omega_0,b_0,xi_0,C,W,l,MyData);

    if fail == 0
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
      %display(sprintf('%f, %f', omega'*omega, b));
      %display(sprintf('Label sum:%d, %g - %g', sum(sign(omega'*MyData+b)), ...
      %      SumQuit, xi*(1+epsilon)));
    else
      flagReturn = 1;
      %display(sprintf('Fail: %f, %f', omega'*omega, b));
    end
   
    if mod(size(W,1), 50) == 0 
      display(size(W));
    end
    
    if size(W,1) > 1000
      flagReturn = 1;
    end
        
end; % end of while
timeElapsed = toc;
