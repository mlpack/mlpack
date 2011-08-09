function [omega,b, timeElapsed] = CP_CCCP_split(MyData, C, l, ...
						initVal)

% In this function, we solve the non-convex optimization problem
% encountered in the Maximum Margin Clustering via CCCP
%
% Input:
%   omega_0:    initial value for omega
%   b_0:        initial value for b
%   xi_0:       initial value for xi
%   C:          regularization parameter in the objective function,
%               soft constraint
%   W:          set of constraints, matrix
%   l:          bound on the label bias
%   MyData:     data set, each sample corresponds to a vector in MyData
%
% Output:
%   omega:   The final value of omega
%   b:       The final value of b
%   xi:      The final value of xi
%   fail:    Whether the CCCP failed (in which case omega = 0, b = 0
%
% Call:[omega,b,xi,fail] = CCCP_MMC_dual(omega_0,b_0,xi_0,C,W,l,MyData)
%
% Author: Bin Zhao
% Created on: 10 July, 2007
% Last updated on : 13 Nov, 2007
%
% Modified by: Parikshit Ram
% Modified on: 11 January, 2011

tic;

% Step 1: initialzation
[nDim,nData] = size(MyData);

% Step 1: Initialization
epsilon = 0.001; % epsilon controls the precision
perQuit = 0.01; % perQuit controls the total step of the CCCP iteration

%testInitVal = - l / sum(sum(MyData, 2));

omega_0 = initVal * ones(nDim,1); % initial value
b_0 = (l - sum(omega_0' * MyData)) / nData;

flagQuit = 0;
iter = 0;

omegaOld = omega_0;
bOld = b_0;
xi_0 = 1 - epsilon;
fValOld = 0.5 * omegaOld' * omegaOld + C * xi_0;

x_k = sum(MyData,2);

%if (isnan(sum(x_k)))
%  display(sprintf('Data NaN: %f', sum(x_k)));
%  keyboard;
%end

% marker for the first loop of the CCCP
floop = 1;

while flagQuit == 0 & iter < 10 % the CCCP loop

  iter = iter + 1;
  W = []; % constraint set

  % constraint = zeros(nData,1);
  % W = [W; constraint'];
  flagReturn = 0; % the cutting plane flag
  xiOld = xi_0;
  
  temp_z_k = zeros(nDim,nData);
  temp_s_k = zeros(nData,1);
  for iData = 1:nData
    temp_s_k(iData) = sign(omegaOld'*MyData(:,iData)+bOld);
    temp_z_k(:,iData) = temp_s_k(iData) * MyData(:,iData);
  end;
  
  display(sum(temp_s_k));

  fail = 0;
  omegaCP = omegaOld;
  bCP = bOld;
  xiCP = xiOld;
  floop2 = 1;

  while(flagReturn == 0) % the cutting plane loop


    if fail == 0
      % Step 3: Find the most violated constraint in the original problem
      constraint = zeros(nData,1);
      SumQuit = 0;
      temp = 0;
      for i = 1:nData
        CountViolate = temp_s_k(i) * (omegaCP'*MyData(:,i)+bCP);
        if(CountViolate<1)
	  constraint(i) = 1;
	  SumQuit = SumQuit + 1 - CountViolate;
	  temp = temp + 1;
        else
	  constraint(i) = 0;
        end; % end of if
      end; % end of for
      SumQuit = SumQuit / nData;

      fprintf(1, 'Obtaining the constraint %f, %d\n', SumQuit, ...
	      temp);
           
      % Step 4: Decide whether quit the 'while' iteration
      %if(SumQuit <= xi * (1 + epsilon))
      if (SumQuit <= xiCP  + epsilon) & ...
	    floop2 == 0
        flagReturn = 1;
	omegaOld = omegaCP;
	bOld = bCP;
	xiOld = xiCP;
	fprintf(1, 'CP loop completed\n');
	keyboard;
	break;
      else
        W = [W;constraint'];
	%fprintf(1, 'adding a constraint\n');
      end;
    else
      display(sprintf('Fail: %f, %f', omegaCP'*omegaCP, bCP));
      break;
      %keyboard;
    end

    [nConstraint, nData] = size(W);
    c_k = mean(W,2);
    s_k = zeros(nConstraint,1);
    z_k = zeros(nDim,nConstraint);


    s_k = W * temp_s_k / nData;
    z_k = temp_z_k * W' / nData;
    
    x_mat = [z_k,-x_k,x_k];
    HQP = x_mat' * x_mat;
    fQP = [-c_k;l;l];
    
    AQP = [ones(1,nConstraint),0,0];
    bQP = C;
    
    Aeq = [-s_k',nData,-nData];
    beq = 0;
    
    LB = zeros(nConstraint+2,1);
    UB = [Inf*ones(1,nConstraint+2)]';

    ops = optimset('LargeScale', 'off',...
		   'Display', 'off',...
		   'MaxIter', 100);
    [XQP,fVal,exitFlag, output, lambda] = quadprog(HQP,fQP,AQP,bQP,...
						   Aeq,beq,LB,UB, ...
						   [],ops);
    
    if (exitFlag <= 0)
      display(exitFlag);
      %keyboard;
      if exitFlag == -7
	fail = 0;
      else
	fail = 1;
      end
    else
      fail = 0;
    end
    
    
    %display(sum(XQP.*XQP));
    omegaCP = x_mat * XQP;
    xiCP = (-fVal - 0.5 * omegaCP' * omegaCP) / C;
    SV_index = find(XQP(1:nConstraint)>0);
    
    % in case no constraints were > 0 (meaning not the theoretical
    % dual optimal, not even close) b cannot be computed from the
    % KKT conditions. Hence we would have to break the CCCP routine
    % and return a failure
    ind_b = -1;
    if numel(SV_index) > 0
      for i = 1:numel(SV_index)
	if s_k(SV_index(i)) ~= 0
	  ind_b = i;
	  break;
	end
      end
    end
    
    
    % if no tight margin constraints present, use the balance
    % constraint to obtain b
    if ind_b == -1 | nConstraint == 1

      fprintf(1, 'using balance constraint\n');
      %keyboard;
      if XQP(nConstraint+1) > 0
	% \sum <w, x_i> + b == -l
	bCP = (-l - omegaCP' * x_k) / nData;
      else
	if XQP(nConstraint+2) > 0
	  bCP = (l - omegaCP' * x_k) / nData;
	else
	  % no tight constraints, hence no way to compute b
	  b_UB = (l - omegaCP' * x_k) / nData;
	  b_LB = (-l - omegaCP' * x_k) / nData;
	  
	  bCP = (b_UB + b_LB) / 2;
	  
	  fprintf(1, 'no tight constraints\n');
	  keyboard;
	  
	end
      end
    else
     
      bCP = (c_k(SV_index(ind_b)) - xiCP - ...
	      omegaCP' * z_k(:,SV_index(ind_b)))...
	     / s_k(SV_index(ind_b));
      
      if isnan(bCP)
	display(sprintf('B new: %f', bCP));
	keyboard;
      end
    end
    
    if floop2 == 1
      floop2 = 0;
    end
      
  end % cutting plane loop
  
  fVal = 0.5 * omegaOld' * omegaOld + C * xiOld;
  
  % if the solution is feasible with the balance constraint
  if (abs(sum(omegaOld' * MyData + bOld)) <= l + 0.0001)
    % check if you are ready to stop
    if fValOld - fVal >= 0 | floop == 1
      if (fValOld - fVal) < perQuit * fValOld & floop == 0
	flagQuit = 1;
      else
	fValOld = fVal;
	if floop == 1
	  floop = 0;
	end
      end
    else
      display(sprintf('negative improvement in obj. fun. %f',...
		      fValOld - fVal));
      keyboard;
    end
  else
    display(sprintf('infeasible solution: l: %f, sum: %f',...
		    l, abs(sum(omegaOld' * MyData + bOld))));
    keyboard;
  end;
  
end % CCCP loop

fprintf(1, 'CCCP loop completed in %d iterations of CCCP', iter);
keyboard;

if flagQuit == 1
  omega = omegaOld;
  b = bOld;
  xi = xiOld;
else
  display(sprintf('Ran out of iterations'));
  keyboard;
end

timeElapsed = toc;