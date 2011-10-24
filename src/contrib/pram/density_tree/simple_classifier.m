function [] = simple_classifier(train, test, classes)
  
  [N d] = size(train); 
  
  % The last column are the classes in both the train and test 
  
  C = cell(classes, 4);
  
  for i = 0:classes -1
    I = [];
    n = 0;
    
    for j = 1:N
      if train(j, d) == i
	n = n+1;
	I(n) = j;
      end
    end
    
    C{i+1, 1} = I; % members of the class
    % display(size(I));
    C{i+1, 2} = n / N; % probability of the class
    display(sprintf('%d, %f', i+1, C{i+1, 2}));
    % display(size(train(I, 1:d-1)));
    C{i+1, 3} = mean(train(I, 1:d-1)); % computing the mean for the class
    if size(I,2) < d-1
      % do something useful
      % computing the dimension-wise variance
      sig = zeros(d-1,1);
      for k = 1:d-1
	sig(k) = (std(train(I,k)))^2;
      end
      C{i+1, 4} = diag(sig);
    else 
      C{i+1, 4} = cov(train(I, 1:d-1));  % computing the covariance
					 % for the class
    end
  end
  
  [N1 d1] = size(test);
  
  error = 0;
  class_error = zeros(classes,1);
  
  for j = 1:N1
    correct_class = int16(test(j,d1)) + 1;
    
    class_conditionals = zeros(classes, 1);
    
    for i = 1:classes
      class_conditionals(i) = log(C{i,2}) + ...
	  log(mvnpdf(test(j,1:d1-1), C{i,3}, C{i,4}));
    end
    
    [max_log_class_cond, predicted_class] = ...
	max(class_conditionals);
    
    if correct_class ~= predicted_class
      error = error + 1;
      class_error(correct_class) = class_error(correct_class) + 1;
    end
    
  end
  
  error_p = error * 100 / N1;
  
  display(error_p);
  display(class_error);
  
  
