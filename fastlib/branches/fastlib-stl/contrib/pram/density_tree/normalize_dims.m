function [NM] = normalize_dims(M)

     mean_M = mean(double(M));
     std_M = std(double(M));
     for i = 1:size(mean_M, 2)
       if std_M(1, i) == 0
	 display(i);
         std_M(1,i) = 1;
       end
     end

     NM = double(M) - repmat(mean_M, size(M,1), 1);
     NM = NM ./ repmat(std_M, size(M,1), 1);
