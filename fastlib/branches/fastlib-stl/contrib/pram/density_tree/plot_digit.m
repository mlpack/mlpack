function [] = plot_digit(digit_file, image_file)

     digit = csvread(digit_file);
     N = size(digit, 2);
     for i = 1:N
       if digit(1,i) ~= 0
         digit(1,i) = log10(digit(1,i));
       end
     end

     size(digit) 
     [S I] = sort(digit);
     mean_x = 0; num = 0;
     sqd_x = 0;
     for i = 1:N
       if S(1,i) < 0
         mean_x = mean_x + S(1,i);
         sqd_x = sqd_x + S(1,i)*S(1,i);
         num = num + 1;
       end
     end
     std_x = (((sqd_x*num) - (mean_x * mean_x)) / ...
	      (num * (num - 1)))^0.5;
     mean_x = mean_x / num;
     
     A = []; max = -100; min = 500;
     for i = 1:N
       if digit(1,i) ~= 0
         A(i) = ((digit(1,i) - mean_x) / std_x) + 100;
           if A(i) > max
              max = A(i);
           end
           if A(i) < min
	      min = A(i);
           end
       else
         A(i) = 0;
       end
     end

     for i = 1:N
        if digit(1,i) ~= 0
           A(i) = (90 * (A(i) - min) / (max - min)) + 10 ;
        end
     end

     A2 = A;
     for i = 1:N
	if digit(1,i) ~= 0
           A2(i) = 100 - A2(i) + 10;
        end
     end

     rgb_matrix(:,:,1) = uint8(reshape(A,sqrt(N),sqrt(N))'.* 255 / 100);
     rgb_matrix(:,:,2) = uint8(reshape(A2,sqrt(N),sqrt(N))'.* 255 / 100);
     rgb_matrix(:,:,3) = uint8(zeros(sqrt(N)));

     figure; hold;
     image(rgb_matrix);
     axis image;
     axis off;
     print('-depsc2', image_file);
     hold;

     clear digit;
