subplot(4,1,1);
[images, images_line] = convert_to_images(X, 20,20); imshow(images_line)
ylabel('original')
subplot(4,1,2);
[B,W] = nmf_run(X, 2); 
[images, images_line] = convert_to_images(B, 20,20); imshow(images_line)
ylabel('r=2');
subplot(4,1,3);
[B,W] = nmf_run(X, 5); 
[images, images_line] = convert_to_images(B, 20,20); imshow(images_line)
ylabel('r=5');
subplot(4,1,4);
[B,W] = nmf_run(X, 10); 
[images, images_line] = convert_to_images(B, 20,20); imshow(images_line)
ylabel('r=10');