% example: detail enhancement
% figure 6 in our paper

close all;

I = im2double( imread('mapped.png'       ) );
p = im2double(imread('img8_00000015.png'));

r = 16;
eps = 0.1^2;
output = guidedFilter( I(:, :), rgb2gray(p), r, eps );

q = blend( output, p, .9 );
figure('Name',);
imshow(q);

I_enhanced = (I - output) * 5 + output;
q2 = blend( I_enhanced, p, .1 );

figure();
imshow(q2);
