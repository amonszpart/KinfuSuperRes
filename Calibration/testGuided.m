% example: detail enhancement
% figure 6 in our paper

close all;

I = im2double( imread('mapped.png'       ) );
guide = im2double(imread('img8_00000015.png'));

r = 1;
eps = 0.1^2;
filtered = guidedFilter( I(:, :), rgb2gray(guide), r, eps );

%%% OUTPUT %%%
figure('Name','filtered');
subplot(1,2,1);
imshow( filtered );
title( 'filtered' );

q = blend( filtered, guide, .9 );
subplot(1,2,2);
imshow( q );
title( 'filtered + guide' );


%%% ENHANCE %%%
I_enhanced = (I - filtered) * 5 + filtered;
q2 = blend( I_enhanced, guide, .1 );

figure('Name','I\_enhanced+p');
subplot(1,2,1);
imshow( I_enhanced );
title( 'I\_enhanced' );
subplot(1,2,2);
imshow( q2 );
title( 'I\_enhanced+guide' );

