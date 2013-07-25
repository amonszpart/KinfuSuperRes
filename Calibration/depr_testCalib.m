clc;

% load('Calib_Results_stereo.mat')

dimg = imread( 'dep8_00000015.png');
imshow( dimg );
img = imread( 'img8_00000015.png');
imshow( img );

%X = 10*randn(3,n);
om = [ 0.04869 0.00974 -0.00206 ];
T = [ -33.25604 9.94095 -3.25410 ];
f = 1000*rand(2,1);
c = 1000*randn(2,1);
k = 0.5*randn(5,1);
alpha = 0.01*randn(1,1);
[ x, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha ] = project_points2(X, om, T,f,c,k,alpha);


x_n = normalize_pixel( x_kk, fc, cc, kc, alpha_c );
project_points2(X, om, T,f,c,k,alpha);
