addpath( '/media/Storage/workspace_ubuntu/rec/calibTrollKinect');
addpath( 'bilateralFilter' );
addpath( 'guidedFilter' );

close all;
%clear all;
clc;
data  = im2double( imread('mapped.png') );
guide = im2double( imread('img8_00000015.png') );

figure();
imshow(data);
figure();
imshow(guide);

% bilateralFilter( data, edge, edgeMin, edgeMax, sigmaSpatial, sigmaRange, samplingSpatial, samplingRange )
output = bilateralFilter( data, rgb2gray(guide), 0, 1, 10, 0.01 );

figure();
imshow(output);

alpha = .1;
figure();
overlay = guide * alpha;
overlay(:,:,1) = overlay(:,:,1) + output * (1 - alpha);
overlay(:,:,2) = overlay(:,:,1) + output * (1 - alpha);
overlay(:,:,3) = overlay(:,:,1) + output * (1 - alpha);
imshow(overlay);
