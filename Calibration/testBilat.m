addpath( '/media/Storage/workspace_ubuntu/rec/calibTrollKinect');
addpath( 'bilateralFilter' );
addpath( 'guidedFilter' );
addpath( '/home/bontius/workspace/3rdparty/wildcardsearch');

close all;
clc;

% get all files
rootdir = '/media/Storage/workspace_ubuntu/rec/calibTrollKinect'
searchstr = 'img8*.png';
files = wildcardsearch( rootdir, searchstr, true, true )
depfiles = files;
for i = 1 : length(files)
    strrep( depfiles{i}, 'img8', 'dep8' );
end

% work
%for id = 1 : length(files)
I = imread('/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130725_1809/dep16_00000001.png_mapped.png');
data  = im2double( double(I)/10001.0 );
guide = im2double( imread('/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130725_1809/img8_00000001.png') );

figure('Name','data');
imshow(data);
figure('Name','guide');
imshow(guide);

% bilateralFilter( data, edge, edgeMin, edgeMax, sigmaSpatial, sigmaRange, samplingSpatial, samplingRange )
output = bilateralFilter( data, rgb2gray(guide), 0, 1, 4, 0.03 );
%max(data(:))
%output = bilateralFilter( data, data, 0, 1, 40, 0.2 );

figure('Name','filtered');
myimshow( output );

figure('Name','filtered+guide');
blended = blend(output,guide,.95);
imshow( blended, [min(blended(:)),max(blended(:))]);
title('blend');

figure();
overlay = zeros( [size(output), 3 ] );
overlay(:,:,1) = output * 10;
overlay(:,:,2) = guide(:,:,2);
overlay(:,:,3) = guide(:,:,3);
myimshow( overlay );
%end
