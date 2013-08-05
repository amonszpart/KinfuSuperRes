addpath( 'util' );

I1 = double(imread('/media/Storage/Dropbox_linux/Dropbox/UCL/project/results/presentation_300713/crossFiltered_02.png'))/255.0;
I2 = double(imread('/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130725_1809/img8_00000001.png'))/255.0;

imshow( [I1 I2] );
o = blend( I1, I2 );
imshow( o );
