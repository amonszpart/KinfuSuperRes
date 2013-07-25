clear all;
close all;
clc;
addpath( 'util' );

DepthCalibPath  = 'Calib_Results_depth.mat';
RgbCalibPath    = 'Calib_Results_rgb.mat';
StereoCalibPath = 'Calib_Results_stereo_noreproj.mat';

dep_KK          = load( DepthCalibPath, 'KK' );
dep_KK          = dep_KK.KK;
dep_inv_KK      = load( DepthCalibPath, 'inv_KK' );
dep_inv_KK      = dep_inv_KK.inv_KK;

dep_kc          = load( DepthCalibPath, 'kc' );
dep_kc          = dep_kc.kc;
dep_alpha_c     = load( DepthCalibPath, 'alpha_c' );
dep_alpha_c     = dep_alpha_c.alpha_c;

rgb_KK          = load( RgbCalibPath, 'KK' );
rgb_KK          = rgb_KK.KK;
rgb_inv_KK      = load( RgbCalibPath, 'inv_KK' );
rgb_inv_KK      = rgb_inv_KK.inv_KK;

rgb_kc          = load( RgbCalibPath, 'kc' );
rgb_kc          = rgb_kc.kc;
rgb_alpha_c     = load( RgbCalibPath, 'alpha_c' );
rgb_alpha_c     = rgb_alpha_c.alpha_c;

om              = load( StereoCalibPath, 'om' );
om              = om.om;

R               = load( StereoCalibPath, 'R' );
R               = R.R

T               = load( StereoCalibPath, 'T' );
T               = T.T

dep8 = imread( 'dep8_00000144.pgm' );
subplot(121); imshow(dep8);
img8 = imread( 'img8_00000144.png' );
subplot(122); imshow(img8);

dsize = size( dep8 );
isize = size( img8 );

%%%%
t_gamma = zeros(1,2048);
k1 = 1.1863;
k2 = 2842.5;
k3 = 0.1236;
for i = 0 : 2047
    %depth = k3 * tan(i/k2 + k1);
    depth = 1.0 / (double(i) * -0.0030711016 + 3.3309495161);
    %depth = ((i/2047)^3) * 36 * 255;
    t_gamma(i+1) = depth;
end

figure();
%hold all;
plot(t_gamma)
%plot(disps);
%hold off;
%%%%

mapped = zeros( size(img8,1), size(img8,2) );
mapped_count = 0;
for x = 0 : 1 : dsize(2) - 1
    %x = dsize(2) / 2;
    x
    for y = 0 : 1 : dsize(1) -1
        % y = dsize(1) / 2;
        
        if 1
            z = dep8(y+1, x+1);
            d = t_gamma( round(double(z) / 255.0 * 2047.0 + 1) );
            
            if ( d > 0 )
                %disp('asdf');
            end
            
            if 0
                x_n = normalize( [ x, y ]', ...
                    [ dep_KK(1,1) dep_KK(2,2) ], ...
                    [ dep_KK(1,3) dep_KK(2,3) ], ...
                    dep_kc, dep_alpha_c );
                
                if ( x_n > 0 )
                    disp('asdf2');
                end
                
                p_rgb = project_points2( [ x_n; 1 ],om,T, ...
                    [ rgb_KK(1,1) rgb_KK(2,2) ], ...
                    [ rgb_KK(1,3) rgb_KK(2,3) ], ...
                    rgb_kc, rgb_alpha_c );
            else
                %P_world = [ x_n * double(d); double(d) ];
                P_world = [                            ...
                    (x - dep_KK(1,3)) * d / dep_KK(1,1), ...
                    (y - dep_KK(2,3)) * d / dep_KK(2,2), ...
                    d ]';
                
                P2 = R * P_world + T/1000;
                p_rgb = [ ...
                            P2(1) * rgb_KK(1,1) / P2(3) + rgb_KK(1,3), ...
                            P2(2) * rgb_KK(2,2) / P2(3) + rgb_KK(2,3)  ...
                        ];
                
                %p_rgb = p_rgb * double(d);
            end
        end
        
        if 0
            z = dep8(y+1, x+1);
            d = RawDepthToMeters( z / 255.0 * 2047.0 );
            
            P = DepthToWorld( x, y, z );
            p_rgb = WorldToColor( P );
        end
        
        if ( (p_rgb(1) > 0) && (p_rgb(2) > 0) && (p_rgb(1) < isize(2)) && (p_rgb(2) < isize(1)) )
            dispr = toDisparity( d );
            coords = [ round(p_rgb(2)) + 1, round(p_rgb(1)) + 1 ];
            if ( (mapped(coords(1), coords(2))) == 0 || (dispr < mapped(coords(1), coords(2))) )
                mapped( coords(1), coords(2) ) = dispr;
            end
            
            mapped_count = mapped_count + 1;
            if ( mapped_count < 100 )
                mapped_count
            end
        end
    end
end

img8(:,:,1) = mapped;

mapped_count / size(mapped,1) / size(mapped,2)

figure();
imshow(mapped, [min(mapped(:)), max(mapped(:)) ] );
figure();
imshow( img8, [min(img8(:)), max(img8(:))] );