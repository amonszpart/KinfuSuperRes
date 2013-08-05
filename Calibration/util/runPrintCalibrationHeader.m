function runPrintCalibrationHeader()
    path ='/media/Storage/workspace_ubuntu/rec/imgs_20130805_1047_calibPrism4/';
    lp = [ path 'Calib_Results_ir_left.mat' ];
    rp = [ path 'Calib_Results_rgb_right.mat' ];
    sp = [ path 'Calib_Results_stereo_noreproj.mat' ];
    
    printCalibrationHeader( lp, rp, sp, path );
    
%     printCalibrationHeader( [ path 'Calib_Results_ir.mat' ], ...
%                             [ path 'Calib_Results_rgb.mat' ], ...
%                             [ path 'Calib_Results_stereo.mat' ] );
end

function printCalibrationHeader( left_path, right_path, stereo_path, out_path )

    DepthCalibPath  = left_path;
    RgbCalibPath    = right_path;
    StereoCalibPath = stereo_path;
    
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
    
    prefix = '#define ';
    fid = fopen([ out_path '/calibration.h'],'w');
    if ( fid )
        fprintf( fid, '// Calibration parameters from:\n' );
        fprintf( fid, '// %s\n', [ left_path ] );
        fprintf( fid, '// %s\n', [ right_path ] );
        fprintf( fid, '// %s\n', [ stereo_path ] );
        fprintf( fid, '\n\n');
        
        fprintf( fid, '%s %s ', [ prefix, 'FX_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_KK(1,1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'FY_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_KK(2,2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'CX_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_KK(1,3) ] );
        fprintf( fid, '%s %s ', [ prefix, 'CY_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_KK(2,3) ] );
        fprintf( fid, '\n');
        
        fprintf( fid, '%s %s ', [ prefix, 'K1_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_kc(1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K2_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_kc(2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K3_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_kc(3) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K4_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_kc(4) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K5_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_kc(5) ] );
        fprintf( fid, '%s %s ', [ prefix, 'ALPHA_RGB' ] ), fprintf( fid, '%ff\n', [ rgb_alpha_c ] );
        fprintf( fid, '\n');
        
        fprintf( fid, '%s %s ', [ prefix, 'FX_D' ] ), fprintf( fid, '%ff\n', [ dep_KK(1,1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'FY_D' ] ), fprintf( fid, '%ff\n', [ dep_KK(2,2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'CX_D' ] ), fprintf( fid, '%ff\n', [ dep_KK(1,3) ] );
        fprintf( fid, '%s %s ', [ prefix, 'CY_D' ] ), fprintf( fid, '%ff\n', [ dep_KK(2,3) ] );
        fprintf( fid, '\n');
        
        fprintf( fid, '%s %s ', [ prefix, 'K1_D' ] ), fprintf( fid, '%ff\n', [ dep_kc(1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K2_D' ] ), fprintf( fid, '%ff\n', [ dep_kc(2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K3_D' ] ), fprintf( fid, '%ff\n', [ dep_kc(3) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K4_D' ] ), fprintf( fid, '%ff\n', [ dep_kc(4) ] );
        fprintf( fid, '%s %s ', [ prefix, 'K5_D' ] ), fprintf( fid, '%ff\n', [ dep_kc(5) ] );
        fprintf( fid, '%s %s ', [ prefix, 'ALPHA_D' ] ), fprintf( fid, '%ff\n', [ dep_alpha_c ] );
        fprintf( fid, '\n');
        
        fprintf( fid, '%s %s ', [ prefix, 'R1' ] ), fprintf( fid, '%ff\n', [ R(1,1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R2' ] ), fprintf( fid, '%ff\n', [ R(1,2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R3' ] ), fprintf( fid, '%ff\n', [ R(1,3) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R4' ] ), fprintf( fid, '%ff\n', [ R(2,1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R5' ] ), fprintf( fid, '%ff\n', [ R(2,2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R6' ] ), fprintf( fid, '%ff\n', [ R(2,3) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R7' ] ), fprintf( fid, '%ff\n', [ R(3,1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R8' ] ), fprintf( fid, '%ff\n', [ R(3,2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'R9' ] ), fprintf( fid, '%ff\n', [ R(3,3) ] );
        fprintf( fid, '\n');
        
        fprintf( fid, '%s %s ', [ prefix, 'T1' ] ), fprintf( fid, '%ff\n', [ T(1) ] );
        fprintf( fid, '%s %s ', [ prefix, 'T2' ] ), fprintf( fid, '%ff\n', [ T(2) ] );
        fprintf( fid, '%s %s ', [ prefix, 'T3' ] ), fprintf( fid, '%ff\n', [ T(3) ] );
    end
    fclose( fid );

    % // Distortion coefficients
% #define K1_RGB 2.5785516449232132e-01f
% #define K2_RGB -9.1141470196267182e-01f
% #define P1_RGB 3.0173013316440469e-04f
% #define P2_RGB 2.5422024034001231e-03f
% #define K3_RGB 1.1823504884394158e+00f
% 

% 
% #define K1_D  -1.3708537316819339e-01f
% #define K2_D 7.2482751812234414e-01f
% #define P1_D 8.0826809257389550e-04f
% #define P2_D 3.4151576458975323e-03f
% #define K3_D -1.4621396186358457e+00f
% 
% // Inverse Rotation matrix in column major order.
% #define R1 0.999985794494467f
% #define R2 -0.003429138557773f
% #define R3 0.00408066391266f
% #define R4 0.003420377768765f
% #define R5 0.999991835033557f
% #define R6 0.002151948451469f
% #define R7 -0.004088009930192f
% #define R8 -0.002137960469802f
% #define R9 0.999989358593300f
% 
% // Translation vector
% #define T1 -2.2142187053089738e-02f
% #define T2 1.4391632009665779e-04f
% #define T3 7.9356552371601212e-03f
    
end
