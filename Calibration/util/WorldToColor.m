function p_rgb = WorldToColor( pt )
rotationMatrix = [ ...
    9.9984628826577793e-01, 1.2635359098409581e-03, -1.7487233004436643e-02, 0;
    -1.4779096108364480e-03, 9.9992385683542895e-01, -1.2251380107679535e-02, 0;
    1.7470421412464927e-02, 1.2275341476520762e-02,  9.9977202419716948e-01, 0;
    0, 0, 0, 1 ];

translation = [ 1.9985242312092553e-02, -7.4423738761617583e-04, -1.0916736334336222e-02, -1 ];
finalMatrix = rotationMatrix' * cat( 2, eye(4,3), -translation');

ratX = (1280/640);
ratY = (1280/640);
fx_rgb = ratX * 5.2921508098293293e+02;
fy_rgb = ratY * 5.2556393630057437e+02;
cx_rgb = ratX * 3.2894272028759258e+02;
cy_rgb = ratY * 2.6748068171871557e+02;

transformedPos = finalMatrix * [ pt'; 1];
invZ = 1.0 / transformedPos(3);

p_rgb(1) = round((transformedPos(1) * fx_rgb * invZ) + cx_rgb);
p_rgb(2) = round((transformedPos(2) * fy_rgb * invZ) + cy_rgb);
end