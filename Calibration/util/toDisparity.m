function disp = toDisparity( depth )
k1 = 1.1863;
k2 = 2842.5;
k3 = 0.1236;
disp = (atan( depth / k3 ) - k1) * k2;

%disp = (double(depth) / 256 / 36 * 1000) ^ (1/3) * 2048;

%disp = (1.0 / depth - 3.3309495161) / -0.0030711016;
%disp = (1.0 / depth - 3.3309495161) / -0.0030711016;

if ( disp > 2047 )
    disp = 2047;
elseif (disp < 0 )
    disp = 0;
end

end