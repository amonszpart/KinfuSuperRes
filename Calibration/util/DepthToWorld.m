function P = DepthToWorld( x, y, depthValue )
fx_d = 1.0 / 5.9421434211923247e+02;
fy_d = 1.0 / 5.9104053696870778e+02;
cx_d = 3.3930780975300314e+02;
cy_d = 2.4273913761751615e+02;

depth = RawDepthToMeters(depthValue);
P(1) = double((x - cx_d) * depth * fx_d);
P(2) = double((y - cy_d) * depth * fy_d);
P(3) = double(depth);
end