function d = RawDepthToMeters( depthValue )
if (depthValue < 2047)
    d = (1.0 / (double(depthValue) * -0.0030711016 + 3.3309495161));
    %d = ((depthValue/2047.0)^3) * 36.0 * 255.0 / 1000.0;
else
    d = 0.0;
end
end