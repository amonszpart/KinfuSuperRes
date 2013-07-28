function output = blend( I, J, alpha )

    if ~exist( 'alpha', 'var' ),
        alpha = .5;
    elseif isempty( alpha ),
        alpha = .5;
    end

    [ h1, w1, c1 ] = size(I);
    [ h2, w2, c2 ] = size(J);
    c3 = max( c1, c2 );

    if ( (h1 ~= h2) || (w1 ~= w2) )
        disp('Warning, input in blend does need images to be the same SIZE!');
        return;
    end

    output = zeros( h1, w1, c3 );

    for c = 1 : c3
        output( :, :, c ) = alpha * I(:,:,min(c,c1)) + (1.0 - alpha) * J(:,:,min(c,c2));
    end

end

