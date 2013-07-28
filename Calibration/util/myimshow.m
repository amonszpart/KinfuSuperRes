function myimshow( I )
    imshow( I, [ min(I(:)), max(I(:)) ] );
end