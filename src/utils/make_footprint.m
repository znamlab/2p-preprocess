function footprint = make_footprint(xpix, ypix, nx, ny)

footprint = sparse(zeros(nx, ny));

footprint(sub2ind([nx, ny], xpix, ypix)) = 1;