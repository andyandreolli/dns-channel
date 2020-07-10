import numpy as np
import tactical as tct
import dns_channel as ch
import itertools
import progressbar



def read_psd(fdir, **kwargs):

    y_symm = kwargs.get('y_symm', True)
    
    fname = fdir + 'psd.bin' # generate file name

    # fetch nx, ny, nz
    dnsdict = ch.read_dnsin(fdir)
    mesh = ch.mesh(dnsdict)

    # create memmap
    diskacc = np.memmap(fname, mode='r', dtype=np.float64, shape=(mesh.ny+3, mesh.nx+1, 2*mesh.nz+1, 6))

    # check file size and allocate memory
    tct.io.size_ram_check(fname)
    all_spectra = np.zeros((6, mesh.nx+1, mesh.nz+1, mesh.ny+1))

    with progressbar.ProgressBar(max_value=6*(mesh.nx+1)*(mesh.nz+1)*(mesh.ny+1)) as bar:
        for ii, (iz, ic, ix, iy) in enumerate(itertools.product(range(mesh.nz+1), range(6), range(mesh.nx+1), range(mesh.ny+1))):
            izb = -1-iz
            izm = mesh.nz + izb
            all_spectra[ic, ix, izm, iy] = diskacc[iy, ix, iz, ic] + diskacc[iy, ix, izb, ic]
            if izb+iz+1 == 0:
                all_spectra[ic, ix, iz, iy] = all_spectra[ic, ix, iz, iy]/2
            bar.update(ii)

    # average on y (if requested)
    if y_symm:
        all_spectra += all_spectra[:,:,:,::-1]
        all_spectra /= 2

    return all_spectra, mesh.kx, mesh.kz, mesh.y[1:-1]