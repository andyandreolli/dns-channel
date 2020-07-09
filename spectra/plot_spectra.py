import numpy as np
import h5py
from matplotlib.pyplot import subplots, imshow, colorbar, savefig
from matplotlib.colors import LogNorm





def get_spectra(v_field, npts, y, kz):

    # unpack input
    nx = npts[0]
    nz = npts[2]

    all_spectra = (v_field * v_field.conj()).real # data needs to be cast to real anyway

    # average on y
    all_spectra += all_spectra[:,:,:,::-1]
    all_spectra /= 2

    # consequently resize spectra
    y_middle = (y[0] + y[-1])/2
    reshaping = np.broadcast_to(y<=y_middle, all_spectra.shape)
    all_spectra = all_spectra[reshaping].reshape((3, nx+1, 2*nz+1, -1))

    # consequently resize y
    y = y[y<=y_middle]

    # now sum over kz (sum contributions with same abs(kz) but different sign)
    temp_spectra = all_spectra[:,:,::-1,:]
    mask = np.broadcast_to(np.reshape(kz>0, (2*nz+1,1)), all_spectra.shape) # here mask does not include 0
    all_spectra[mask] += temp_spectra[mask]

    # consequently resize spectra
    reshaping = np.broadcast_to(np.reshape(kz>=0, (2*nz+1,1)), all_spectra.shape)
    all_spectra = all_spectra[reshaping].reshape((3, nx+1, nz+1, len(y)))

    # consequently resize kz
    kz = kz[kz >= 0]

    # now multiply times two over kx (except for kx = 0)
    all_spectra[:, 1:, :, :] *= 2

    return all_spectra, y, kz





def get_variance(all_spectra):
    var = np.sum(all_spectra, axis=(1,2))
    k_times_2 = np.sum(var, axis=0)
    return var, k_times_2





def plot_premultiplied(all_spectra, components, desired_ys, y, kx, kz, **kwargs):

    # unpack input
    nxp1 = len(kx)
    with_wavelengths = kwargs.get('with_wavelengths', False)
    plot_title = kwargs.get('title', '')
    save_fig = kwargs.get('save_fig', '')
    if not save_fig == '':
        save_format = save_fig
        save_fig = True
    else:
        save_fig = False
        save_format = ''
    if not plot_title == '':
        plot_title = plot_title + ', '

    if not (hasattr(components, '__iter__') or hasattr(components, '__getitem__')):
        components = [components]
    if not (hasattr(desired_ys, '__iter__') or hasattr(desired_ys, '__getitem__')):
        desired_ys = [desired_ys]

    for component in components:
        
        # convert component to index
        idx = 0
        component = component.lower()
        if component == 'y':
            idx = 1
        elif component == 'z':
            idx = 2
        elif not component == 'x':
            print('Error: invalid component. Please pass either "x", "y", "z" as the second argument.')
        
        # select desired spectrum
        spectrum = all_spectra[idx, :, :, :]

        for desired_y in desired_ys:

            y_idx = (np.abs(y - desired_y)).argmin()

            # calculate premultiplier using array broadcasting
            bcast_premult = kx.reshape((nxp1, 1)) * kz
            bcast_premult = bcast_premult.reshape((len(kx), len(kz), 1))
            bcast_premult = abs(bcast_premult)

            # premultiply
            premultiplied = spectrum * bcast_premult

            # actually print
            fig, ax = subplots(figsize=(4.7,4))
            ax.set_xscale("log")
            ax.set_yscale("log")
            # when plotting, 0 modes are excluded
            if with_wavelengths:
                pos = ax.pcolormesh(2*np.pi/kx[1:],2*np.pi/kz[1:],premultiplied[1:,1:,y_idx].transpose(), linewidth=0, rasterized=True)
                pos.set_edgecolor('face')
                ax.set_xlabel(r'$\lambda_x$')
                ax.set_ylabel(r'$\lambda_z$')
            else:
                pos = ax.pcolormesh(kx[1:], kz[1:], premultiplied[1:, 1:, y_idx].transpose(), linewidth=0, rasterized=True)
                pos.set_edgecolor('face')
                ax.set_xlabel(r'$k_x$')
                ax.set_ylabel(r'$k_z$')
            ax.set_xlim([kx[1], kx[-1]])
            ax.set_ylim([kz[1], kz[-1]])
            fig.colorbar(pos)
            cmp = gcmp(component)
            ax.set_title(plot_title + r'$k_xk_z \langle \hat{'+cmp+'}^\dagger \hat{'+cmp+'} \\rangle$, ' + 'y = {}'.format(round(y[y_idx], 3)))
            if save_fig:
                figname = 'Figs/' + plot_title[:-2].lower() + '_' + 'premultiplied' + '_' + component + '_y' + str(round(y[y_idx], 3)).replace('.','') + '.' + save_format
                savefig(figname, format=save_format)





def plot(all_spectra, components, desired_ys, y, kx, kz, **kwargs):

    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    plot_title = kwargs.get('title', '')
    save_fig = kwargs.get('save_fig', '')
    limx = kwargs.get('limx', None)
    limy = kwargs.get('limy', None)
    if not save_fig == '':
        save_format = save_fig
        save_fig = True
    else:
        save_fig = False
        save_format = ''
    if not plot_title == '':
        plot_title = plot_title + ', '

    # unpack input
    if not (hasattr(components, '__iter__') or hasattr(components, '__getitem__')):
        components = [components]
    if not (hasattr(desired_ys, '__iter__') or hasattr(desired_ys, '__getitem__')):
        desired_ys = [desired_ys]

    for component in components:
        
        # convert component to index
        idx = 0
        component = component.lower()
        if component == 'y':
            idx = 1
        elif component == 'z':
            idx = 2
        elif not component == 'x':
            print('Error: invalid component. Please pass either "x", "y", "z" as the second argument.')
        
        # select desired spectrum
        spectrum = all_spectra[idx, :, :, :]

        for desired_y in desired_ys:

            y_idx = (np.abs(y - desired_y)).argmin()

            # actually print
            fig, ax = subplots(figsize=(4.7,4))
            ax.set_xlabel(r'$k_x$') 
            ax.set_ylabel(r'$k_z$')
            pos = ax.pcolormesh(kx,kz,spectrum[:,:,y_idx].transpose(), vmin=vmin, vmax=vmax, linewidth=0, rasterized=True)
            pos.set_edgecolor('face')
            fig.colorbar(pos)
            cmp = gcmp(component)
            ax.set_title(plot_title + r'$ \tilde{'+cmp+'}^\dagger \\tilde{'+cmp+'}$, ' + 'y = {}'.format(round(y[y_idx], 3)))
            ax.set_xlim(limx)
            ax.set_ylim(limy)
            if save_fig:
                figname = 'Figs/' + plot_title[:-2].lower() + '_' + component + '_y' + str(round(y[y_idx], 3)).replace('.','') + '.' + save_format
                savefig(figname, format=save_format)





def plot_cumulative_zy(all_spectra, components, y, kz, **kwargs):

    # unpack input
    with_wavelengths = kwargs.get('with_wavelengths', True)
    plot_title = kwargs.get('title', '')
    save_fig = kwargs.get('save_fig', '')
    if not save_fig == '':
        save_format = save_fig
        save_fig = True
    else:
        save_fig = False
        save_format = ''
    if not plot_title == '':
        plot_title = plot_title + ', '

    if not (hasattr(components, '__iter__') or hasattr(components, '__getitem__')):
        components = [components]

    for component in components:
        
        # convert component to index
        idx = 0
        component = component.lower()
        if component == 'y':
            idx = 1
        elif component == 'z':
            idx = 2
        elif not component == 'x':
            print('Error: invalid component. Please pass either "x", "y", "z" as the second argument.')
        
        # select desired spectrum
        spectrum = all_spectra[idx, :, :, :]

        # sum along x-axis
        spectrum = spectrum.sum(axis=0)

        # premultiply (reshape kz for broadcasting)
        premultiplied = spectrum * abs(kz.reshape(len(kz), 1))

        # actually print
        fig, ax = subplots(figsize=(4.7,4))
        ax.set_xscale("log") # logarithmic scale only for lambda z
        # when plotting, 0 modes are excluded
        if with_wavelengths:
            pos = ax.pcolormesh(2*np.pi/kz[1:], y, premultiplied[1:,:].transpose(), linewidth=0, rasterized=True)
            pos.set_edgecolor('face')
            ax.set_xlabel(r'$\lambda_z$') 
            ax.set_ylabel(r'$y$')
        else:
            pos = ax.pcolormesh(kz[1:], y, premultiplied[1:,:].transpose(), linewidth=0, rasterized=True)
            pos.set_edgecolor('face')
            ax.set_xlabel(r'$k_z$') 
            ax.set_ylabel(r'$y$')
        fig.colorbar(pos)
        cmp = gcmp(component)
        ax.set_title(plot_title + r'$k_z\sum\,_{k_x} \langle \hat{'+cmp+'}^\dagger\hat{'+cmp+'} \\rangle(k_x, k_z, y)$')
        if save_fig:
            figname = 'Figs/' + plot_title[:-2].lower() + '_cumulative_' + component + '.' + save_format
            savefig(figname, format=save_format)





def plot_cumulative_xy(all_spectra, components, y, kx, **kwargs):

    # unpack input
    with_wavelengths = kwargs.get('with_wavelengths', True)
    plot_title = kwargs.get('title', '')
    save_fig = kwargs.get('save_fig', '')
    if not save_fig == '':
        save_format = save_fig
        save_fig = True
    else:
        save_fig = False
        save_format = ''
    if not plot_title == '':
        plot_title = plot_title + ', '

    if not (hasattr(components, '__iter__') or hasattr(components, '__getitem__')):
        components = [components]

    for component in components:
        
        # convert component to index
        idx = 0
        component = component.lower()
        if component == 'y':
            idx = 1
        elif component == 'z':
            idx = 2
        elif not component == 'x':
            print('Error: invalid component. Please pass either "x", "y", "z" as the second argument.')
        
        # select desired spectrum
        spectrum = all_spectra[idx, :, :, :]

        # sum along x-axis
        spectrum = spectrum.sum(axis=1)

        # premultiply (reshape kz for broadcasting)
        premultiplied = spectrum * abs(kx.reshape(len(kx), 1))

        # actually print
        fig, ax = subplots(figsize=(4.7,4))
        ax.set_xscale("log") # logarithmic scale only for lambda z
        # when plotting, 0 modes are excluded
        if with_wavelengths:
            pos = ax.pcolormesh(2*np.pi/kx[1:], y, premultiplied[1:,:].transpose(), linewidth=0, rasterized=True)
            pos.set_edgecolor('face')
            ax.set_xlabel(r'$\lambda_x$') 
            ax.set_ylabel(r'$y$')
        else:
            pos = ax.pcolormesh(kx[1:], y, premultiplied[1:,:].transpose(), linewidth=0, rasterized=True)
            pos.set_edgecolor('face')
            ax.set_xlabel(r'$k_x$') 
            ax.set_ylabel(r'$y$')
        fig.colorbar(pos)
        ax.set_title(plot_title + r'$k_x\sum\,_{k_z} E_{'+component+component+'}(k_x, k_z, y)$')
        if save_fig:
            figname = 'Figs/' + plot_title[:-2].lower() + '_cumulative_' + component + '.' + save_format
            savefig(figname, format=save_format)





def load_LM_spectra(filename, **kwargs):

    rolls_to_remove_start = kwargs.get('rm_roll0', 0)
    rolls_to_remove_end = kwargs.get('rm_roll1', 0)
    copy_left_mode = kwargs.get('acc_rem', False)

    f = h5py.File(filename, 'r')

    nx = f['nx'].value
    nz = f['nz'].value

    ux_spectrum = f['Euu_kx'].value.transpose()
    sz_ux = ux_spectrum.shape

    uz_spectrum = f['Euu_kz'].value.transpose()
    for ii in range(int(nz)):
        uz_spectrum[:,ii] += uz_spectrum[:,-1-ii]
    uz_spectrum = uz_spectrum[:,:int(nz/2)]

    vx_spectrum = f['Evv_kx'].value.transpose()

    vz_spectrum = f['Evv_kz'].value.transpose()
    for ii in range(int(nz)):
        vz_spectrum[:,ii] += vz_spectrum[:,-1-ii]
    vz_spectrum = vz_spectrum[:,:int(nz/2)]

    wx_spectrum = f['Eww_kx'].value.transpose()

    wz_spectrum = f['Eww_kz'].value.transpose()
    for ii in range(int(nz)):
        wz_spectrum[:,ii] += wz_spectrum[:,-1-ii]
    wz_spectrum = wz_spectrum[:,:int(nz/2)]

    uvx_spectrum = f['Euv_kx'].value.transpose()

    uvz_spectrum = f['Euv_kz'].value.transpose()
    for ii in range(int(nz)):
        uvz_spectrum[:,ii] += uvz_spectrum[:,-1-ii]
    uvz_spectrum = uvz_spectrum[:,:int(nz/2)]

    y = f['LABS_COL'].value

    # some reshaping
    y = y[:int(len(y)/2)]
    ux_spectrum = ux_spectrum[:len(y), :]
    uz_spectrum = uz_spectrum[:len(y), :]
    vx_spectrum = vx_spectrum[:len(y), :]
    vz_spectrum = vz_spectrum[:len(y), :]
    wx_spectrum = wx_spectrum[:len(y), :]
    wz_spectrum = wz_spectrum[:len(y), :]
    uvx_spectrum = uvx_spectrum[:len(y), :]
    uvz_spectrum = uvz_spectrum[:len(y), :]
    
    dkx = 2*np.pi/f['Lx'].value
    dkz = 2*np.pi/f['Lz'].value

    kx = np.arange(sz_ux[1]) * dkx
    kz = np.arange(nz/2) * dkz

    # remove rolls if necessary
    if copy_left_mode:
        uz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = uz_spectrum[:,rolls_to_remove_start-1].reshape((len(y),1))
        vz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = vz_spectrum[:,rolls_to_remove_start-1].reshape((len(y),1))
        wz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = wz_spectrum[:,rolls_to_remove_start-1].reshape((len(y),1))
        uvz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = uvz_spectrum[:,rolls_to_remove_start-1].reshape((len(y),1))
    else:
        uz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = 0
        vz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = 0
        wz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = 0
        uvz_spectrum[:,rolls_to_remove_start:rolls_to_remove_end] = 0

    return y, kx, kz, ux_spectrum, uz_spectrum, vx_spectrum, vz_spectrum, wx_spectrum, wz_spectrum, uvx_spectrum, uvz_spectrum





def get_variance_LM(spectrum):
    var = np.sum(spectrum, axis=(1))
    return var





def gcmp(xis):
    if xis == 'x' or xis == 'X':
        cmp = 'u'
    elif xis == 'y' or xis == 'Y':
        cmp = 'v'
    elif xis == 'z' or xis == 'Z':
        cmp = 'w'
    return cmp