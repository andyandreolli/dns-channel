import numpy as np
import h5py
from matplotlib.pyplot import subplots, imshow, colorbar, savefig
from matplotlib.colors import LogNorm





def plot_premultiplied(all_spectra, components, desired_ys, y, kx, kz, **kwargs):

    # unpack input
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
        idx = get_comp_idx(component)
        
        # select desired spectrum
        spectrum = all_spectra[idx, :, :, :]

        # calculate premultiplier using array broadcasting
        bcast_premult = kz.reshape((-1, 1)) * kx
        bcast_premult = abs(bcast_premult)

        # premultiply
        premultiplied = spectrum * bcast_premult

        for desired_y in desired_ys:

            y_idx = (np.abs(y - desired_y)).argmin()

            # actually print
            fig, (ax,cb_ax) = subplots(ncols=2,figsize=(10,7),gridspec_kw={"width_ratios":[1, 0.05]})
            ax.set_xscale("log")
            ax.set_yscale("log")
            # when plotting, 0 modes are excluded
            pos = ax.pcolormesh(kx[1:], kz[1:], premultiplied[y_idx, 1:, 1:], linewidth=0, rasterized=True,shading='gouraud',cmap='hot_r')
            pos.set_edgecolor('face')
            ax.set_xlabel(r'$k_x$')
            ax.set_ylabel(r'$k_z$')
            secaxx = ax.secondary_xaxis('top', functions=(get_wavelength_fwr,get_wavelength_inv))
            secaxx.set_xlabel(r'$\lambda_x$')
            secaxy = ax.secondary_yaxis('right', functions=(get_wavelength_fwr,get_wavelength_inv))
            secaxy.set_ylabel(r'$\lambda_z$')
            fig.colorbar(pos, cax=cb_ax)
            cmp = gcmp(component)
            ax.set_title(plot_title + r'$k_xk_z \langle \hat{'+cmp[0]+r'}^\dagger \hat{'+cmp[1]+r'} \rangle$, ' + 'y = {}'.format(round(y[y_idx], 3)))
            if save_fig:
                figname = 'Figs/' + plot_title[:-2].lower() + '_' + 'premultiplied' + '_' + component + '_y' + str(round(y[y_idx], 3)).replace('.','') + '.' + save_format
                savefig(figname, format=save_format)





def plot(all_spectra, components, desired_ys, y, kx, kz, **kwargs):

    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
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

    # unpack input
    if not (hasattr(components, '__iter__') or hasattr(components, '__getitem__')):
        components = [components]
    if not (hasattr(desired_ys, '__iter__') or hasattr(desired_ys, '__getitem__')):
        desired_ys = [desired_ys]

    for component in components:
        
        # convert component to index
        idx = get_comp_idx(component)
        
        # select desired spectrum
        spectrum = all_spectra[idx, :, :, :]

        for desired_y in desired_ys:

            y_idx = (np.abs(y - desired_y)).argmin()

            # actually print
            fig, (ax,cb_ax) = subplots(ncols=2,figsize=(10,7),gridspec_kw={"width_ratios":[1, 0.05]})
            ax.set_xlabel(r'$k_x$')
            ax.set_ylabel(r'$k_z$')
            pos = ax.pcolormesh(kx,kz,spectrum[y_idx,:,:], vmin=vmin, vmax=vmax, linewidth=0, rasterized=True,shading='gouraud',cmap='hot_r')
            pos.set_edgecolor('face')
            secaxx = ax.secondary_xaxis('top', functions=(get_wavelength_fwr,get_wavelength_inv))
            secaxx.set_xlabel(r'$\lambda_x$')
            secaxy = ax.secondary_yaxis('right', functions=(get_wavelength_fwr,get_wavelength_inv))
            secaxy.set_ylabel(r'$\lambda_z$')
            fig.colorbar(pos,cax=cb_ax)
            cmp = gcmp(component)
            ax.set_title(plot_title + r'$ \tilde{'+cmp[0]+r'}^\dagger \tilde{'+cmp[1]+'}$, ' + 'y = {}'.format(round(y[y_idx], 3)))
            if save_fig:
                figname = 'Figs/' + plot_title[:-2].lower() + '_' + component + '_y' + str(round(y[y_idx], 3)).replace('.','') + '.' + save_format
                savefig(figname, format=save_format)





def plot_cumulative_zy(all_spectra, components, y, kz, **kwargs):

    # unpack input
    plot_title = kwargs.get('title', '')
    save_fig = kwargs.get('save_fig', '')
    y_symm = kwargs.get('y_symm', True)
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
        idx = get_comp_idx(component)
        
        # select desired spectrum component
        spectrum = all_spectra[idx, :, :, :]

        # sum along x-axis
        spectrum = spectrum.sum(axis=(-1))

        # premultiply (reshape kz for broadcasting)
        premultiplied = spectrum * abs(kz)

        # actually print
        fig, (ax,cb_ax) = subplots(ncols=2,figsize=(10,7),gridspec_kw={"width_ratios":[1, 0.05]})
        ax.set_xscale("log") # logarithmic scale only for lambda z
        # when plotting, 0 modes are excluded
        pos = ax.pcolormesh(kz[1:], y, premultiplied[:,1:], linewidth=0, rasterized=True,shading='gouraud',cmap='hot_r')
        pos.set_edgecolor('face')
        ax.set_xlabel(r'$k_z$') 
        ax.set_ylabel(r'$y$')
        secaxx = ax.secondary_xaxis('top', functions=(get_wavelength_fwr,get_wavelength_inv))
        secaxx.set_xlabel(r'$\lambda_z$')
        fig.colorbar(pos, cax=cb_ax)
        cmp = gcmp(component)
        ax.set_title(plot_title + r'$k_z\sum\,_{k_x} \langle \hat{'+cmp[0]+r'}^\dagger\hat{'+cmp[1]+r'} \rangle(k_x, k_z, y)$')
        if y_symm:
            ax.set_ylim([0,1])
        if save_fig:
            figname = 'Figs/' + plot_title[:-2].lower() + '_cumulative_' + component + '.' + save_format
            savefig(figname, format=save_format)





def plot_cumulative_xy(all_spectra, components, y, kx, **kwargs):

    # unpack input
    plot_title = kwargs.get('title', '')
    save_fig = kwargs.get('save_fig', '')
    y_symm = kwargs.get('y_symm', True)
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
        idx = get_comp_idx(component)
        
        # select desired spectrum component
        spectrum = all_spectra[idx, :, :, :]

        # sum along z-axis
        spectrum = spectrum.sum(axis=(-2))

        # premultiply (reshape kz for broadcasting)
        premultiplied = spectrum * abs(kx)

        # actually print
        fig, (ax,cb_ax) = subplots(ncols=2,figsize=(10,7),gridspec_kw={"width_ratios":[1, 0.05]})
        ax.set_xscale("log") # logarithmic scale only for lambda z
        # when plotting, 0 modes are excluded
        pos = ax.pcolormesh(kx[1:], y, premultiplied[:,1:], linewidth=0, rasterized=True,shading='gouraud',cmap='hot_r')
        pos.set_edgecolor('face')
        ax.set_xlabel(r'$k_z$') 
        ax.set_ylabel(r'$y$')
        secaxx = ax.secondary_xaxis('top', functions=(get_wavelength_fwr,get_wavelength_inv))
        secaxx.set_xlabel(r'$\lambda_x$')
        fig.colorbar(pos,cax=cb_ax)
        cmp = gcmp(component)
        ax.set_title(plot_title + r'$k_z\sum\,_{k_x} \langle \hat{'+cmp[0]+r'}^\dagger\hat{'+cmp[1]+r'} \rangle(k_x, k_z, y)$')
        if y_symm:
            ax.set_ylim([0,1])
        if save_fig:
            figname = 'Figs/' + plot_title[:-2].lower() + '_cumulative_' + component + '.' + save_format
            savefig(figname, format=save_format)





def gcmp(component):
    label = 'uu'
    component = component.lower()
    if component == 'y' or component == 'vv' or component == 'v':
        label = 'vv'
    elif component == 'z' or component == 'ww' or component == 'w':
        label = 'ww'
    elif component == 'xy' or component == 'yx' or component == 'uv' or component == 'vu':
        label = 'uv'
    elif component == 'xz' or component == 'zx' or component == 'uw' or component == 'wu':
        label = 'uw'
    elif component == 'zy' or component == 'yz' or component == 'wv' or component == 'vw':
        label = 'vw'
    elif not component == 'x' or component == 'uu' or component == 'u':
        print('Error: invalid component. Please pass either "x", "y", "z" as the second argument.')
    return label





def get_comp_idx(component):
    idx = 0
    component = component.lower()
    if component == 'y' or component == 'vv' or component == 'v':
        idx = 1
    elif component == 'z' or component == 'ww' or component == 'w':
        idx = 2
    elif component == 'xy' or component == 'yx' or component == 'uv' or component == 'vu':
        idx = 3
    elif component == 'xz' or component == 'zx' or component == 'uw' or component == 'wu':
        idx = 4
    elif component == 'zy' or component == 'yz' or component == 'wv' or component == 'vw':
        idx = 5
    elif not component == 'x' or component == 'uu' or component == 'u':
        print('Error: invalid component. Please pass either "x", "y", "z" as the second argument.')
    return idx



def get_wavelength_fwr(x):
    return 2*np.pi/x

def get_wavelength_inv(x):
    return 2*np.pi/x