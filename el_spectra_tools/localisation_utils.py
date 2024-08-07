# sys.path.append("..")  # Makes genutils package visible
import numpy as np
from ab_initio_postprocessing.abInitio_io_parse import gau_parse
from ab_initio_postprocessing.graph_utils import my_graphs
from ab_initio_postprocessing.graph_utils.formatting import set_subplots_pos_cm
import matplotlib.pyplot as plt
from cclib.io import ccread
import os


def draw_dos_ipr(list_of_gau, span=(-5.5, 5), ax_per_fig=3, spectra_dir='spectra'):

    if isinstance(list_of_gau, str):
        with open(list_of_gau) as list_fid:
            list_of_gau = [x for x in list_fid]

    input_list = [x.strip().split(':') for x in list_of_gau]
    files = [x[0] for x in input_list]
    labels = [x[1] for x in input_list]

    plt.rcParams['xtick.major.pad'] = '2'
    plt.rcParams['ytick.major.pad'] = '1'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    full_spectra_path = os.getcwd() + '/' + spectra_dir
    if not os.path.exists(full_spectra_path):
        os.makedirs(full_spectra_path)

    for no, file in enumerate(files):
        curr_ax_no = (no + 1) % ax_per_fig
        curr_fig_no = no // ax_per_fig
        if not os.path.isfile(file):
            continue
        else:
            if curr_ax_no == 1:
                fig, axs = plt.subplots(ax_per_fig, 1, sharex=True, sharey=True, num=curr_fig_no)
                plt.setp(axs, yticks=np.array([-0.5, 0, 0.5]))

                set_subplots_pos_cm([0.7, 0.7, 5.8, 4.5], figsize_cm=(8.5, 5.5), subpl_shape=(1, ax_per_fig), fig=fig)
                fig.subplots_adjust(hspace=0)
                plt.setp(axs, yticks=[0])
                # plt.tight_layout(pad=0.0)

        print(file)
        gaudata = ccread(file)
        up_states, dn_states, fermi = gau_parse.get_kohn_sham(gaudata)
        ipr = gau_parse.get_ipr(gaudata)
        n_atoms = gaudata.natom
        # ipr_redef = 1 - 1 / (ipr * n_atoms)
        clname = gau_parse.get_formula(gaudata)
        label = labels[no]
        mo_nos = np.arange(gaudata.nmo)[(up_states - fermi > span[0]) & (up_states - fermi < span[1])] - gaudata.homos[
            0]
        mo_ens = up_states[(up_states - fermi > span[0]) & (up_states - fermi < span[1])] - fermi
        towrite = np.array([mo_nos, mo_ens]).transpose()
        with open(full_spectra_path + '/' + label, 'w') as specfile:
            for level in towrite:
                specfile.write('%3d\t%6.3f\n' % (level[0], level[1]))
        # np.save()

        try:
            curr_shareax = axs[curr_ax_no - 1]
        except TypeError:
            curr_shareax = axs

        my_graphs.draw_spectrum(span, up_states, specdn=None, e_fermi=fermi, sigma=0.1, units='eV',
                                curr_fig_no=curr_fig_no, shareax=curr_shareax,  # total_axes=ax_per_fig, curr_ax_no=curr_ax_no,
                                dos_area_fill=True, dos_area_kwargs={'color': 'k', 'alpha': 0.2, 'lw': 0},
                                dos_line_kwargs=None,
                                pdosup_area_kwargs=None, pdosup_clr='k',
                                label=label, label_kwargs={'x': 0.1, 'y': 0.7, 'size': 8})
        # (span, up_states, specdn=None, e_fermi=fermi, sigma=0.03, units='ev',
        #                         curr_fig_no=curr_fig_no, total_axes=n_ax, curr_ax_no=curr_ax_no, label=label,
        #                         shareax=curr_shareax, ticksarg={})

        _, ipr_handle = my_graphs.draw_ipr(span, up_states, ipr, eaxdn=None, eFermi=fermi, units='ev',
                                           currFigNo=curr_fig_no, totalAxs=ax_per_fig, currAxNo=curr_ax_no,
                                           shareax=curr_shareax)

        if curr_ax_no + 1 == ax_per_fig:
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.ylabel('DOS, a.u.', labelpad=-8, fontsize=10)
            # plt.ylabel('test', position='r')
            plt.xlabel(r'$\epsilon_i-\epsilon_{\mathrm{F}}$, eV', labelpad=-2, fontsize=10)
            plt.text(1.01, 0.5, r"IPR", {'color': 'r', 'fontsize': 8},
                     horizontalalignment='left',
                     verticalalignment='center',
                     rotation=90,
                     clip_on=False, )
            # plt.legend(ipr_handle, 'IPR')
            # transform=plt.gca().transAxes)
            plt.savefig('fig01_' + clname + '.pdf')
    plt.show()
