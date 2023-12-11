from utils import read_orca_output, plot_spectrum, PlotOptions
import matplotlib.pyplot as plt  # plots


def smooth_spectrum(comm, path, dir, min_energy, max_energy, min_wavelength, max_wavelength):
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    plt.rcParams.update({"font.size": 22})
    
    # global constants
    found_uv_section = False  # check for uv data in out
    specstring_start = 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS'  # check orca.out from here
    
    ORCA_METHOD = "EOM-CCSD"
    
    if ORCA_METHOD == "TD-DFT":
        specstring_end = 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS'  # stop reading orca.out from here
    elif ORCA_METHOD == "EOM-CCSD":
        specstring_end = "CD SPECTRUM" # stop reading orca.out from here
    
    w_wn = 1000  # w = line width for broadening - wave numbers, FWHM
    w_nm = 10  # w = line width for broadening - nm, FWHM
    export_delim = " "  # delimiter for data export
    
    # plot config section - configure here
    nm_plot = True  # wavelength plot /nm if True, if False wave number plot /cm-1
    show_single_gauss = True  # show single gauss functions if True
    show_single_gauss_area = True  # show single gauss functions - area plot if True
    show_conv_spectrum = True  # show the convoluted spectra if True (if False peak labels will not be shown)
    show_sticks = True  # show the stick spectra if True
    label_peaks = True  # show peak labels if True
    minor_ticks = True  # show minor ticks if True
    show_grid = False  # show grid if True
    linear_locator = False  # tick locations at the beginning and end of the spectrum x-axis, evenly spaced
    spectrum_title = "Absorption spectrum"  # title
    spectrum_title_weight = "bold"  # weight of the title font: 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight'
    y_label = "intensity"  # label of y-axis
    x_label_eV = r"energy (eV)"  # label of the x-axis - eV
    x_label_nm = r"wavelength (nm)"  # label of the x-axis - nm
    plt_y_lim = 0.4
    figure_dpi = 100  # DPI of the picture

    PlotOptions_object = PlotOptions(nm_plot,
                                 show_single_gauss,
                                 show_single_gauss_area,
                                 show_conv_spectrum,
                                 show_sticks,
                                 label_peaks,
                                 x_label_nm,
                                 x_label_eV,
                                 y_label,
                                 plt_y_lim,
                                 minor_ticks,
                                 linear_locator,
                                 spectrum_title_weight,
                                 show_grid,
                                 0, # show_spectrum,
                                 1, #save_spectrum,
                                 1, #export_spectrum,
                                 figure_dpi,
                                 export_delim,
                                 ORCA_METHOD)

    spectrum_discretization_step = 0.02
    xmin_spectrum = 0.0  # could be min_wavelength
    xmax_spectrum = 750  # could be max_wavelength

    spectrum_file = path + '/' + dir + '/' + "orca.stdout"

    # open a file
    # check existence
    try:
        statelist, energylist, intenslist = read_orca_output(spectrum_file, specstring_start, specstring_end)

    # file not found -> exit here
    except IOError:
        print(f"'{spectrum_file}'" + " not found", flush=True)
        return
    except Exception as e:
        print("Rank: ", comm_rank, " encountered Exception: ", e, e.args)
        return

    if nm_plot:
        # convert wave number to nm for nm plot
        energylist = [1 / wn * 10 ** 7 for wn in energylist]
        w = w_nm  # use line width for nm axis
    else:
        w = w_wn  # use line width for wave number axis

    # convert wave number to nm for nm plot
    valuelist = energylist
    valuelist.sort()
    w = w_nm  # use line width for nm axis

    plot_spectrum(comm, path, dir, spectrum_file, xmin_spectrum, xmax_spectrum, spectrum_discretization_step, valuelist, w, intenslist, PlotOptions_object)

