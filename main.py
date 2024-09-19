import numpy as np
import time; import datetime
import argparse

import data
from cross_sec import CrossSection
from figures import Figures

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_file', type=str, #required=True, 
        help='Name of input file (e.g. input.example_ExoMol)', 
        )

    parser.add_argument('--download', '-d', action='store_true')
    parser.add_argument('--cross_sections', '-cs', action='store_true')
    parser.add_argument('--save', '-s', action='store_true')
    parser.add_argument('--plot', action='store_true')

    parser.add_argument('--convert_to_pRT2', action='store_true')

    # Index to read if multiple .trans files are given
    parser.add_argument('--trans_idx_min', '-i_min', default=0, type=int)
    parser.add_argument('--trans_idx_max', '-i_max', default=1, type=int)
    parser.add_argument('--show_pbar', action='store_true')
    
    # Optionally, overwrite the P and T values of conf
    parser.add_argument('--P', type=float, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--tmp_output_file', '-out', type=str, default='cross{}.hdf5')
    args = parser.parse_args()

    # Import input file as 'conf'
    input_string = str(args.input_file).replace('.py', '').replace('/', '.')
    conf = __import__(input_string, fromlist=[''])

    # Download from the ExoMol/HITEMP database
    if args.download:
        if conf.database.lower() == 'exomol':
            data.ExoMol.download_data(conf)
        elif conf.database.lower() in ['hitemp', 'hitran']:
            data.HITEMP.download_data(conf)
        elif conf.database.lower() in ['vald', 'kurucz']:
            data.VALD_Kurucz.download_data(conf)

        import sys; sys.exit()
            
    # Load data
    D = data.load_data(conf)
    trans_file      = conf.files['transitions']
    tmp_output_dir  = conf.files['tmp_output_dir']
    tmp_output_file = f'{tmp_output_dir}/{args.tmp_output_file}'

    N_trans = 1
    if isinstance(trans_file, (list, tuple)):
        N_trans = len(trans_file)
    else:
        trans_file = [trans_file] # Insert into list

    if args.cross_sections:

        show_pbar = True
        if N_trans > 1:
            show_pbar = False
        if args.show_pbar:
            show_pbar = True

        d_idx = 1
        if args.trans_idx_max < args.trans_idx_min:
            d_idx = -1
        trans_idx = np.arange(args.trans_idx_min, args.trans_idx_max, d_idx)

        # Update tmp file to avoid overwriting
        if args.P is not None:
            conf.P_grid = np.array([args.P])
            tmp_output_file = tmp_output_file.replace('.hdf5', f'_P{args.P}.hdf5')
        if args.T is not None:
            conf.T_grid = np.array([args.T])
            tmp_output_file = tmp_output_file.replace('.hdf5', f'_T{args.T}.hdf5')
                
        time_start = time.time()
        for i in trans_idx:
            time_start_i = time.time()
            # Read 1 .trans file at a time
            trans_file_i = trans_file[i]
            # If 'output' is an fstring
            tmp_output_file = tmp_output_file.format(i)

            # Compute + save cross-sections
            CS = CrossSection(conf, Q=D.Q, mass=D.mass)
            CS = D.get_cross_sections(CS, trans_file_i, show_pbar=show_pbar)
            CS.save_cross_sections(tmp_output_file)
            
            time_finish_i = time.time()
            time_elapsed_i = time_finish_i - time_start_i
            if not show_pbar:
                print('Time elapsed: {}'.format(str(datetime.timedelta(seconds=time_elapsed_i))))
        
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        if not show_pbar:
            print('Time elapsed (total): {}'.format(str(datetime.timedelta(seconds=time_elapsed))))

    if args.save:
        # (Possibly) combine cross-sections from different .trans files
        D.combine_cross_sections(tmp_output_dir)
        
    if args.plot:
        F = Figures(
            D, wave_range=[(1/3,50.0), (1.05,1.35), (1.9,2.5), (2.29,2.4), (2.332,2.339)]
            )
        F.plot_P(
            T=1000, P=10**np.array([-4,-2,0,2], dtype=np.float64), 
            ylim=(1e-28,1e-16), save_file=f'{conf.cross_sec_outputs}/plots/P.pdf'
            )
        F.plot_T(
            P=1, T=np.array([500,1000,1500,2000,2500]), 
            ylim=(1e-28,1e-16), save_file=f'{conf.cross_sec_outputs}/plots/T.pdf'
            )

    if args.convert_to_pRT2:
        D.convert_to_pRT2_format(
            out_dir=conf.pRT['out_dir'], 
            pRT_wave_file=conf.pRT['wave'], 
            make_short_file=conf.pRT['make_short_file']
        )