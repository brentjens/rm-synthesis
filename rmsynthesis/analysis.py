import logging
from itertools import izip
from numpy import array, exp, zeros, float32, float64, complex64, newaxis
import rmsynthesis.fits as fits
from rmsynthesis.main import wavelength_squared_m2_from_freq_hz, output_pqu_headers

def correct_and_average_cubes(p_out_name, q_out_name, u_out_name,
                              weights_name,
                              qu_input_names,
                              freq_hz,
                              force_overwrite=False,
                              rm_to_remove=None,
                              ignore_frames=None):
    r'''
    TBD
    '''
    logging.info('correct_and_average_cubes()')
    q_header = fits.get_header(qu_input_names[0][0])
    u_header = fits.get_header(qu_input_names[0][1])
    qu_iterators = [[fits.image_frames(name) for name in qu_name]
                    for qu_name in qu_input_names]
    flat_qu_iterators = [item for sub_list in qu_iterators
                         for item in sub_list]
    if rm_to_remove is None:
        rm_to_remove = zeros(len(qu_input_names), dtype=float64)
    if ignore_frames is None:
        ignore_frames = [[]]*len(qu_input_names)

    p_out_hdr, q_out_hdr, u_out_hdr = output_pqu_headers(q_header)
    p_out, q_out, u_out = [fits.streaming_output_hdu(fits_name,
                                                     header,
                                                     force_overwrite)
                           for fits_name, header in [(p_out_name, p_out_hdr),
                                                     (q_out_name, q_out_hdr),
                                                     (u_out_name, u_out_hdr)]]

    weights = []
    for channel, freq_qu_list in enumerate(izip(freq_hz, *flat_qu_iterators)):
        freq = freq_qu_list[0]
        logging.info('Channel %r: %.3f MHz', channel, freq/1e6)
        wl2 = wavelength_squared_m2_from_freq_hz(freq)
        q_frames = array(freq_qu_list[1::2], dtype=float32)
        u_frames = array(freq_qu_list[2::2], dtype=float32)
        mask = array([channel in ignore for ignore in ignore_frames])
        phasors = exp(-2.j*array(rm_to_remove)*wl2)*mask
        corr_p_frames = (q_frames +1j*u_frames)*phasors[:, newaxis, newaxis]
        p_final = corr_p_frames.sum(axis=0)/mask.sum()
        weights.append(mask.sum())
        p_out.write(abs(p_final))
        q_out.write(p_final.real)
        u_out.write(p_final.imag)
    p_out.close()
    q_out.close()
    u_out.close()
    with open(weights_name, 'w') as w_out:
        w_out.write('\n'.join([str(w) for w in weights]))
    logging.info('Done correcting')
