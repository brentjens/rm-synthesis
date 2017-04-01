import logging
try:
    from itertools import izip
except ImportError:
    izip = zip

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
    logging.info('P header:\n%r', p_out_hdr)
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
        mask = 1.0 - array([channel in ignore for ignore in ignore_frames])
        phasors = exp(-2.j*array(rm_to_remove)*wl2)*mask
        corr_p_frames = (q_frames +1j*u_frames)*phasors[:, newaxis, newaxis]
        if mask.sum() == 0.0:
            p_final = corr_p_frames.sum(axis=0)*0.0
        else:
            p_final = corr_p_frames.sum(axis=0)/float(mask.sum())
        weights.append(mask.sum())
        p_out.write(array(abs(p_final), dtype=float32))
        q_out.write(array(p_final.real, dtype=float32))
        u_out.write(array(p_final.imag, dtype=float32))
    p_out.close()
    q_out.close()
    u_out.close()
    with open(weights_name, 'w') as w_out:
        w_out.write('\n'.join([str(w) for w in weights]))
    logging.info('Done correcting')





def average_psf_cubes(psf_out_name,
                      weights_name,
                      psf_input_names,
                      force_overwrite=False,
                      ignore_frames=None):
    r'''
    TBD
    '''
    logging.info('correct_and_average_cubes()')
    psf_header = fits.get_header(psf_input_names[0])
    psf_iterators = [fits.image_frames(name) for name in psf_input_names]

    if ignore_frames is None:
        ignore_frames = [[]]*len(qu_input_names)

    psf_out_hdr = psf_header
    logging.info('PSF header:\n%r', psf_out_hdr)
    psf_out = fits.streaming_output_hdu(psf_out_name, psf_header, force_overwrite)

    weights = []
    for channel, frames in enumerate(izip(*psf_iterators)):
        logging.info('Channel %r', channel)
        psf_frames = array(frames, dtype=float32)
        mask = 1.0 - array([channel in ignore for ignore in ignore_frames])
        masked_psf_frames = psf_frames*mask[:, newaxis, newaxis]
        if mask.sum() == 0.0:
            psf_final = psf_frames[0,:,:]*0.0
        else:
            psf_final = psf_frames.sum(axis=0)/float(mask.sum())
        weights.append(mask.sum())
        psf_out.write(array(psf_final, dtype=float32))
    psf_out.close()
    with open(weights_name, 'w') as w_out:
        w_out.write('\n'.join([str(w) for w in weights]))
    logging.info('Done averaging')
