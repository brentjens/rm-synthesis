r'''
The ``main`` module implements the actual RM-synthesis.
'''

from numpy import exp, newaxis, array, zeros, floor   #pylint: disable=E0611
from numpy import concatenate, real, imag, frombuffer #pylint: disable=E0611
from numpy import complex64, array_split, absolute    #pylint: disable=E0611
from numpy import float32                             #pylint: disable=E0611

import multiprocessing as mp
import gc, os, sys, ctypes, logging

try:
    from itertools import izip
except ImportError:
    def izip(*args):
        'itertools.izip replacement for older python versions.'
        return zip(*args)
    izip.__doc__ = zip.__doc__

import rmsynthesis.fits as fits

__version__ = '1.0-rc4'
RMSYNTHESIS_VERSION = __version__


class ParseError(RuntimeError):
    r'''
    Raised whenever there is a problem with parsing a file.
    '''
    pass

class ShapeError(RuntimeError):
    r'''
    Raised whenever shapes of FITS cubes are incompatible.
    '''
    pass



def file_exists(filename, verbose=False):
    """
    Returns True if *filename* exists, False if it does not. If
    *verbose* is True, it also prints an error message if the file
    does not exist.
    """
    try:
        os.stat(filename)
        return True
    except (OSError,):
        err = sys.exc_info()[1]
        if verbose:
            logging.error(str(err))
        return False


def almost_equal(number_x, number_y, epsilon=1e-9):
    r'''
    Returns True if

    .. math::
       \frac{|x-y|}{\mathrm{max}(|x|, |y|)} < \epsilon


    If either of the numbers is exactly zero, it returns True if the
    other number's absolute value is below :math:``\epsilon``.

    **Parameters**

    number_x : float or complex
        One of the numbers to compare.

    number_y : float or complex
        The other number to compare.

    epsilon : float
        Fractional accuracy to which the numbers must be equal.

    **Returns**

    A boolean indicating if ``number_x`` and ``number_y`` are almost equal.

    **Examples**

    First the cases when one of the arguments is exactly 0:

    >>> almost_equal(0.0, -1e-8, epsilon = 1e-8)
    False
    >>> almost_equal(0.0, -1e-8+1e-12, epsilon = 1e-8)
    True
    >>> almost_equal(-1e-9, 0.0)
    False
    >>> almost_equal(-1e-9+1e-12, 0.0)
    True
    '''

    if number_x == 0.0:
        return abs(number_y) < epsilon
    if number_y == 0.0:
        return abs(number_x) < epsilon
    return abs(number_x - number_y)/max(abs(number_x), abs(number_y)) < epsilon




def wavelength_squared_m2_from_freq_hz(freq_hz): #pylint: disable=invalid-name
    r'''
    Convert *freq_hz* (in Hz) to wavelength squared (in
    m^2).

    **Parameters**

    freq_hz : scalar or numpy.array
        The frequencies for which :math:``\lambda^2`` must be computed.

    **Returns**

    A scalar or numpy.array with the same shape as *freq_hz*.

    **Examples**

    >>> c = 299792458.0
    >>> wavelength_squared_m2_from_freq_hz(c)
    1.0
    >>> wavelength_squared_m2_from_freq_hz(array([c, c/2.0, c/3.0]))
    array([1., 4., 9.])
    >>> wavelength_squared_m2_from_freq_hz(array([[c], [c/2.0], [c/3.0]]))
    array([[1.],
           [4.],
           [9.]])
    >>> wavelength_squared_m2_from_freq_hz(array([]))
    array([], dtype=float64)

    '''
    return (299792458.0/freq_hz)**2




def phases_lambda2_to_phi(wavelength_squared_m2, phi_rad_m2):
    r'''
    Computes the phase factor

    .. math::
      \mathrm{e}^{-2\mathrm{i}\phi\lambda^2},

    necessary in the transform from wavelength squared space to
    Faraday depth space.

    **Parameters**

    wavelength_squared_m2 : scalar or numpy.array
        Wavelength squared in :math:``m^2``

    phi_rad_m2 : scalar or numpy.array
        Faraday depth in :math:``\mathrm{rad m}^{-2}``

    **Examples**

    >>> from math import pi
    >>> almost_equal(phases_lambda2_to_phi(1.0, pi), 1.0)
    True
    >>> almost_equal(phases_lambda2_to_phi(pi/2, 0.5), -1.j)
    True
    >>> list(map(almost_equal,
    ...      phases_lambda2_to_phi(array([pi, 0.5*pi]), 0.5), [-1., -1.j]))
    [True, True]
    >>> list(map(almost_equal,
    ...          phases_lambda2_to_phi(pi, array([0.5, -0.25])), [-1., +1.j]))
    [True, True]

    '''
    return exp(-2j*phi_rad_m2*wavelength_squared_m2)



def phases_phi_to_lambda2(wavelength_squared_m2, phi_rad_m2):
    r'''
    Computes the phase factor

    .. math::
      \mathrm{e}^{+2\mathrm{i}\phi\lambda^2},

    necessary in the transform from Faraday depth to wavelength
    squared space.

    **Parameters**

    wavelength_squared_m2 : scalar or numpy.array
        Wavelength squared in :math:``m^2``

    phi_rad_m2 : scalar or numpy.array
        Faraday depth in :math:``\mathrm{rad m}^{-2}``

    **Examples**

    >>>
    '''
    return exp(+2j*phi_rad_m2*wavelength_squared_m2)




def parse_frequency_file(filename):
    r'''
    Read a text file containing one floating point number per line,
    and return an array of those values. Empty lines are
    ignored. Comments can be included in the file behind a # mark.
    The frequencies should be listed in Hz.

    Raises a printable ParseError in case of problems with the
    contents of the file.
    '''
    try:
        return array([float(x.split('#')[0].strip())
                         for x in open(filename).readlines()
                         if x.split('#')[0].strip() != ''])
    except (ValueError,):        # Use this construction to be backwards
        err = sys.exc_info()[1]  # compatible with Python 2.5.  Proper
                                 # Python 2.6/2.7/3.0 is
                                 # except ValueError as err:
                                 # etc...
        raise ParseError('while parsing '+filename+': '+str(err))




def proper_fits_shapes(qname, uname, frequencyname):
    """
    Verify that the Q and U FITS cubes and the file with frequency
    data have compatible shapes. *qname* and *uname* are the filenames
    of the Q and U FITS files, respectively; *frequencyname* is the
    filename of the frequency file.

    Returns True if all is well, raises ShapeError otherwise.
    """
    frequencies = parse_frequency_file(frequencyname)
    q_h = fits.get_header(qname)
    u_h = fits.get_header(uname)
    errors = []
    for name, hdr in [(qname, q_h), (uname, u_h)]:
        if hdr['NAXIS'] != 3:
            errors.append('number of axes in %s  is  %d, not 3' %
                          (name, hdr['NAXIS']))

    for axis in ['NAXIS1', 'NAXIS2', 'NAXIS3']:
        if q_h[axis] != u_h[axis]:
            errors.append('%s in %s (%d) not equal to %s in %s (%d)' %
                          (axis, qname, q_h[axis], axis, uname, u_h[axis]))

    if q_h['NAXIS3'] != len(frequencies):
        msg = ('frames in %r and %r (%d) not the same as in %r (%d)' %
               (qname, uname, q_h['NAXIS3'], frequencyname, len(frequencies)))
        errors.append(msg)
    if len(errors) > 0:
        raise ShapeError('\n'.join(errors))
    return True




def rmsynthesis_dirty(qcube, ucube, frequencies, phi_array):
    """
    Perform an RM synthesis on Q and U image cubes with given
    frequencies. The rmcube that is returned is complex valued and has
    a frame for every value in phi_array. It is assumed that the
    dimensions of qcube, ucube, and frequencies have been verified
    before with the help of the proper_fits_shapes() function. The
    polarization vectors are derotated to the average lambda^2.
    """
    wl2 = wavelength_squared_m2_from_freq_hz(frequencies)
    rmcube = zeros((len(phi_array), qcube.shape[1], qcube.shape[2]),
                   dtype=complex64)
    wl2_0 = wl2.mean()
    p_complex = qcube+1j*ucube

    num = len(phi_array)
    nfreq = len(frequencies)
    for i, phi in enumerate(phi_array):
        logging.info('processing frame %4d/%d, phi = %7.1f' % (i+1, num, phi))
        phases = phases_lambda2_to_phi(wl2-wl2_0, phi)[:, newaxis, newaxis]
        rmcube[i, :, :] = (p_complex*phases).sum(axis=0)/nfreq
    return rmcube



def rmsynthesis_dirty_lowmem(qname, uname, q_factor, u_factor,
                             frequencies, phi_array,
                             bad_frames=None):
    """
    Perform an RM synthesis on Q and U image cubes with given
    frequencies. The rmcube that is returned is complex valued and has
    a frame for every value in phi_array. It is assumed that the
    dimensions of qcube, ucube, and frequencies have been verified
    before with the help of the proper_fits_shapes() function. The
    polarization vectors are derotated to the average lambda^2.
    """
    wl2 = wavelength_squared_m2_from_freq_hz(frequencies)
    qheader = fits.get_header(qname)
    rmcube = zeros((len(phi_array), qheader['NAXIS2'], qheader['NAXIS1']),
                   dtype=complex64)
    wl2_0 = wl2.mean()

    nfreq = len(frequencies)
    q_frames = fits.image_frames(qname)
    u_frames = fits.image_frames(uname)
    frame_id = 0
    skipped_frames = 0
    wl2_norm = wl2 - wl2_0
    for q_frame, u_frame in izip(q_frames, u_frames):
        if bad_frames is not None and frame_id in bad_frames:
            logging.warn('skipping frame % d: in bad frame list.', frame_id)
            skipped_frames += 1
        else:
            logging.info('processing frame '+str(frame_id+1)+'/'+str(nfreq))
            p_complex = q_frame*q_factor + 1.0j*u_frame*u_factor
            wl2_frame = wl2_norm[frame_id]
            phases = phases_lambda2_to_phi(wl2_frame, phi_array)
            for frame, phase in enumerate(phases):
                rmcube[frame, :, :] += p_complex*phase
        gc.collect()
        frame_id += 1
    frames_added = nfreq - skipped_frames
    return rmcube/float(frames_added)





def rmsynthesis_crosscorr_dirty_lowmem(q_template_name, u_template_name,
                                       qname, uname, q_factor, u_factor,
                                       frequencies, phi_array,
                                       bad_frames=None):
    """Perform a cross correlation in Faraday space by multiplying QU
    frames with template QU frames, and performing an RM synthesis on
    the resulting Q and U image cubes with given frequencies. The
    rmcube that is returned is complex valued and has a frame for
    every value in phi_array. It is assumed that the dimensions of
    qcube, ucube, and frequencies have been verified before with the
    help of the proper_fits_shapes() function. The polarization
    vectors are derotated to the average lambda^2.
    """
    wl2 = wavelength_squared_m2_from_freq_hz(frequencies)
    qheader = fits.get_header(qname)
    rmcube = zeros((len(phi_array), qheader['NAXIS2'], qheader['NAXIS1']),
                   dtype=complex64)
    wl2_0 = wl2.mean()

    nfreq = len(frequencies)
    q_template_frames = fits.image_frames(q_template_name)
    u_template_frames = fits.image_frames(u_template_name)
    q_frames = fits.image_frames(qname)
    u_frames = fits.image_frames(uname)
    frame_id = 0
    skipped_frames = 0
    wl2_norm = wl2 - wl2_0
    for q_temp, u_temp, q_frame, u_frame in izip(q_template_frames, u_template_frames, q_frames, u_frames):
        if bad_frames is not None and frame_id in bad_frames:
            logging.warn('skipping frame % d: in bad frame list.', frame_id)
            skipped_frames += 1
        else:
            logging.info('processing frame '+str(frame_id+1)+'/'+str(nfreq))
            template_complex = q_temp*q_factor +1.0j*u_temp*u_factor
            p_complex = (q_frame*q_factor + 1.0j*u_frame*u_factor)*template_complex.conj()
            wl2_frame = wl2_norm[frame_id]
            phases = phases_lambda2_to_phi(wl2_frame, phi_array)
            for frame, phase in enumerate(phases):
                rmcube[frame, :, :] += p_complex*phase
        gc.collect()
        frame_id += 1
    frames_added = nfreq - skipped_frames
    return rmcube/float(frames_added)




def rmsynthesis_worker(queue, shared_arr, frame_shape, phi_array):
    r'''
    '''
    rmcube = zeros((len(phi_array),)+frame_shape,
                      dtype=complex64)
    p_complex = frombuffer(shared_arr.get_obj(),
                           dtype=complex64).reshape(frame_shape)

    while True:
        item = queue.get()
        if item is None:
            queue.put(rmcube)
            break
        else:
            wl2_frame = item
            phases = phases_lambda2_to_phi(wl2_frame, phi_array)
            for frame, phase in enumerate(phases):
                rmcube[frame, :, :] += p_complex*phase
            gc.collect()
            queue.task_done()


def rmsynthesis_dirty_lowmem_mp(qname, uname, q_factor, u_factor,
                                frequencies, phi_array):
    """
    Perform an RM synthesis on Q and U image cubes with given
    frequencies. The rmcube that is returned is complex valued and has
    a frame for every value in phi_array. It is assumed that the
    dimensions of qcube, ucube, and frequencies have been verified
    before with the help of the proper_fits_shapes() function. The
    polarization vectors are derotated to the average lambda^2.

    Uses the multiprocessing module to speed up the calculations.
    """
    num_workers = mp.cpu_count()-1

    wl2 = wavelength_squared_m2_from_freq_hz(frequencies)
    qheader = fits.get_header(qname)
    wl2_0 = wl2.mean()

    nfreq = len(frequencies)
    q_frames = fits.image_frames(qname)
    u_frames = fits.image_frames(uname)
    frame_id = 0
    wl2_norm = wl2 - wl2_0

    phi_arrays = array_split(phi_array, num_workers)
    frame_shape = (qheader['NAXIS2'], qheader['NAXIS1'])
    mp_shared_p_complex = mp.Array(ctypes.c_float,
                                   frame_shape[0]*frame_shape[1]*2)
    p_complex = frombuffer(mp_shared_p_complex.get_obj(),
                           dtype=complex64).reshape(frame_shape)

    queues = [mp.JoinableQueue() for _ in range(num_workers)]
    workers = [mp.Process(target=rmsynthesis_worker,
                          args=(queue, mp_shared_p_complex,
                                frame_shape, phi))
                   for queue, phi in zip(queues, phi_arrays)]
    for worker in workers:
        worker.daemon = True
        worker.start()

    for q_frame, u_frame in izip(q_frames, u_frames):
        logging.info('processing frame '+str(frame_id+1)+'/'+str(nfreq))
        p_complex[:, :] = q_frame*q_factor + 1.0j*u_frame*u_factor
        wl2_frame = wl2_norm[frame_id]

        for queue in queues:
            queue.put(wl2_frame)
        for queue in queues:
            queue.join()
        frame_id += 1
        gc.collect()
    for queue in queues:
        queue.put(None)
    logging.info('Collecting partial results')
    partial_rmcubes = [queue.get() for queue in queues]
    rmcube = concatenate(partial_rmcubes)/nfreq
    for worker in workers:
        worker.join()
    gc.collect()
    return rmcube



def compute_rmsf(frequencies, phi_array):
    """
    Compute the Rotation Measure Spread Function, derotating to the
    average lambda^2.
    """
    wl2 = wavelength_squared_m2_from_freq_hz(frequencies)
    wl2_0 = wl2.mean()
    return array([phases_lambda2_to_phi((wl2 - wl2_0), phi).mean()
                     for phi in phi_array])


def add_phi_to_fits_header(fits_header, phi_array):
    """
    Returns a deep copy of *fits_header*. The returned copy has
    Faraday depth as the third axis, with reference values and
    increments as derived from *phi_array*. It is assumed that
    *phi_array* contains values on a regular comb.
    """
    if len(phi_array) < 2:
        raise ShapeError('RM cube should have two or more frames to be a cube')
    fhdr = fits_header.copy()
    fhdr.set('NAXIS3', len(phi_array))
    fhdr.set('CRPIX3', 1.0)
    fhdr.set('CRVAL3', phi_array[0])
    fhdr.set('CDELT3', phi_array[1]-phi_array[0])
    fhdr.set('CTYPE3', 'FARDEPTH')
    fhdr.set('CUNIT3', 'RAD/M^2')
    return fhdr



def write_rmcube(rmcube, fits_header, output_dir, force_overwrite=False):
    r'''
    Writes a complex valued, 3D  *rmcube* to *output_dir* as three
    separate FITS cubes:

    - *output_dir*/p-rmcube-dirty.fits : Absolute value.
    - *output_dir*/q-rmcube-dirty.fits : Real part.
    - *output_dir*/u-rmcube-dirty.fits : Imaginary part.

    This function raises an IOError if *output_dir* does not exist, or
    is not writable, or if the output file(s) already exist and
    *force_overwrite* is False. If *force_overwrite* is True, the
    output files will be overwritten.
    '''
    fhp = fits_header.copy()
    fhp.set('POL', 'P')
    fits.write_cube(os.path.join(output_dir, 'p-rmcube-dirty.fits'),
                    fhp, absolute(rmcube),
                    force_overwrite=force_overwrite)

    fhq = fits_header.copy()
    fhq.set('POL', 'Q')
    fits.write_cube(os.path.join(output_dir, 'q-rmcube-dirty.fits'),
                    fhq, rmcube.real,
                    force_overwrite=force_overwrite)

    fhu = fits_header.copy()
    fhu.set('POL', 'U')
    fits.write_cube(os.path.join(output_dir, 'u-rmcube-dirty.fits'),
                    fhu, rmcube.imag,
                    force_overwrite=force_overwrite)


def write_rmsf(phi, rmsf, output_dir):
    r'''
    '''
    rmsf_out = open(os.path.join(output_dir, 'rmsf.txt'), 'w')
    for phi, f_phi in zip(phi, rmsf):
        rmsf_out.write('%10.4f  %10.4f %10.4f\n' %
                       (phi, real(f_phi), imag(f_phi)))
    rmsf_out.flush()
    rmsf_out.close()


def output_pqu_fits_names(output_dir):
    r'''
    '''
    return tuple([os.path.join(output_dir, pol+'-rmcube-dirty.fits')
                  for pol in 'pqu'])


def output_pqu_headers(fits_header):
    r'''
    '''
    p_hdr = fits_header.copy()
    p_hdr.set('POL', 'P')
    p_hdr.set('BITPIX', -32)

    q_hdr = fits_header.copy()
    q_hdr.set('POL', 'Q')
    q_hdr.set('BITPIX', -32)

    u_hdr = fits_header.copy()
    u_hdr.set('POL', 'U')
    u_hdr.set('BITPIX', -32)

    return  p_hdr, q_hdr, u_hdr



def rmsynthesis_dirty_lowmem_main(q_name, u_name, q_factor, u_factor,
                                  output_dir, freq_hz, phi_rad_m2,
                                  force_overwrite, max_mem_gb=2.0,
                                  bad_frames=None):
    r'''
    '''
    logging.info('rmsynthesis_dirty_lowmem_main()')
    q_header = fits.get_header(q_name)
    pixels_per_frame = q_header['NAXIS1']*q_header['NAXIS2']

    max_mem_bytes = max_mem_gb*1024**3
    bytes_per_output_pixel = 4
    max_output_pixels = max_mem_bytes/bytes_per_output_pixel
    # 7 = (re, im) + p_out + q_out + u_out
    block_length = int(floor(max_output_pixels/pixels_per_frame/5.0))
    num_blocks = int(floor(len(phi_rad_m2)/block_length))
    if len(phi_rad_m2) % block_length > 0:
        num_blocks += 1


    output_header = add_phi_to_fits_header(q_header.copy(), phi_rad_m2)
    p_out_name, q_out_name, u_out_name = output_pqu_fits_names(output_dir)
    p_out_hdr, q_out_hdr, u_out_hdr = output_pqu_headers(output_header)
    p_out, q_out, u_out = [fits.streaming_output_hdu(fits_name,
                                                     header,
                                                     force_overwrite)
                           for fits_name, header in [(p_out_name, p_out_hdr),
                                                     (q_out_name, q_out_hdr),
                                                     (u_out_name, u_out_hdr)]]
    try:
        for block in range(num_blocks):
            phi = phi_rad_m2[block*block_length:(block+1)*block_length]
            logging.info('Processing block %d / %d', block + 1, num_blocks)
            logging.info('Phi in  [%.1f, %.1f]', phi[0], phi[-1])
            rmcube = rmsynthesis_dirty_lowmem(q_name, u_name,
                                              q_factor, u_factor,
                                              freq_hz, phi,
                                              bad_frames=bad_frames)
            logging.info('Saving data')

            p_out.write(absolute(rmcube))
            q_out.write(rmcube.real)
            u_out.write(rmcube.imag)
    finally:
        p_out.close()
        q_out.close()
        u_out.close()
        logging.info('Done')





def rmsynthesis_crosscorr_dirty_lowmem_main(q_template_name, u_template_name,
                                            q_name, u_name,
                                            q_factor, u_factor,
                                            output_dir, freq_hz, phi_rad_m2,
                                            force_overwrite, max_mem_gb=2.0,
                                            bad_frames=None):
    r'''
    '''
    logging.info('rmsynthesis_crosscorr_dirty_lowmem_main()')
    q_header = fits.get_header(q_name)
    pixels_per_frame = q_header['NAXIS1']*q_header['NAXIS2']

    max_mem_bytes = max_mem_gb*1024**3
    bytes_per_output_pixel = 4
    max_output_pixels = max_mem_bytes/bytes_per_output_pixel
    # 7 = (re, im) + p_out + q_out + u_out
    block_length = int(floor(max_output_pixels/pixels_per_frame/9.0))
    num_blocks = int(floor(len(phi_rad_m2)/block_length))
    if len(phi_rad_m2) % block_length > 0:
        num_blocks += 1


    output_header = add_phi_to_fits_header(q_header.copy(), phi_rad_m2)
    p_out_name, q_out_name, u_out_name = output_pqu_fits_names(output_dir)
    p_out_hdr, q_out_hdr, u_out_hdr = output_pqu_headers(output_header)
    p_out, q_out, u_out = [fits.streaming_output_hdu(fits_name,
                                                     header,
                                                     force_overwrite)
                           for fits_name, header in [(p_out_name, p_out_hdr),
                                                     (q_out_name, q_out_hdr),
                                                     (u_out_name, u_out_hdr)]]
    try:
        logging.info('Computing cross correlation with templates\n  - Q:%s\n  - U:%s',
                     q_template_name, u_template_name)
        for block in range(num_blocks):
            phi = phi_rad_m2[block*block_length:(block+1)*block_length]
            logging.info('Processing block %d / %d', block + 1, num_blocks)
            logging.info('Phi in  [%.1f, %.1f]', phi[0], phi[-1])
            rmcube = rmsynthesis_crosscorr_dirty_lowmem(
                q_template_name, u_template_name,
                q_name, u_name,
                q_factor, u_factor,
                freq_hz, phi,
                bad_frames=bad_frames)
            logging.info('Saving data')

            p_out.write(absolute(rmcube))
            q_out.write(rmcube.real)
            u_out.write(rmcube.imag)
    finally:
        p_out.close()
        q_out.close()
        u_out.close()
        logging.info('Done')



def mean_psf(psf_name, frequencies, output_fits_name, force_overwrite,
             max_mem_gb=2.0, bad_frames=None):
    logging.info('mean_psf()')
    psf_header = fits.get_header(psf_name)
    pixels_per_frame = psf_header['NAXIS1']*psf_header['NAXIS2']
    max_mem_bytes = max_mem_gb*1024**3
    bytes_per_input_pixel = psf_header['BITPIX']/8.0
    max_input_pixels = max_mem_bytes/bytes_per_input_pixel
    block_length = int(floor(max_input_pixels/pixels_per_frame/4.0))
    num_blocks = int(floor(psf_header['NAXIS3']/block_length))
    if psf_header['NAXIS3'] % block_length > 0:
        num_blocks += 1
    output_header = psf_header

    nfreq = len(frequencies)
    psf_frames = fits.image_frames(psf_name)
    frame_id = 0
    skipped_frames = 0
    sum_psf = zeros((psf_header['NAXIS2'], psf_header['NAXIS1']), dtype=float32)
    sum_freq_hz = 0.0
    for freq_hz, psf in izip(frequencies, psf_frames):
        if bad_frames is not None and frame_id in bad_frames:
            logging.warn('skipping frame % d: in bad frame list.', frame_id)
            skipped_frames += 1
        else:
            logging.info('processing frame '+str(frame_id+1)+'/'+str(nfreq))
            sum_psf += psf
            sum_freq_hz += freq_hz
        gc.collect()
        frame_id += 1
    frames_added = nfreq - skipped_frames
    mean_psf_frame = sum_psf/frames_added
    mean_freq_hz = sum_freq_hz/frames_added
    fits.write_cube(output_fits_name, output_header, mean_psf_frame, force_overwrite)
    logging.info('Done')



def mean_psf_product(psf_name, template_psf_name, frequencies,
                     output_fits_name, force_overwrite,
                     max_mem_gb=2.0, bad_frames=None):
    logging.info('mean_psf_product()')
    psf_header = fits.get_header(psf_name)
    pixels_per_frame = psf_header['NAXIS1']*psf_header['NAXIS2']
    max_mem_bytes = max_mem_gb*1024**3
    bytes_per_input_pixel = psf_header['BITPIX']/8.0
    max_input_pixels = max_mem_bytes/bytes_per_input_pixel
    block_length = int(floor(max_input_pixels/pixels_per_frame/9.0))
    num_blocks = int(floor(psf_header['NAXIS3']/block_length))
    if psf_header['NAXIS3'] % block_length > 0:
        num_blocks += 1
    output_header = psf_header

    nfreq = len(frequencies)
    psf_frames = fits.image_frames(psf_name)
    template_psf_frames = fits.image_frames(template_psf_name)
    frame_id = 0
    skipped_frames = 0
    sum_psf_product = zeros((psf_header['NAXIS2'], psf_header['NAXIS1']),
                            dtype=float32)
    sum_freq_hz = 0.0
    for freq_hz, psf, tmpl_psf in izip(frequencies, psf_frames, template_psf_frames):
        if bad_frames is not None and frame_id in bad_frames:
            logging.warn('skipping frame % d: in bad frame list.', frame_id)
            skipped_frames += 1
        else:
            logging.info('processing frame '+str(frame_id+1)+'/'+str(nfreq))
            sum_psf_product += psf*tmpl_psf
            sum_freq_hz += freq_hz
        gc.collect()
        frame_id += 1
    frames_added = nfreq - skipped_frames
    mean_psf_product_frame = sum_psf_product/frames_added
    mean_freq_hz = sum_freq_hz/frames_added
    fits.write_cube(output_fits_name, output_header, mean_psf_product_frame,
                    force_overwrite)
    logging.info('Done')
