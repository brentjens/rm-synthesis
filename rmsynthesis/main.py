r'''
The ``main`` module implements the actual RM-synthesis.
'''


from numpy import array, complex64, pi, exp, zeros, newaxis, real, imag
import gc, os, sys

try:
    from itertools import izip
except ImportError:
    def izip(*args):
        return zip(*args)
    izip.__doc__ = zip.__doc__
    
import rmsynthesis.fits as fits

RMSYNTHESIS_VERSION = '0.9'

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



def almost_equal(x, y, epsilon = 1e-9):
    r'''
    Returns True if

    .. math::
       \frac{|x-y|}{\mathrm{max}(|x|, |y|)} < \epsilon


    If either of the numbers is exactly zero, it returns True if the
    other number's absolute value is below :math:``\epsilon``.
 
    **Parameters**

    x : float or complex
        One of the numbers to compare.

    y : float or complex
        The other number to compare.

    epsilon : float
        Fractional accuracy to which the numbers must be equal.

    **Returns**

    A boolean indicating if ``x`` and ``y`` are almost equal.

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

    if x == 0.0:
        return abs(y) < epsilon
    if y == 0.0:
        return abs(x) < epsilon
    return abs(x - y)/max(abs(x), abs(y)) < epsilon




def wavelength_squared_m2(freq_hz):
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
    >>> wavelength_squared_m2(c)
    1.0
    >>> wavelength_squared_m2(array([c, c/2.0, c/3.0]))
    array([ 1.,  4.,  9.])
    >>> wavelength_squared_m2(array([[c], [c/2.0], [c/3.0]]))
    array([[ 1.],
           [ 4.],
           [ 9.]])
    >>> wavelength_squared_m2(array([]))
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

    >>> phases_lambda2_to_phi(1.0, pi)
    1.0
    >>> phases_lambda2_to_phi(pi/2, 0.5)
    -1.j
    >>> phases_lambda2_to_phi(array([pi, 0.5*pi]), 0.5)
    array([-1., -1.j])
    >>> phases_lambda2_to_phi(pi, array([0.5, -0.25]))
    array([-1., 1.j])

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
    q_h         = fits.get_header(qname)
    u_h         = fits.get_header(uname)
    errors      = []
    for name, hdr in [(qname, q_h), (uname, u_h)]:
        if hdr['NAXIS'] != 3:
            errors.append('error: number of axes in %s  is  %d, not 3' %
                          (name, hdr['NAXIS']))

    for axis in ['NAXIS1', 'NAXIS2', 'NAXIS3']:
        if q_h[axis] != u_h[axis]:
            errors.append('error: %s in %s (%d) not equal to %s in %s (%d)' % 
                          (axis, qname, q_h[axis], axis, uname, u_h[axis]))

    if q_h['NAXIS3'] != len(frequencies):
        msg = ('error: frames in %r and %r (%d) not the same as in %r (%d)' %
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
    wl2       = wavelength_squared_m2(frequencies)
    rmcube    = zeros((len(phi_array), qcube.shape[1], qcube.shape[2]),
                      dtype=complex64)
    wl2_0     = wl2.mean()
    p_complex = qcube+1j*ucube
    
    num   = len(phi_array)
    nfreq = len(frequencies)
    for i, phi in enumerate(phi_array):
        print('processing frame %4d/%d, phi = %7.1f' % (i+1, num, phi))
        phases = phases_lambda2_to_phi(wl2-wl2_0, phi)[:, newaxis, newaxis]
        rmcube[i, :, :] = (p_complex*phases).sum(axis=0)/nfreq
    return rmcube



def mul(matrix, scalar):
    return matrix*scalar

def rmsynthesis_dirty_lowmem(qname, uname, q_factor, u_factor, 
                             frequencies, phi_array):
    """
    Perform an RM synthesis on Q and U image cubes with given
    frequencies. The rmcube that is returned is complex valued and has
    a frame for every value in phi_array. It is assumed that the
    dimensions of qcube, ucube, and frequencies have been verified
    before with the help of the proper_fits_shapes() function. The
    polarization vectors are derotated to the average lambda^2.
    """
    wl2       = wavelength_squared_m2(frequencies)
    qheader   = fits.get_header(qname)
    rmcube    = zeros((len(phi_array), qheader['NAXIS2'], qheader['NAXIS1']),
                      dtype = complex64)
    wl2_0     = wl2.mean()
    
    num      = len(phi_array)
    nfreq    = len(frequencies)
    q_frames = fits.image_frames(qname)
    u_frames = fits.image_frames(uname)
    frame_id = 0
    wl2_norm = wl2 - wl2_0 
    for q_frame, u_frame in izip(q_frames, u_frames):
        print('processing frame '+str(frame_id+1)+'/'+str(nfreq))
        p_complex = q_frame*q_factor + 1.0j*u_frame*u_factor
        wl2_frame = wl2_norm[frame_id]
        phases    = phases_lambda2_to_phi(wl2_frame, phi_array)
        for frame, phase in enumerate(phases):
            rmcube[frame, :, :] += p_complex*phase
        frame_id += 1
        gc.collect()
        
    return rmcube/nfreq




def compute_rmsf(frequencies, phi_array):
    """
    Compute the Rotation Measure Spread Function, derotating to the
    average lambda^2.
    """
    wl2   = wavelength_squared_m2(frequencies)
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
    fhdr.update('NAXIS3', len(phi_array))
    fhdr.update('CRPIX3', 1.0)
    fhdr.update('CRVAL3', phi_array[0])
    fhdr.update('CDELT3', phi_array[1]-phi_array[0])
    fhdr.update('CTYPE3', 'FARDEPTH')
    fhdr.update('CUNIT3', 'RAD/M^2')
    return fhdr



def write_rmcube(rmcube, fits_header, output_dir, force_overwrite=False):
    """
    Writes a complex valued, 3D  *rmcube* to *output_dir* as three
    separate FITS cubes:
    
    - *output_dir*/p-rmcube-dirty.fits : Absolute value.
    - *output_dir*/q-rmcube-dirty.fits : Real part.
    - *output_dir*/u-rmcube-dirty.fits : Imaginary part.

    This function raises an IOError if *output_dir* does not exist, or
    is not writable, or if the output file(s) already exist and
    *force_overwrite* is False. If *force_overwrite* is True, the
    output files will be overwritten.
    """
    fhp = fits_header.copy()
    fhp.update('POL', 'P')
    fits.write_cube(os.path.join(output_dir, 'p-rmcube-dirty.fits'),
                    fhp, abs(rmcube),
                    force_overwrite=force_overwrite)
    
    fhq = fits_header.copy()
    fhq.update('POL', 'Q')
    fits.write_cube(os.path.join(output_dir, 'q-rmcube-dirty.fits'),
                    fhq, rmcube.real,
                    force_overwrite=force_overwrite)
    
    fhu = fits_header.copy()
    fhu.update('POL', 'U')
    fits.write_cube(os.path.join(output_dir, 'u-rmcube-dirty.fits'),
                    fhu, rmcube.imag,
                    force_overwrite=force_overwrite)


def write_rmsf(phi, rmsf, output_dir):
    rmsf_out = open(os.path.join(output_dir, 'rmsf.txt'), 'w')
    for phi, f_phi in zip(phi, rmsf):
        rmsf_out.write('%10.4f  %10.4f %10.4f\n' %
                       (phi, real(f_phi), imag(f_phi)))
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
    p_hdr.update('POL', 'P')
    
    q_hdr = fits_header.copy()
    q_hdr.update('POL', 'Q')
    
    u_hdr = fits_header.copy()
    u_hdr.update('POL', 'U')
    
    return  p_hdr, q_hdr, u_hdr



def rmsynthesis_dirty_lowmem_main(q_name, u_name, q_factor, u_factor,
                                  output_dir, freq_hz, phi_rad_m2,
                                  force_overwrite):
    r'''
    '''
    q_header      = fits.get_header(q_name)
    output_header = add_phi_to_fits_header(q_header.copy(), phi_rad_m2)
    p_out_name, q_out_name, u_out_name = output_pqu_fits_names(output_dir)
    p_out_hdr , q_out_hdr , u_out_hdr  = output_pqu_headers(output_header)

    rmcube = rmsynthesis_dirty_lowmem(q_name, u_name, q_factor, u_factor,
                                      freq_hz, phi_rad_m2)

    fits.write_cube(p_out_name, p_out_hdr,  abs(rmcube),
                    force_overwrite = force_overwrite)
    fits.write_cube(q_out_name, q_out_hdr, real(rmcube),
                    force_overwrite = force_overwrite)
    fits.write_cube(u_out_name, u_out_hdr, imag(rmcube),
                    force_overwrite = force_overwrite)
    
