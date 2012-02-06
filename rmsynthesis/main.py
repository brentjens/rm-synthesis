from numpy import array, exp, zeros, newaxis, real, imag, float32, complex64
from numpy import product, fromfile
import gc, os, sys

try:
    from itertools import izip
except ImportError:
    def izip(*args):
        return zip(*args)
    izip.__doc__ = zip.__doc__
    
import rmsynthesis.fits as fits

RMSYNTHESIS_VERSION = '0.9'

class ParseError(Exception):
    pass

class ShapeError(Exception):
    pass



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


def as_wavelength_squared(frequencies):
    """
    Convert *frequencies* (in Hz) to wavelength squared (in
    m^2). Accepts a scalar value as well as arrays. The return value
    has the same shape as *frequencies*.
    """
    return (299792458.0/frequencies)**2



    

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
            errors.append('error: number of axes in ' + name + ' is ' + str(hdr['NAXIS']) + ', not 3')

    for axis in ['NAXIS1', 'NAXIS2', 'NAXIS3']:
        if q_h[axis] != u_h[axis]:
            errors.append('error: '+axis+' in '+qname+' ('+str(q_h[axis])+') not equal to '+axis+' in '+uname+' ('+str(u_h[axis])+')')

    if q_h['NAXIS3'] != len(frequencies):
        errors.append('error: number of frames in image cubes '+qname+' and '+uname+' ('+str(q_h['NAXIS3'])+') not equal to number of frequencies in frequency file '+frequencyname+' ('+str(len(frequencies))+')')
    if len(errors) > 0:
        raise ShapeError('\n'.join(errors))
    return True
        

def rmsynthesis_phases(wavelength_squared, phi):
    """
    Compute the phase factor exp(-2j*phi*wavelength_squared).
    """
    return exp(-2j*phi*wavelength_squared)




def rmsynthesis_dirty(qcube, ucube, frequencies, phi_array):
    """
    Perform an RM synthesis on Q and U image cubes with given
    frequencies. The rmcube that is returned is complex valued and has
    a frame for every value in phi_array. It is assumed that the
    dimensions of qcube, ucube, and frequencies have been verified
    before with the help of the proper_fits_shapes() function. The
    polarization vectors are derotated to the average lambda^2.
    """
    wl2       = as_wavelength_squared(frequencies)
    rmcube    = zeros((len(phi_array), qcube.shape[1], qcube.shape[2]), dtype=complex64)
    wl2_0     = wl2.mean()
    p_complex = qcube+1j*ucube
    
    num   = len(phi_array)
    nfreq = len(frequencies)
    for i, phi in enumerate(phi_array):
        print('processing frame '+str(i+1)+'/'+str(num)+' with phi = '+str(phi))
        phases = rmsynthesis_phases(wl2-wl2_0, phi)[:, newaxis, newaxis]
        rmcube[i, :, :] = (p_complex*phases).sum(axis=0)/nfreq
    return rmcube



def mul(matrix, scalar):
    return matrix*scalar

def rmsynthesis_dirty_lowmem(qname, uname, q_factor, u_factor, frequencies, phi_array):
    """
    Perform an RM synthesis on Q and U image cubes with given
    frequencies. The rmcube that is returned is complex valued and has
    a frame for every value in phi_array. It is assumed that the
    dimensions of qcube, ucube, and frequencies have been verified
    before with the help of the proper_fits_shapes() function. The
    polarization vectors are derotated to the average lambda^2.
    """
    wl2       = as_wavelength_squared(frequencies)
    qheader   = fits.get_header(qname)
    rmcube    = zeros((len(phi_array), qheader['NAXIS2'], qheader['NAXIS1']), dtype = complex64)
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
        phases    = rmsynthesis_phases(wl2_frame, phi_array)
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
    wl2   = as_wavelength_squared(frequencies)
    wl2_0 = wl2.mean()
    return array([rmsynthesis_phases((wl2 - wl2_0), phi).mean() for phi in phi_array])


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
        rmsf_out.write('%10.4f  %10.4f %10.4f\n' % (phi, real(f_phi), imag(f_phi)))
    rmsf_out.close()


def output_pqu_fits_names(output_dir):
    r'''
    '''
    return tuple([os.path.join(output_dir, pol+'-rmcube-dirty.fits') for pol in 'pqu'])


def output_pqu_headers(fits_header):
    r'''
    '''
    p_hdr = fits_header.copy()
    p_hdr.update('POL', 'P')
    
    q_hdr = fits_header.copy()
    q_hdr.update('POL', 'Q')
    
    u_hdr = fits_header.copy()
    u_hdr.update('POL', 'U')
    
    return  p_hdr, q_hdr, u_hdr #tuple([fits_header.copy().update('POL', pol) for pol in ['P', 'Q', 'U']])



def rmsynthesis_dirty_lowmem_main(q_name, u_name, q_factor, u_factor,
                                  output_dir, freq_hz, phi_rad_m2,
                                  force_overwrite):
    r'''
    '''
    q_header      = fits.get_header(q_name)
    output_header = add_phi_to_fits_header(q_header.copy(), phi_rad_m2)
    p_out_name, q_out_name, u_out_name = output_pqu_fits_names(output_dir)
    p_out_hdr , q_out_hdr , u_out_hdr  = output_pqu_headers(output_header)

    rmcube = rmsynthesis_dirty_lowmem(q_name, u_name, q_factor, u_factor, freq_hz, phi_rad_m2)

    fits.write_cube(p_out_name, p_out_hdr,  abs(rmcube), force_overwrite = force_overwrite)
    fits.write_cube(q_out_name, q_out_hdr, real(rmcube), force_overwrite = force_overwrite)
    fits.write_cube(u_out_name, u_out_hdr, imag(rmcube), force_overwrite = force_overwrite)
    
