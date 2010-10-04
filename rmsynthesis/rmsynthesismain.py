from  numpy import *
import pyfits
import os,sys

class ParseError(Exception):
    pass

class ShapeError(Exception):
    pass



def file_exists(filename, verbose=False):
    """
    Returns True oif *filename* exists, False if it does not. If
    *verbose* is True, it also prints an error message if the file
    does not exist.
    """
    try:
        os.stat(filename)
        return True
    except (OSError,):
        e = sys.exc_info()[1]
        if verbose:
            print 'error: '+str(e)
        return False
        

def parse_frequency_file(filename):
    """
    Read a text file containing one floating point number per line,
    and return an array of those values. Empty lines are
    ignored. Comments can be included in the file behind a # mark.
    The frequencies should be listed in Hz.

    Raises a printable ParseError in case of problems with the
    contents of the file.
    """
    try:
        return array([float(x.split('#')[0].strip()) for x in open(filename).readlines() if x.split('#')[0].strip() != ''])
    except (ValueError,):      # Use this construction to be backwards
        e = sys.exc_info()[1]  # compatible with Python 2.5.  Proper
                               # Python 2.6/2.7/3.0 is
                               # except ValueError as e:
                               # etc...
        raise ParseError('while parsing '+filename+': '+str(e))


def as_wavelength_squared(frequencies):
    """
    Convert *frequencies* (in Hz) to wavelength squared (in
    m^2). Accepts a scalar value as well as arrays. The return value
    has the same shape as *frequencies*.
    """
    return (299792458.0/frequencies)**2



def get_fits_header(fitsname):
    """
    Return the header of the first Header Data Unit (HDU) of the FITS
    file with name *fitsname*. May raise an OSError if the file cannot
    be opened.
    """
    hdulist=pyfits.open(fitsname)
    header=hdulist[0].header
    hdulist.close()
    return header


def get_fits_header_data(fitsname):
    """
    Return a (header, data) tuple of the first Header Data Unit (HDU)
    of the FITS file with name *fitsname*. May raise an OSError if the
    file cannot be opened.
    """    
    hdulist=pyfits.open(fitsname)
    header=hdulist[0].header
    data=hdulist[0].data
    hdulist.close()
    return header,data



def proper_fits_shapes(qname, uname, frequencyname):
    """
    Verify that the Q and U FITS cubes and the file with frequency
    data have compatible shapes. *qname* and *uname* are the filenames
    of the Q and U FITS files, respectively; *frequencyname* is the
    filename of the frequency file.

    Returns True if all is well, raises ShapeError otherwise.
    """
    frequencies=parse_frequency_file(frequencyname)
    qh = get_fits_header(qname)
    uh = get_fits_header(uname)
    errors=[]
    for name,h in [(qname, qh), (uname, uh)]:
        if h['NAXIS'] != 3:
            errors.append('error: number of axes in '+name+' is '+str(h['NAXIS'])+', not 3')
            pass
        pass

    for axis in ['NAXIS1', 'NAXIS2', 'NAXIS3']:
        if qh[axis] != uh[axis]:
            errors,append('error: '+axis+' in '+qname+' ('+str(qh[axis])+') not equal to '+axis+' in '+uname+' ('+str(uh[axis])+')')
            pass
        pass

    if qh['NAXIS3'] != len(frequencies):
        errors.append('error: number of frames in image cubes '+qname+' and '+uname+' ('+str(qh['NAXIS3'])+') not equal to number of frequencies in frequency file '+frequencyname+' ('+str(len(frequencies))+')')
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
    wl2 = as_wavelength_squared(frequencies)
    rmcube=zeros((len(phi_array), qcube.shape[1], qcube.shape[2]), dtype=complex64)
    wl2_0 = wl2.mean()
    p_complex= qcube+1j*ucube
    
    n     = len(phi_array)
    nfreq = len(frequencies)
    for i,phi in enumerate(phi_array):
        print 'processing frame '+str(i+1)+'/'+str(n)+' with phi = '+str(phi)
        phases=rmsynthesis_phases(wl2-wl2_0, phi)[:,newaxis,newaxis]
        rmcube[i,:,:] = (p_complex*phases).sum(axis=0)/nfreq
        pass
    return rmcube


def compute_rmsf(frequencies, phi_array):
    """
    Compute the Rotation Measure Spread Function, derotating to the
    average lambda^2.
    """
    wl2   = as_wavelength_squared(frequencies)
    wl2_0 = wl2.mean()
    return array([rmsynthesis_phases((wl2 - wl2_0), phi).mean() for phi in phi_array])



def write_fits_cube(data_array, fits_header, fits_name, force_overwrite=False):
    hdulist    = pyfits.HDUList()
    hdu        = pyfits.PrimaryHDU()
    hdu.header = fits_header
    hdu.data   = data_array
    hdulist.append(hdu)
    hdulist.writeto(fits_name, clobber=force_overwrite)
    hdulist.close()
    pass


def write_rmcube(rmcube, fits_header, output_dir, force_overwrite=False):
    write_fits_cube(abs(rmcube), fits_header,
                    os.path.join(output_dir, '/p-rmcube-dirty.fits'),
                    force_overwrite=force_overwrite)
    
    write_fits_cube(rmcube.real, fits_header,
                    os.path.join(output_dir, '/q-rmcube-dirty.fits'),
                    force_overwrite=force_overwrite)
    
    write_fits_cube(rmcube.imag, fits_header,
                    os.path.join(output_dir, '/u-rmcube-dirty.fits'),
                    force_overwrite=force_overwrite)
    pass


def write_rmsf(phi, rmsf, output_dir):
    rmsf_out=open(os.path.join(output_dir, '/rmsf.txt'), 'w')
    for phi,y in zip(phi, rmsf):
        rmsf_out.write('%10.4f  %10.4f %10.4f\n' % (phi, real(y), imag(y)))
        pass
    rmsf_out.close()
    pass
