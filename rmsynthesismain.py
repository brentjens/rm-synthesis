from  numpy import *
import pyfits

import os

class ParseError(Exception):
    pass

class ShapeError(Exception):
    pass



def file_exists(filename, verbose=False):
    try:
        os.stat(filename)
        return True
    except OSError as e:
        if verbose:
            print e
        return False
        

def parse_frequency_file(filename):
    try:
        return array([float(x.split('#')[0].strip()) for x in open(filename).readlines() if x.split('#')[0].strip() != ''])
    except ValueError as e:
        raise ParseError('while parsing '+filename+': '+e.message)


def as_wavelength_squared(frequencies):
    return (299792458.0/frequencies)**2


def get_fits_header_data(fitsname):
    hdulist=pyfits.open(fitsname)
    header=hdulist[0].header
    data=hdulist[0].data
    hdulist.close()
    return header,data

def get_fits_header(fitsname):
    hdulist=pyfits.open(fitsname)
    header=hdulist[0].header
    hdulist.close()
    return header


def proper_fits_shapes(qname, uname, frequencyname):
    ok=True
    frequencies=parse_frequency_file(frequencyname)
    qh = get_fits_header(qname)
    uh = get_fits_header(uname)
    for name,h in [(qname, qh), (uname, uh)]:
        if h['NAXIS'] != 3:
            print 'number of axes in '+name+' is '+str(h['NAXIS'])+', not 3'
            ok=False
            pass
        pass

    for axis in ['NAXIS1', 'NAXIS2', 'NAXIS3']:
        if qh[axis] != uh[axis]:
            print axis+' in '+qname+' ('+str(qh[axis])+') not equal to '+axis+' in '+uname+' ('+str(uh[axis])+')'
            ok=False
            pass
        pass

    if qh['NAXIS3'] != len(frequencies):
            print 'number of frames in image cubes '+qname+' and '+uname+' ('+str(qh['NAXIS3'])+') not equal to number of frequencies in frequency file '+frequencyname+' ('+str(len(frequencies))+')'
            ok=False
    return ok
        

def rmsynthesis_phases(wavelength_squared, phi):
    return exp(-2j*phi*wavelength_squared)
    

def rmsynthesis_dirty(qcube, ucube, frequencies, phi_array):
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
    write_fits_cube(abs(rmcube), fits_header, output_dir+'/p-rmcube-dirty.fits', force_overwrite=force_overwrite)
    
    write_fits_cube(rmcube.real, fits_header, output_dir+'/q-rmcube-dirty.fits', force_overwrite=force_overwrite)
    
    write_fits_cube(rmcube.imag, fits_header, output_dir+'/u-rmcube-dirty.fits', force_overwrite=force_overwrite)
    pass


def write_rmsf(phi, rmsf, output_dir):
    rmsf_out=open(output_dir+'/rmsf.txt', 'w')
    for phi,y in zip(phi, rmsf):
        rmsf_out.write('%10.4f  %10.4f %10.4f\n' % (phi, real(y), imag(y)))
        pass
    rmsf_out.close()
    pass
