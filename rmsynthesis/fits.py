r'''
A collection of small utilities to interface with ``pyfits``.
'''

import pyfits, os


def get_header(fits_name):
    r'''
    Read only the header in FITS file ``fits_name``. Internally simply
    calls ``pyfits.getheader()``.

    **Parameters**

    fits_name : string
        File name from which to read the header.

    **Returns**

    A pyfits.Header instance.

    **Raises**

    OSError
        If ``fits_name`` cannot be opened.

    **Example**

    >>> print(get_header('testdata/Q_Fthinsource.fits'))
    SIMPLE  =                    T / Written by IDL:  Tue Sep 28 22:42:54 2010      
    BITPIX  =                  -32 / Number of bits per data pixel                  
    NAXIS   =                    3 / Number of data axes                            
    NAXIS1  =                  100 /                                                
    NAXIS2  =                  100 /                                                
    NAXIS3  =                  100 /                                                
    DATE    = '2010-09-28'         / Creation UTC (CCCC-MM-DD) date of FITS header  
    COMMENT FITS (Flexible Image Transport System) format is defined in 'Astronomy  
    COMMENT and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H    
    CTYPE1  = 'X       '           /                                                
    CRVAL1  =                    0 /                                                
    CDELT1  =          1.30000E+06 /                                                
    CRPIX1  =                    0 /                                                
    CUNIT1  = 'Hz      '           /                                                
    CTYPE2  = 'Y       '           /                                                
    CRVAL2  =                    0 /                                                
    CRVAL3  =          1.10000E+08 /                                                
    POL     = 'Q       '           /                                                

    >>> print(get_header('testdata/non-existent.fits'))
    Traceback (most recent call last):
    ...
    IOError: [Errno 2] No such file or directory: 'testdata/non-existent.fits'

    '''
    return pyfits.getheader(fits_name)


def get_header_data(fits_name):
    r'''
    Returns the first HDU of ``fits_name`` as a (header, data) tuple.

    **Parameters**

    fits_name : string
        File name from which to read the header.

    **Returns**

    A (pyfits.Header, numpy.ndarray) tuple. 

    **Raises**

    IOError
        If ``fits_name`` cannot be opened.

    **Example**

    >>> hdr, data = get_header_data('testdata/Q_Fthinsource.fits')
    >>> print(hdr)
    SIMPLE  =                    T / Written by IDL:  Tue Sep 28 22:42:54 2010      
    BITPIX  =                  -32 / Number of bits per data pixel                  
    NAXIS   =                    3 / Number of data axes                            
    NAXIS1  =                  100 /                                                
    NAXIS2  =                  100 /                                                
    NAXIS3  =                  100 /                                                
    DATE    = '2010-09-28'         / Creation UTC (CCCC-MM-DD) date of FITS header  
    COMMENT FITS (Flexible Image Transport System) format is defined in 'Astronomy  
    COMMENT and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H    
    CTYPE1  = 'X       '           /                                                
    CRVAL1  =                    0 /                                                
    CDELT1  =          1.30000E+06 /                                                
    CRPIX1  =                    0 /                                                
    CUNIT1  = 'Hz      '           /                                                
    CTYPE2  = 'Y       '           /                                                
    CRVAL2  =                    0 /                                                
    CRVAL3  =          1.10000E+08 /                                                
    POL     = 'Q       '           /                                                

    >>> data.shape
    (100, 100, 100)

    FITS single floats are big endian 32 bit numbers.
    >>> str(data.dtype)
    '>f4'

    Operating on them in any way, converts the output to 'float32':    
    >>> str((data*1).dtype)
    'float32'

    Of course, opening a non-existent file results in an ``OSError``...    
    >>> get_header_data('testdata/non-existent.fits')
    Traceback (most recent call last):
    ...
    IOError: [Errno 2] No such file or directory: 'testdata/non-existent.fits'

    '''
    return pyfits.getheader(fits_name), pyfits.getdata(fits_name)


def get_data_offset_length(fits_name):
    r'''
    Returns the offset from the beginning of the file as well as the
    length of the data block, in bytes, for the first Header Data Unit
    (HDU). The length includes padding to the next 2880 byte
    boundary.

    **Parameters**

    fits_name : string
        File name from which to read the header.

    **Returns**

    A tuple of integers (offset, data_length).

    **Raises**

    IOError:
        If ``fits_name`` cannot be opened.

    **Example**

    >>> get_data_offset_length('testdata/Q_Fthinsource.fits')
    (2880, 4000320)
    '''
    hdulist = pyfits.open(fits_name)
    info    = hdulist.fileinfo(0)
    hdulist.close()
    return info['datLoc'], info['datSpan']
    


def fits_image_frames(fits_name):
    """
    An iterator over a FITS image (hyper) cube:

    for frame in fits_image_frames('example.fits'):
        print frame.shape, frame.max()

    """
    header = get_header(fits_name)
    dtype  = pyfits.hdu.PrimaryHDU.NumCode[header['BITPIX']]
    shape  = (header['NAXIS2'], header['NAXIS1'])
    frame_size = product(shape)*abs(header['BITPIX']/8)
    
    data_start, data_length = get_fits_data_start_and_size(fits_name)
    file_stream = open(fits_name, mode='rb')
    file_stream.seek(data_start)
    try:
        while file_stream.tell() +frame_size < data_start + data_length:
            frame = fromfile(file_stream,
                             count = product(shape),
                             dtype = dtype).reshape(shape)
            if sys.byteorder == 'little':
                yield frame.byteswap()
            else:
                yield frame
    finally:
        file_stream.close()
    



def streaming_output_hdu(fits_name, fits_header, force_overwrite):
    r'''
    Creates an HDU stream to which data can be written
    incrementally. This is very useful for writing large FITS image
    cubes.

    **Parameters**

    fits_name : string
        Output file name..

    fits_header : pyfits.Header
        The header for the output FITS file.

    force_overwrite : Bool
        If ``True``, overwrite existing file with name ``fits_name``.
        If ``False``, an ``OSError`` is raised if the file already
        exists.

    **Returns**

    A ``pyfits.core.StreamingHDU`` instance.

    **Raises**

    IOError
        If output file exists and ``force_overwrite`` is ``False``

    **Example**

    >>> fits_name = 'testdata/partial_output.fits'
    >>> if os.path.exists(fits_name): os.remove(fits_name)
    >>> 
    >>> hdr, data = get_header_data('testdata/Q_Fthinsource.fits')
    >>> shdu  = streaming_output_hdu(fits_name, hdr, force_overwrite = False)
    >>> for image in data:
    ...     reached_end = shdu.write(image)
    >>> shdu.close()
    >>> 
    >>> os.path.exists(fits_name)
    True
    >>> os.stat(fits_name).st_size
    4003200
    >>> hdr2, data2 = get_header_data(fits_name)
    >>> print(hdr2)
    SIMPLE  =                    T / Written by IDL:  Tue Sep 28 22:42:54 2010      
    BITPIX  =                  -32 / Number of bits per data pixel                  
    NAXIS   =                    3 / Number of data axes                            
    NAXIS1  =                  100 /                                                
    NAXIS2  =                  100 /                                                
    NAXIS3  =                  100 /                                                
    DATE    = '2010-09-28'         / Creation UTC (CCCC-MM-DD) date of FITS header  
    COMMENT FITS (Flexible Image Transport System) format is defined in 'Astronomy  
    COMMENT and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H    
    CTYPE1  = 'X       '           /                                                
    CRVAL1  =                    0 /                                                
    CDELT1  =          1.30000E+06 /                                                
    CRPIX1  =                    0 /                                                
    CUNIT1  = 'Hz      '           /                                                
    CTYPE2  = 'Y       '           /                                                
    CRVAL2  =                    0 /                                                
    CRVAL3  =          1.10000E+08 /                                                
    POL     = 'Q       '           /                                                
    >>> (data2 == data).all()
    True

    If we try this again:
    >>> shdu  = streaming_output_hdu(fits_name, hdr, force_overwrite = False)
    Traceback (most recent call last):
    ...
    IOError: testdata/partial_output.fits already exists. Will not overwrite unless forced.

    Let's try that again:    
    >>> import time
    >>> current_time = time.time()
    >>> shdu  = streaming_output_hdu(fits_name, hdr, force_overwrite = True)
    >>> for image in data:
    ...     reached_end = shdu.write(image)
    >>> shdu.close()
    >>> 
    >>> os.path.exists(fits_name)
    True
    >>> os.stat(fits_name).st_size
    4003200
    >>> os.stat(fits_name).st_mtime > current_time
    True
    >>> os.remove(fits_name)

    '''

    if os.path.exists(fits_name):
        if force_overwrite:
            os.remove(fits_name)
        else:
            raise IOError('%s already exists. Will not overwrite unless forced.' %
                          fits_name)
    return pyfits.core.StreamingHDU(fits_name, fits_header)
    
    
def write_cube(data_array, fits_header, fits_name, force_overwrite = False):
    r'''
    '''
    hdulist    = pyfits.HDUList()
    hdu        = pyfits.PrimaryHDU()
    hdu.header = fits_header
    hdu.data   = data_array
    hdulist.append(hdu)
    hdulist.writeto(fits_name, clobber = force_overwrite)
    hdulist.close()
