r'''
A collection of small utilities to interface with ``pyfits``.
'''

from numpy import fromfile, product #pylint: disable=no-name-in-module
try:
    from astropy.io import fits as pyfits
    from astropy.io.fits import StreamingHDU
except ImportError:
    import pyfits
    from pyfits.core import StreamingHDU

import os, sys, logging


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

    >>> get_header('testdata/Q_Fthinsource.fits').cards
    ('SIMPLE', True, 'Written by IDL:  Tue Sep 28 22:42:54 2010')
    ('BITPIX', -32, 'Number of bits per data pixel')
    ('NAXIS', 3, 'Number of data axes')
    ('NAXIS1', 100, '')
    ('NAXIS2', 100, '')
    ('NAXIS3', 100, '')
    ('DATE', '2010-09-28', 'Creation UTC (CCCC-MM-DD) date of FITS header')
    ('COMMENT', "FITS (Flexible Image Transport System) format is defined in 'Astronomy", '')
    ('COMMENT', "and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H", '')
    ('CTYPE1', 'X', '')
    ('CRVAL1', 0, '')
    ('CDELT1', 1300000.0, '')
    ('CRPIX1', 0, '')
    ('CUNIT1', 'Hz', '')
    ('CTYPE2', 'Y', '')
    ('CRVAL2', 0, '')
    ('CRVAL3', 110000000.0, '')
    ('POL', 'Q', '')
    >>> get_header('testdata/non-existent.fits')
    Traceback (most recent call last):
    ...
    FileNotFoundError: [Errno 2] No such file or directory: 'testdata/non-existent.fits'

    IOError: File does not exist: 'testdata/non-existent.fits'
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

    FileNotFoundError
        If ``fits_name`` cannot be opened.

    **Example**

    >>> hdr, data = get_header_data('testdata/Q_Fthinsource.fits')
    >>> get_header('testdata/Q_Fthinsource.fits') == hdr
    True
    >>> data.shape
    (100, 100, 100)

    FITS single floats are big endian 32 bit numbers:
    >>> str(data.dtype)
    '>f4'

    Operating on them in any way, converts the output to 'float32':
    >>> str((data*1).dtype)
    'float32'

    Of course, opening a non-existent file results in an ``OSError``...
    >>> get_header_data('testdata/non-existent.fits')
    Traceback (most recent call last):
    ...
    FileNotFoundError: [Errno 2] No such file or directory: 'testdata/non-existent.fits'

    IOError: File does not exist: 'testdata/non-existent.fits'

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
    info = hdulist.fileinfo(0)
    hdulist.close()
    return info['datLoc'], info['datSpan']



def image_frames(fits_name):
    r'''
    An iterator over a FITS image (hyper) cube:

    **Parameters**

    fits_name : string
        Fits file over which to iterate.

    **Returns**

    A 2D numpy.array for each image in the (hyper)cube contained in
    the file. It has shape (NAXIS2, NAXIS1).

    **Example**

    >>> i = 0
    >>> for frame in image_frames('testdata/Q_Fthinsource.fits'):
    ...     if i % 10 == 0:
    ...         print('%r: %3.2f' % (frame.shape, frame[50,50]))
    ...     i += 1
    (100, 100): -9.29
    (100, 100): 23.13
    (100, 100): 81.73
    (100, 100): -87.40
    (100, 100): 40.75
    (100, 100): 68.86
    (100, 100): -72.43
    (100, 100): -87.07
    (100, 100): -0.70
    (100, 100): 76.28

    '''
    header = get_header(fits_name)
    dtype = pyfits.BITPIX2DTYPE[header['BITPIX']]
    shape = (header['NAXIS2'], header['NAXIS1'])
    frame_size = product(shape)*abs(header['BITPIX']/8)

    data_start, data_length = get_data_offset_length(fits_name)
    file_stream = open(fits_name, mode='rb')
    file_stream.seek(data_start)
    try:
        while file_stream.tell() +frame_size < data_start + data_length:
            frame = fromfile(file_stream,
                             count=product(shape),
                             dtype=dtype).reshape(shape)
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

    A ``astropy.io.fits.StreamingHDU`` instance.

    **Raises**

    OSError
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
    >>> hdr2 == hdr
    True

    >>> (data2 == data).all()
    True

    If we try this again:

    >>> shdu  = streaming_output_hdu(fits_name, hdr, force_overwrite = False)
    Traceback (most recent call last):
    ...
    OSError: testdata/partial_output.fits already exists and is not overwritten.

    IOError: testdata/partial_output.fits already exists and is not overwritten.

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
            logging.warn('Overwriting existing file %r.' % fits_name)
            os.remove(fits_name)
        else:
            raise IOError('%s already exists and is not overwritten.' %
                          fits_name)
    return StreamingHDU(fits_name, fits_header)


def write_cube(fits_name, fits_header, data, force_overwrite=False):
    r'''
    Write an image cube to a FITS file containing only one HDU.

    **Parameters**

    fits_name : string
        Output file name.

    fits_header : pyfits.Header
        The header describing ``data``.

    data : numpy.array
        The (hyper)cube to write to the FITS file.

    force_overwrite : bool

        If True, the output file will be overwritten. Otherwise, an
        IOError is raised if the output file already exists.

    **Returns**

    None

    **Raises**

    IOError
        If the output file already exists and ``force_overwrite == False``.

    **Example**

    >>> fits_name = 'testdata/write_cube_output.fits'
    >>> if os.path.exists(fits_name): os.remove(fits_name)
    >>>
    >>> hdr, data = get_header_data('testdata/Q_Fthinsource.fits')
    >>> write_cube(fits_name, hdr, data)
    >>> hdr2, data2 = get_header_data(fits_name)
    >>> hdr2 == hdr
    True

    >>> (data2 == data).all()
    True

    >>> write_cube(fits_name, hdr, data)
    Traceback (most recent call last):
    ...
    OSError: File 'testdata/write_cube_output.fits' already exists.

    >>> import time
    >>> current_time = time.time()
    >>> _=[_ for i in range(100000)]
    >>> write_cube(fits_name, hdr, data, force_overwrite = True)
    >>> os.path.exists(fits_name)
    True
    >>> os.stat(fits_name).st_size
    4003200
    >>> os.stat(fits_name).st_ctime >= current_time
    True
    >>> os.remove(fits_name)

    '''
    hdu = pyfits.PrimaryHDU()
    hdu.header = fits_header
    hdu.data = data
    hdulist = pyfits.HDUList()
    hdulist.append(hdu)
    hdulist.writeto(fits_name, overwrite=force_overwrite)
    hdulist.close()
