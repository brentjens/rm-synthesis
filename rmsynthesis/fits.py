
def get_fits_header(fitsname):
    """
    Return the header of the first Header Data Unit (HDU) of the FITS
    file with name *fitsname*. May raise an OSError if the file cannot
    be opened.
    """
    hdulist = pyfits.open(fitsname)
    header  = hdulist[0].header
    hdulist.close()
    return header



def get_fits_data_start_and_size(fitsname):
    """
    Return the first byte of the data of the first HDU in the file,
    followed by the length of the data block, including padding, in
    bytes.
    """
    hdulist = pyfits.open(fitsname)
    info  = hdulist.fileinfo(0)
    hdulist.close()
    return info['datLoc'], info['datSpan']
    


def get_fits_header_data(fitsname):
    """
    Return a (header, data) tuple of the first Header Data Unit (HDU)
    of the FITS file with name *fitsname*. May raise an OSError if the
    file cannot be opened.
    """    
    hdulist = pyfits.open(fitsname)
    header  = hdulist[0].header
    data    = hdulist[0].data
    hdulist.close()
    return header, data


def fits_image_frames(fitsname):
    """
    An iterator over a FITS image cube:

    for frame in fits_image_frames('example.fits'):
        print frame.shape, frame.max()

    """
    header = get_fits_header(fitsname)
    dtype  = pyfits.hdu.PrimaryHDU.NumCode[header['BITPIX']]
    shape  = (header['NAXIS2'], header['NAXIS1'])
    frame_size = product(shape)*abs(header['BITPIX']/8)
    
    data_start, data_length = get_fits_data_start_and_size(fitsname)
    file_stream = open(fitsname, mode='rb')
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
    


def fits_write_header(fits_name, fits_header, force_overwrite):
    r'''
    Write only the header (including padding) to file ``fits_name``
    '''
    header_lines = [str(card) for card in fits_header.ascardlist()]
    padding = 'END'+' '*77+' '*80*(36 - ((len(header_lines) + 1) % 36))

    if force_overwrite or not os.path.exists(fits_name):
        out_file = open(fits_name, 'w')
        for line in header_lines:
            out_file.write(line)
            out_file.write(padding)
        return out_file



def fits_write_partial_cube(file_stream, cube):
    r'''
    '''
    if sys.byteorder == 'little':
        cube.byteswap().tofile(file_stream)
    else:
        cube.tofile(file_stream)


def fits_padding(shape, bytes_per_sample):
    r'''
    '''
    last_chunk = (product(shape)*bytes_per_sample) % 2880
    if last_chunk == 0:
        return zeros((0), dtype = uint8)
    else:
        return zeros((2880 - last_chunk), dtype = uint8)
