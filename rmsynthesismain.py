from  numpy import *
import pyfits

import os

class ParseError(Exception):
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

