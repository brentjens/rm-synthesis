RM-Synthesis Overview
=====================

Copyright (2005--2014) Michiel Brentjens <brentjens@astron.nl>

This Python script can perform a basic (dirty) RM-synthesis, given a
FITS cube with Stokes Q images, a FITS cube with Stokes U images, and
a text file containing the frequencies of these images. RM synthesis
is described in M.A. Brentjens & A.G. de Bruyn, A&A (2005). The script
runs under Python 2.7 as well as Python 3.


Installation
------------

This script assumes that the numpy and pyfits libraries are installed.
Install using: 

    user@machine: /.../rmsynthesis-x.y$ python setup.py install.


Usage
-----

usage: rmsynthesis [-h] [--output DIRECTORY] [--low PHI] [--high PHI]
                   [--dphi DELTA_PHI] [-f] [-c] [-q QFACTOR] [-u UFACTOR]
                   [-m GB] [-v]
                   QCUBE UCUBE FREQ_FILE

positional arguments:
  QCUBE                 FITS file containing the Q cube.
  UCUBE                 FITS file containing the U cube.
  FREQ_FILE             Ascii file with frame frequencies.

optional arguments:
  -h, --help            show this help message and exit
  --output DIRECTORY, -o DIRECTORY
                        Name of the output directory [.].
  --low PHI             Lowest Faraday depth in output cube. Default value is
                        -sqrt(3)/delta (lambda^2), where delta (lambda^2) is
                        the smallest one as computed from the frequency list.
  --high PHI            Highest Faraday depth in output cube. Default value is
                        +sqrt(3)/delta (lambda^2), where delta (lambda^2) is
                        the smallest one as computed from the frequency list.
  --dphi DELTA_PHI      Faraday depth increment between frames from the RM
                        cube. Default value is sqrt(3)/Delta (lambda^2), where
                        Delta (lambda^2) is max(lambda^2) - min(lambda^2),
                        computed from the frequency list.
  -f, --force           Force overwriting files in output directory if they
                        already exist.
  -c, --check           Perform all possible checks, but do not write any
                        files or compute an RM cube
  -q QFACTOR, --qfactor QFACTOR
                        Factor to multiply values in Q cube with, Default
                        [1.000000]
  -u UFACTOR, --ufactor UFACTOR
                        Factor to multiply values in U cube with, Default
                        [1.000000]. For WSRT data, this factor must be 1.2 if
                        it has not already been applied.
  -m GB, --maxmem GB    Maximum amount of memory to be used in GB. Default:
                        [12.505261]
  -v, --version         Print version number and exit.

Input
-----

The Q and U fits cubes are required and must have three axes. The
fastest varying axis (AXIS1) must be right ascension, the second axis
(AXIS2) declination, and the slowest varying axis (AXIS3) is the frame
number. The rmsynthesis script ignores frequency information in the
FITS cubes. It only uses frequency information provided in the text
file. Note that the order of the axes in Python/numpy is the reverse
of that in the FITS file. That is, in Python, the first axis (axis 0)
is the slowest varying axis. The pyfits library transparently handles
this conversion. Note that the Q and U cubes must be similar in the
sense that their shape and scales (ra, dec, and frame number) must be
the same.

The third required input is the list of frequencies. This must be a
text file with one frequency per line. The frequency must be in Hz and
can be either an integer or a floating point number. A (tiny) example:

1.420e9
1680000000
4800000000

Output
------

The output files are written in the current working directory, unless
otherwise specified with the -o option.

- p-rmcube-dirty.fits FITS cube with axis RA (AXIS1), Dec (AXIS2),
                      Faraday depth (AXIS3). Total linear polarization.

- q-rmcube-dirty.fits FITS cube with axis RA (AXIS1), Dec (AXIS2),
                      Faraday depth (AXIS3). Derotated Q.

- u-rmcube-dirty.fits FITS cube with axis RA (AXIS1), Dec (AXIS2),
                      Faraday depth (AXIS3). Derotated U.

- rmsf.txt            Text file with the RM spread function. The first
                      column is Faraday depth, the second column the
                      response parallel to the original polarization
                      direction ("q"), and the third column the
                      response at 45 degrees with respect to the
                      original polarization direction ("u").

- rmsynthesis.log     Contains the command line options used to obtain
                      this output.
    

                      

    
