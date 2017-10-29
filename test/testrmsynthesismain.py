import os, unittest, shutil, sys
from rmsynthesis.main import *
import rmsynthesis.fits as fits

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

from numpy import arange, complex128, exp, float32, newaxis, ones, pi

rmsynthesis_phases = phases_lambda2_to_phi


class RmSynthesisTest(unittest.TestCase):
    def setUp(self):
        self.test_dir       = 'testdata'
        self.freq_filename  = os.path.join(self.test_dir, 'frequencies.txt')
        self.freq_filename_parse_error = os.path.join(self.test_dir, 'frequencies-parse-error.txt')
        self.does_not_exist = os.path.join(self.test_dir, 'does-not-exist.txt')
        self.qname = os.path.join(self.test_dir, 'Q_Fthinsource.fits')
        self.uname = os.path.join(self.test_dir, 'U_Fthinsource.fits')

        self.freq = arange(100)*10e6 +300e6
        self.phi  = arange(-100.0, 100.0, 4.0)

        for from_file, reference in zip(parse_frequency_file(self.freq_filename),
                                        [314.159265e6, 314.359265e6, 320e6, 330e6,
                                         340000000, 350e6, 3.6e8, 3.7e8,]):
            self.assertAlmostEquals(from_file, reference)

        self.assertRaises(ParseError, lambda : parse_frequency_file(self.freq_filename_parse_error))

    def test_rmsynthesis_phases(self):
        self.assertAlmostEquals(rmsynthesis_phases(1.0, pi), 1.0)
        self.assertAlmostEquals(rmsynthesis_phases(pi, 0.5), -1.0)
        map(self.assertAlmostEquals,
            rmsynthesis_phases(array([pi, 0.5*pi]), 0.5),
            [-1.0, -1j])
        map(self.assertAlmostEquals,
            rmsynthesis_phases(pi, array([0.5, -0.25])),
            [-1.0, +1j])

    

    def test_rmsynthesis_dirty(self):
        f_phi = zeros((len(self.phi)), dtype = complex64)
        f_phi[15] = 3-4j

        p_f = (f_phi[newaxis, :]*exp(2j*wavelength_squared_m2_from_freq_hz(self.freq)[:, newaxis]*self.phi[newaxis, :])).sum(axis = 1)

        qcube = zeros((len(self.freq), 5, 7), dtype = complex64)
        ucube = zeros((len(self.freq), 5, 7), dtype = complex64)
        qcube[:, 2, 3] = p_f.real
        ucube[:, 2, 3] = p_f.imag

        rmcube     = rmsynthesis_dirty(qcube, ucube, self.freq, self.phi)
        rmcube_cut = rmcube.copy()
        rmcube_cut[:, 2, 3] *= 0.0
        self.assertTrue((rmcube_cut == 0.0).all())
        self.assertNotAlmostEquals(rmcube[:, 2, 3].mean(), 0.0)
        self.assertAlmostEquals(
            rmcube[15, 2, 3],
            (3-4j)*exp(2j*self.phi[15]*wavelength_squared_m2_from_freq_hz(self.freq).mean()),
            places=6)
        self.assertEquals(list(rmcube[:, 2, 3] == rmcube.max()).index(True), 15)


    def test_compute_rmsf(self):
        rmsf   = compute_rmsf(self.freq, self.phi)
        wl_m   = wavelength_squared_m2_from_freq_hz(self.freq)
        wl0_m  = wl_m.mean()
        map(self.assertAlmostEquals,
            rmsf,
            exp(-2j*self.phi[newaxis, :]*(wl_m - wl0_m)[:, newaxis]).mean(axis = 0))



    def test_add_phi_to_fits_header(self):
        head = pyfits.Header()
        head.set('SIMPLE', True)
        head.set('BITPIX', -32)
        head.set('NAXIS', 3)
        head.set('NAXIS1', 1)
        head.set('NAXIS2', 1)

        head_phi = add_phi_to_fits_header(head, [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0])
        self.assertEquals(head_phi['NAXIS3'], 8)
        self.assertEquals(head_phi['CTYPE3'], 'FARDEPTH')
        self.assertEquals(head_phi['CUNIT3'], 'RAD/M^2')
        self.assertAlmostEquals(head_phi['CRPIX3'], 1.0)
        self.assertAlmostEquals(head_phi['CRVAL3'], -10)
        self.assertAlmostEquals(head_phi['CDELT3'], +2.0)

        self.assertRaises(KeyError, lambda: head['NAXIS3'])
        self.assertEquals(type(add_phi_to_fits_header(head, [-10.0, -8.0])), type(pyfits.Header()))
        
        self.assertRaises(ShapeError, lambda: add_phi_to_fits_header(head, [-10.0]))
        self.assertRaises(ShapeError, lambda: add_phi_to_fits_header(head, []))


    def test_write_rmcube(self):
        if sys.version_info.major < 3:
            output_dir = os.tempnam()
            os.mkdir(output_dir)
        else:
            import tempfile
            output_dir = tempfile.mkdtemp()
        try:
            header_out = pyfits.Header()
            header_out.set('SIMPLE', True)
            header_out.set('BITPIX', -32)
            header_out.set('NAXIS', 3)
            header_out.set('NAXIS1', 7)
            header_out.set('NAXIS2', 5)
            header_out.set('NAXIS3', 10)

            rmcube = (3.0-4.0j)*ones((10, 5, 7), dtype = complex128)
            self.assertRaises(IOError,
                              lambda: write_rmcube(rmcube, header_out,
                                                   output_dir+'no-existent',
                                                   force_overwrite = False))
            write_rmcube(rmcube, header_out, output_dir, force_overwrite = False)
            file_names = os.listdir(output_dir)
            self.assertEquals(len(file_names), 3)
            self.assertTrue('p-rmcube-dirty.fits' in file_names)
            self.assertTrue('q-rmcube-dirty.fits' in file_names)
            self.assertTrue('u-rmcube-dirty.fits' in file_names)

            hdr_p, data_p = fits.get_header_data(os.path.join(output_dir, 'p-rmcube-dirty.fits'))
            hdr_q, data_q = fits.get_header_data(os.path.join(output_dir, 'q-rmcube-dirty.fits'))
            hdr_u, data_u = fits.get_header_data(os.path.join(output_dir, 'u-rmcube-dirty.fits'))

            self.assertEquals(hdr_p['POL'], 'P')
            self.assertEquals(hdr_q['POL'], 'Q')
            self.assertEquals(hdr_u['POL'], 'U')

            self.assertEquals(data_p.shape, (10, 5, 7))
            self.assertAlmostEquals((data_p-5.0).sum(), 0.0)
            self.assertAlmostEquals((data_p-5.0).std(), 0.0)

            self.assertEquals(data_q.shape, (10, 5, 7))
            self.assertAlmostEquals((data_q-3.0).sum(), 0.0)
            self.assertAlmostEquals((data_q-3.0).std(), 0.0)

            self.assertEquals(data_u.shape, (10, 5, 7))
            self.assertAlmostEquals((data_u+4.0).sum(), 0.0)
            self.assertAlmostEquals((data_u+4.0).std(), 0.0)
            
        finally:
            shutil.rmtree(output_dir)
 

    def test_write_rmsf(self):
        if sys.version_info.major < 3:
            output_dir = os.tempnam()
            os.mkdir(output_dir)
        else:
            import tempfile
            output_dir = tempfile.mkdtemp()
        try:
            rmsf  = compute_rmsf(self.freq, self.phi)
            fname = os.path.join(output_dir, 'rmsf.txt')
            write_rmsf(self.phi, rmsf, output_dir)
            self.assertTrue(os.path.exists(fname))
            lines    = open(fname).readlines()
            contents = array([[float(x) for x in  l.split()] for l in lines])
            phi_in   = contents[:, 0]
            rmsf_in  = contents[:, 1]+1j*contents[:, 2]

            self.assertTrue(abs(phi_in-self.phi).mean()< 0.0001)
            self.assertTrue(abs(phi_in-self.phi).std() < 0.0001)

            self.assertTrue(abs(rmsf_in-rmsf).mean()< 0.0001)
            self.assertTrue(abs(rmsf_in-rmsf).std()< 0.0001)

        finally:
            shutil.rmtree(output_dir)

    

if __name__ == '__main__':
    unittest.main()
