import unittest
from rmsynthesis.rmsynthesismain import *
import pyfits

class RmSynthesisTest(unittest.TestCase):
    def setUp(self):
        self.test_dir='testdata'
        self.freq_filename=os.path.join(self.test_dir, 'frequencies.txt')
        self.freq_filename_parse_error=os.path.join(self.test_dir,'frequencies-parse-error.txt')
        self.does_not_exist=os.path.join(self.test_dir, 'does-not-exist.txt')
        self.qname=os.path.join(self.test_dir, 'Q_Fthinsource.fits')
        self.uname=os.path.join(self.test_dir, 'U_Fthinsource.fits')

        self.freq=arange(100)*10e6 +300e6
        self.phi=arange(-100.0,100.0,4.0)

        pass


    def test_file_exists(self):
        self.assertTrue(file_exists(self.freq_filename))
        self.assertFalse(file_exists(self.does_not_exist))
        self.assertFalse(file_exists(self.does_not_exist, verbose=True))
        pass


    def test_parse_frequency_file(self):
        map(self.assertAlmostEquals, parse_frequency_file(self.freq_filename),
            [314.159265e6,
             314.359265e6,
             320e6,
             330e6,
             340000000,
             350e6,
             3.6e8,
             3.7e8,])
        self.assertRaises(ParseError, lambda : parse_frequency_file(self.freq_filename_parse_error))
        pass


    def test_as_wavelength_squared(self):
        self.assertAlmostEquals(as_wavelength_squared(299792458.0), 1.0)
        map(self.assertAlmostEquals, as_wavelength_squared(array([299792458.0, 299792458.0/2.0, 299792458.0/2.0])), [1.0, 4.0, 4.0])
        empty=as_wavelength_squared(array([]))
        self.assertEquals(len(empty), 0)
        pass


    def test_get_fits_header(self):
        qheader=get_fits_header(self.qname)
        uheader=get_fits_header(self.uname)
        self.assertEquals(qheader['POL'].strip(), 'Q')
        self.assertEquals(qheader['NAXIS'], 3)
        self.assertAlmostEquals(qheader['CDELT1'], 1.3e+6)
        self.assertEquals(uheader['POL'].strip(), 'U')
        self.assertRaises(IOError, lambda : get_fits_header(self.does_not_exist))
        self.assertRaises(IOError, lambda : get_fits_header(self.freq_filename))
        pass


    def test_get_fits_header_data(self):
        hq,dq=get_fits_header_data(self.qname)
        self.assertEquals(type(hq), type(pyfits.Header()))
        self.assertEquals(type(dq), type(array([], dtype='>f4')))
        self.assertEquals(dq.dtype, '>f4')
        self.assertEquals(len(dq.shape), 3)
        self.assertEquals(dq.shape, (100,100,100))
         
        self.assertRaises(IOError, lambda : get_fits_header_data(self.does_not_exist))
        self.assertRaises(IOError, lambda : get_fits_header_data(self.freq_filename))
        pass


    def test_rmsynthesis_phases(self):
        self.assertAlmostEquals(rmsynthesis_phases(1.0, pi), 1.0)
        self.assertAlmostEquals(rmsynthesis_phases(pi, 0.5), -1.0)
        map(self.assertAlmostEquals,
            rmsynthesis_phases(array([pi, 0.5*pi]), 0.5),
            [-1.0, -1j])
        map(self.assertAlmostEquals,
            rmsynthesis_phases(pi, array([0.5, -0.25])),
            [-1.0, +1j])
        pass

    

    def test_rmsynthesis_dirty(self):

        f_phi=zeros((len(self.phi)), dtype=complex64)
        f_phi[15] = 3-4j

        p_f=(f_phi[newaxis,:]*exp(2j*as_wavelength_squared(self.freq)[:,newaxis]*self.phi[newaxis,:])).sum(axis=1)

        qcube=zeros((len(self.freq),5,7), dtype=complex64)
        ucube=zeros((len(self.freq),5,7), dtype=complex64)
        qcube[:,2,3]=p_f.real
        ucube[:,2,3]=p_f.imag

        rmcube=rmsynthesis_dirty(qcube, ucube, self.freq, self.phi)
        rmcube_cut= rmcube.copy()
        rmcube_cut[:,2,3]*= 0.0
        self.assertTrue(all(rmcube_cut==0.0))
        self.assertNotAlmostEquals(rmcube[:,2,3].mean(), 0.0)
        self.assertAlmostEquals(rmcube[15,2,3],
                                (3-4j)*exp(2j*self.phi[15]*as_wavelength_squared(self.freq).mean()),
                                places=6)
        self.assertEquals(list(rmcube[:,2,3] == rmcube.max()).index(True), 15)
        pass


    def test_compute_rmsf(self):
        rmsf = compute_rmsf(self.freq, self.phi)
        wl=as_wavelength_squared(self.freq)
        wl0=wl.mean()
        map(self.assertAlmostEquals,
            rmsf,
            exp(-2j*self.phi[newaxis,:]*(wl-wl0)[:,newaxis]).mean(axis=0))
        pass
    
    pass



    

if __name__ == '__main__':
    unittest.main()
    pass
