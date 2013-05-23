from distutils.core import setup

setup(name='rmsynthesis',
      version='1.0-rc2',
      description='Simple Faraday rotation measure synthesis tool',
      author='Michiel Brentjens',
      author_email='brentjens@astron.nl',
      url='',
      requires=['numpy', 'pyfits'],
      packages=['rmsynthesis'],
      scripts=['bin/rmsynthesis'],
     )
