from setuptools import setup
from rmsynthesis.main import __version__
setup(name='rmsynthesis',
      version=__version__,
      description='Simple Faraday rotation measure synthesis tool',
      author='Michiel Brentjens',
      author_email='brentjens@astron.nl',
      url='http://www.astron.nl/~brentjens/',
      requires=['numpy', 'astropy'],
      packages=['rmsynthesis'],
      scripts=['bin/rmsynthesis'],
     )
