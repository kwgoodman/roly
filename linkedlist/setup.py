"""  
$ python func/setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("move_median", ["move_median.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
  name = 'move_median',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

