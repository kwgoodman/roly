"""  
$ python func/setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("linkedlist", ["linkedlist.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
  name = 'linkedlist',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

