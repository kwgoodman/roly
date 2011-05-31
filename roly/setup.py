"""  
$ python setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("doubleheap", ["doubleheap.pyx"],
               include_dirs=[numpy.get_include()]),
               Extension("linkedlist", ["linkedlist.pyx"],
               include_dirs=[numpy.get_include()]),
               Extension("doubleheap2", ["doubleheap2.pyx"],
               include_dirs=[numpy.get_include()]),
               Extension("doubleheap3", ["doubleheap3.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

