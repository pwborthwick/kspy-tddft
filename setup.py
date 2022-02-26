from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[Extension("ks_aello",["ks_aello.pyx"],libraries=["m"],extra_compile_args=["-ffast-math"])]
setup(name='ks_aello',ext_modules = cythonize('ks_aello.pyx',language_level=3))

