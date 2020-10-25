
from setuptools import setup, dist, Extension
from os import path

dist.Distribution().fetch_build_eggs(['numpy>=1.19.0'])

try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.19.0.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '1.0'

here = path.abspath(path.dirname(__file__))

# Get description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()

# Get dependencies and install
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as fh:
    all_req = fh.read().split('\n')

install_requires = [i.strip() for i in all_req if 'git+' not in i]
dependency_links = [i.strip().replace('git+', '') for i in all_req if i.startswith('git+')]

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extentions = [
    Extension(
        'surprisica/utils',
        ['surprisica/utils' + ext],
        include_dirs=[np.get_include()]
    )]

if USE_CYTHON:
    ext_modules = cythonize(extentions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extentions

setup(
    name='surprisica',
    author='Amirali',
    description='',
    long_description=long_description,
    version=__version__,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3'
    ],
    keywords='recommender, recommender system, context_aware recommender',
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    dependency_links=dependency_links,
    entry_points={'console_script':
                  ['surprisica = surprisica.__main__:main']}
)
