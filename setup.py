
from setuptools import setup, find_packages

setup(
    name='ottensap',
    version='0.0',
    description='Tensap / OpenTURNS bridge',
    url='None',
    author='Julien Schueller',
    author_email='schueller@phimeca.com',
    license='License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
    ],
    keywords='Karhunen-Loeve',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['tensap' ,'openturns'],
    package_data={},

)
