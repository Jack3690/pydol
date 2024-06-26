from setuptools import setup, find_packages

setup(
        name='pydol',
        version='0.0.1',
        description="Python wrapper for DOLPHOT: Photometry tool",
        url='https://github.com/Jack3690/pydol',
        author='Avinash CK',
        author_email='avinash@inaoep.mx',
        license='BSD 2-clause',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        install_requires=['matplotlib', 'astropy', 'photutils',
                          'numpy', 'jwst','crds'],
        include_package_data=True,
        package_data={'': ['pydol/photometry/*','pydol/pipline/*']},
        classifiers=[
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.9',
        ],
)
