from setuptools import setup, find_packages

setup(
        name='pydol',
        version='1.0.0',
        description="Python wrapper for DOLPHOT",
        url='https://github.com/Jack3690/pydol',
        author='Avinash CK',
        author_email='avinash@inaoep.mx',
        license='BSD 3-clause',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,
        package_data={'': ['pydol/photometry/*','pydol/pipline/*',
                           'pydol/bayestar/data/*']},
        classifiers=[
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3.11',
        ],
)
