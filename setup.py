from setuptools import setup
import sys
import os

#install dependency if not a dependency is installed else continue
def check_dependency(package):
      try:
            import package
      except ImportError:
            print('Dependency {} is not installed, installing it now!'.format(package))
            os.system('sudo pip3 install {}'.format(package))

packages = ['numpy', 'scipy', 'bokeh', 'matplotlib']
for package in packages:
      check_dependency(package)

#abort installation if Python version does not meet requirement
if sys.version_info < (3, 5):
      sys.exit('Sorry, Python < 3.5 is not supported. Please update to Python 3.5 or above')

setup(name='calpy',
      packages = ['calpy','calpy.dsp', 'calpy.entropy', 'calpy.rqa', 'calpy.plots', 'calpy.utilities'],
      version='0.1.4',
      description='Communication Analytics Lab Toolbox',
      author =['Yvonne Yu & Paul Vrbik'],
      author_email = ['yeyang.yu@uqconnect.edu.au', 'paulvrbik@gmail.com'],
      url = 'https://github.com/YvonneYYu/calpy',
      license = 'MIT',
      keywords = ['Natural Language Understanding', 'Signal Processing'],
      install_requires = [
            'numpy',
            'scipy',
            'bokeh',
            'matplotlib'
      ]
)
