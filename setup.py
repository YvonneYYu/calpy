from setuptools import setup
import sys

if sys.version_info < (3, 6):
      sys.exit('Sorry, Python < 3.6 is not supported. Please update Python.')

setup(name='calpy',
      packages = ['calpy','calpy.dsp', 'calpy.entropy', 'calpy.rqa', 'calpy.plots', 'calpy.utilities'],
      version='0.1.3',
      description='Communication Analytics Lab Toolbox',
      author =['Yvonne Yu & Paul Vrbik'],
      author_email = ['yeyang.yu@uqconnect.edu.au', 'paulvrbik@gmail.com'],
      url = 'https://github.com/YvonneYYu/calpy',
      license = 'MIT',
      keywords = ['Natural Language Understanding', 'Signal Processing'],
      install_requires = [
            'numpy',
            'scipy',
      ]
)
