from setuptools import setup
import sys

#abort installation if Python version does not meet requirement
if sys.version_info < (3, 5):
      sys.exit('Sorry, Python < 3.5 is not supported. Please update to Python 3.5 or above')

setup(name='calpy',
      packages = ['calpy','calpy.dsp', 'calpy.entropy', 'calpy.rqa', 'calpy.plots', 'calpy.utilities', 'calpy.students'],
      version='1.0',
      description='Communication Analytics Lab Toolbox',
      author =['Yvonne Yu & Paul Vrbik'],
      author_email = ['yeyang.yu@uqconnect.edu.au', 'paulvrbik@gmail.com'],
      url = 'https://github.com/YvonneYYu/calpy',
      license = 'MIT',
      keywords = ['Natural Language Understanding'],
      install_requires = [
            'numpy',
            'scipy',
            'bokeh',
            'matplotlib'
      ]
)
