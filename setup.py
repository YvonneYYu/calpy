from setuptools import setup, find_packages

setup(name='calpy',
      packages = ['calpy'],
      version='0.1.2',
      packages = find_packages(exclude=['testing']),
      description='Communication Analytics Lab Toolbox',
      author =['Yvonne Yu & Paul Vrbik'],
      author_email = ['yeyang.yu@uqconnect.edu.au', 'paulvrbik@gmail.com'],
      url = 'https://github.com/YvonneYYu/calpy',
      license = 'MIT',
      keywords = ['Natural Language Understanding', 'Signal Processing']
)
