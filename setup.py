from setuptools import setup, find_packages

setup(
    name='gensim',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="GenSim: Generating Robotic Simulation Tasks via Large Language Models.",
    author='Lirui Wang',
    author_email='liruiw@mit.edu',
    url='https://liruiw.github.io/gensim',
   #  install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
    keywords=['Large Language Models', 'Simulation', 'Vision Language Grounding', 'Robotics', 'Manipulation'],
)
