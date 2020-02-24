import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ergonomics_metrics',
     version='0.2',
     author="Rosario Di Carlo",
     author_email="rs.dicarlo@gmail.com",
     description="Ergonomics metrics",
     long_description="Compute ergonomics metrics from human pose",
     long_description_content_type="text/markdown",
     url="https://github.com/rs9000/ergonomics",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )