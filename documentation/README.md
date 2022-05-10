# Documentation for RL homeworks and project

To make reproducing the RL homeworks and project easier, some documentation is provided. An overview of this documentation is given here. It is noted that the reports of the homeworks and/or project may also contain important information to run the code that is not repeated here.

## Table of contents

- [Contact information](#contact-information)
- [Homework](#homework)
  - [Setup Anaconda environment](#setup-anaconda-environment)
  - [Import Anaconda environment](#import-anaconda-environment)
  
- [Project](#project)

<hr>


## Contact information

| Name             | Student ID | VUB mail                                                  | Personal mail                                               |
| ---------------- | ---------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| Lennert Bontinck | 0568702    | [lennert.bontinck@vub.be](mailto:lennert.bontinck@vub.be) | [info@lennertbontinck.com](mailto:info@lennertbontinck.com) |

<hr>


## Homework

The RL course has 3 different homework assignments, of which 2 were obligatory. An anaconda environment was used to run the Python code. The environment can be set up using the procedure described below or by importing the exported environment that is provided.

### Setup Anaconda environment

- Install [the free version of Anaconda Navigator](https://www.anaconda.com/products/individual). V2.1.4 was used.
- Using the `Anaconda Prompt (Anaconda3)` application, create and setup a new environment.

  - **NOTE**: it might be required to run the prompt as administrator for all of the below steps.

  - ```shell
    # Create the rl-homework Anaconda environment.
    conda create -n rl-homework python=3.8.10
    
    # Activates the previously created rl-homework Anaconda environment.
    conda activate rl-homework
    ```

- Install some conda available packages on the environment

  - ```shell
    # Pandas is a famous Python Data Analysis Library and is used by a lot of other packages.
    # The following command installs Pandas and its dependencies. V1.4.1 was used.
    conda install pandas=1.4.1
    
    # We install pip to install packages not available from conda install. V21.2.2 was used.
    conda install pip=21.2.2
    ```
  
- Install some pip available packages on the environment

  - ```shell
    # Install Matplotlib for plotting purposes.
    # V3.5.1 was used.
    pip install matplotlib==3.5.1
    ```

### Import Anaconda environment

The anaconda Windows and MacOS environment is also exported to the YML files. This means it can be imported using only a few lines of code rather then having to set it up yourself. These YML files are available in the [environments subfolder](environments/). You can load it in via the terminal as follows:


```shell
# Navigate to the folder where the YML file is located
cd VUB-RL/documentation/environments
# Configure a new environment from the YML file
## These were exported using: conda env export > rl-environment-{platform}.yml --no-builds
conda env create -f rl-environment-{platform}.yml
```

<hr>


## Project

TODO
