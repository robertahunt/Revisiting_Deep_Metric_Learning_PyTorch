FROM nvidia/cuda:11.3.0-base-ubuntu20.04 AS base

# necessary to be able to save files as 'rob' user instead of root
ARG USERNAME=rob
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN apt-get update
RUN apt-get install nano git openssh-client -y 
#RUN apt-get install python3 pip



RUN mkdir -p /home/rob/.ssh 
RUN chown -R rob:rob /home/rob/.ssh
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /home/rob/.ssh/config



# Install necessary python packages
#RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# For type checking
#RUN pip3 install types-six types-requests pandas

# For running jupyter notebooks
#RUN pip3 install jupyter
ENV JUPYTER_TOKEN="easy"

# CMD needs to be run apparently so that everything afterwards can access the volumes 
CMD ["/bin/bash"]




# In case it already exists, remove the old git folder
#RUN rm -rf $REPO_FOLDER
RUN export PATH=$PATH:/home/rob/.local/bin
#RUN python3 -m pip install black pytest gitpython

RUN apt-get install nano git openssh-client openssl make gcc libssl-dev wget  libbz2-dev zlib1g-dev build-essential -y
ENV TZ="Africa/Lusaka"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime  && echo $TZ > /etc/timezone
RUN apt-get install tzdata -y
RUN apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 liblzma-dev libsqlite3-dev  -y
RUN apt-get install libffi-dev

#RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh"
#RUN bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -p /miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
RUN bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p /miniconda
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init


# RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch -y 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade

RUN python -m pip install albumentations
# used to speed up adding changes to the build process. Everything above should be pretty static
# FROM base AS build1

RUN python -m pip install pretrainedmodels
RUN conda install -y pandas matplotlib faiss-gpu -c conda-forge

RUN mkdir /home/ngw861
RUN mkdir /home/ngw861/01_abbey_rove
RUN chown rob: /home/ngw861 -R
RUN chown rob:rob /miniconda -R
# Switch to rob user
USER rob


RUN python -m pip freeze > /home/ngw861/01_abbey_rove/requirements.txt