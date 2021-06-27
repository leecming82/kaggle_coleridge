FROM nvcr.io/nvidia/pytorch:21.04-py3
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /root

RUN pip install scikit-learn pandas matplotlib pathos sentencepiece deepspeed datasets deberta
RUN apt-get update && apt-get install -y openssh-server screen
RUN mkdir /var/run/sshd
RUN echo 'root:testdocker' | chpasswd
RUN sed -i 's/.*PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN git clone https://github.com/facebookresearch/fastText.git
WORKDIR /root/fastText
RUN pip install .

WORKDIR /root
RUN pip install git+https://github.com/huggingface/transformers

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /root
COPY . .

RUN /opt/conda/condabin/conda init

EXPOSE 22
CMD ["bash", "-c", "/usr/sbin/sshd && jupyter notebook"]