FROM tensorflow/tensorflow:2.15.0-gpu-jupyter
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update
RUN apt-get install -y libgl1-mesa-dev

RUN pip install transformers
RUN pip install sentencepiece
RUN pip install pandas
RUN pip install opencv-python