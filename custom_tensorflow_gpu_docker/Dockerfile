FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tf-sohail

COPY requirements.txt requirements.txt

RUN pip install -q -U -r requirements.txt

RUN pip install miceforest --no-cache-dir

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y iputils-ping

RUN apt-get install git -y

EXPOSE 8888

CMD [ "tail","-f","/dev/null" ]
