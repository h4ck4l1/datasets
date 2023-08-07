FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tf-sohail

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install miceforest --no-cache-dir

RUN apt-get update && apt-get install -y iputils-ping

RUN apt-get install git -y

EXPOSE 8888

CMD [ "tail","-f","/dev/null" ]


