FROM tensorflow/tensorflow:latest-gpu

WORKDIR /tf-sohail

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y iputils-ping

RUN pip install --upgrade pip

RUN pip install -q -U -r requirements.txt --ignore-installed

RUN pip install "dask[complete]"

RUN pip install miceforest --no-cache-dir

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get install git -y

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-459.0.0-linux-x86_64.tar.gz

RUN tar -xf google-cloud-cli-459.0.0-linux-x86_64.tar.gz

RUN ./google-cloud-sdk/install.sh

EXPOSE 8888

CMD ["tail","-f","/dev/null"]
