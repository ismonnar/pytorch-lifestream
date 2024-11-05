FROM nvcr.io/nvidia/k8s/cuda-sample:latest
COPY --from=openjdk:11-jre-slim /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME=/usr/local/openjdk-11
COPY requirements.txt .
RUN apt-get update
RUN apt install python3.8
RUN apt install pip
RUN pip install --upgrade pip==23.3
RUN pip install -r requirements.txt
CMD pytest -v ptls_tests/