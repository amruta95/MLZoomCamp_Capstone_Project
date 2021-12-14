FROM python:3.8.12-slim

RUN apt update -y && apt install -y python-pip python-venv

RUN python -m venv python3

RUN . ./python3/bin/activate

WORKDIR /app
COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "train.py", "rfc_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]