FROM python:3.12-rc-slim-buster

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ["python", "./web_inference.py"]