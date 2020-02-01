FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

ADD models ./models
COPY app.py .
ADD uploads ./uploads
ADD templates ./templates

EXPOSE 5000

CMD [ "python", "app.py" ]
