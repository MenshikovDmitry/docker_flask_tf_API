FROM python:3.6
#FROM tensorflow/tensorflow

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

ADD models ./models
COPY app.py .
COPY backlog.txt /home/
#ADD uploads ./uploads
ADD templates ./templates
RUN mkdir '/home/uploads'
EXPOSE 5000

CMD [ "python", "app.py" ]
