FROM python:3.9


COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt
COPY . /tmp/
COPY t_bot.py .
COPY config.py .
COPY causal_discovery_algs.py .
COPY schemas.py .
COPY server.py .
COPY utils.py .
COPY crud.py .
COPY db_settings.py .
COPY models.py .
COPY ready.txt .
COPY wrapper.sh .
RUN chmod +777 ./wrapper.sh

ENTRYPOINT ./wrapper.sh