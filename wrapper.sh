#!/bin/bash

python ./models.py&
wait
python ./server.py&
python ./t_bot.py&
wait
