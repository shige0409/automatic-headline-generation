#!/usr/bin/zsh
nohup pipenv run jupyter lab --port 9999 >> jupyter.log 2>&1 &