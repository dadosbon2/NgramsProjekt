#!/bin/bash
ssh -L 8888:localhost:8888 apthy@mcgarret.informatik.uni-halle.de #Change your username 
#activate your enviroment in the server (ask for gpu is you want) and then use the normal jupyter lab command
#After the server iniates you can type http://localhost:8888 on your browser to open Jupyter lab, it might ask for a token, you cant find it on the logs
#of that the jupyter server prints when initializing something like https:1.1.20.30.20.token=sakdjsalkfjlq2321094823094jofj is it quite long.
#you want to use sakdjsalkfjlq2321094823094jofj ask token in this case

