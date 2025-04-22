#!/bin/bash

# optional argument to specify the sweep name
SWEEP_NAME=$1

# a script to detach, and kill all the screens
# useful when you have a lot of screens running
# and you want to clear them all
if [ -z "$SWEEP_NAME" ]; then
    # no sweep name provided, kill all screens
    for session in $(screen -ls | grep -o '[0-9]*\.' | grep -o '[0-9]*'); do
        # stop any code running
        screen -S "${session}" -X stuff "^C"
        # kill the screen
        screen -S "${session}" -X quit
    done
else
    # sweep name provided, kill only matching screens
    for session in $(screen -ls | grep "$SWEEP_NAME" | grep -o '[0-9]*\.' | grep -o '[0-9]*'); do
        # stop any code running
        screen -S "${session}" -X stuff "^C"
        # kill the screen
        screen -S "${session}" -X quit
    done
fi
