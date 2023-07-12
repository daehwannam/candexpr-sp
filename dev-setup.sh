#!/usr/bin/sh

set -e
CONFIG_NAME=$1

if [ -z "$CONFIG_NAME" ]; then
    echo 'No name is specified'
else
    FILE_PATH=".dir-locals-files/$CONFIG_NAME"
    if [ -f "$FILE_PATH" ]; then
        ln -s ".dir-locals-files/$CONFIG_NAME" .dir-locals.el
        echo ".dir-locals.el is linked to .dir-locals-files/$CONFIG_NAME"
    else
        echo "$FILE_PATH does not exist"
    fi
fi
