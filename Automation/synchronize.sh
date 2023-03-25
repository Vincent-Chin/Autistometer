#!/bin/bash
pushd /home/ubuntu/Autistometer/
/home/ubuntu/Autistometer/venv/bin/python3 -O -c "import Engine.main; Engine.main.daily_maintenance()"
popd
