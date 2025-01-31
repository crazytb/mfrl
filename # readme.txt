vscode 상에서 Tensorboard를 실행하면 파일 다운로드가 불가능함.

vscode terminal에서
>> tensorboard --logdir=outputs/ --host localhost --port 8088

안되면
>> python3 -m tensorboard.main --logdir=outputs/ --host localhost --port 8088