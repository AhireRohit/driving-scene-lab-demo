"""Dataset-free deployment entrypoint. Full recorded replay uses scripts/launch_replay.py."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from shadowdrive.replay import create_app
app=create_app(ROOT,demo=True)
server=app.server

@server.get('/healthz')
def health():return {'status':'ok','mode':'procedural-demo'}

if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT','7860')),debug=False)
