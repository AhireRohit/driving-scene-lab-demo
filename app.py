"""Hosted entrypoint for the original recorded replay. Missing data is an error."""
import os
from pathlib import Path
import sys
from html import escape
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from shadowdrive.replay import create_app
bundle=Path(os.environ.get('RECORDED_BUNDLE',str(ROOT/'recorded_bundle')))
try:
    app=create_app(ROOT,recorded_bundle=bundle)
    server=app.server
    @server.get('/healthz')
    def health():return {'status':'ok','mode':'recorded-replay','scenes':['scene-0103','scene-0553','scene-0796','scene-0916'],'frames':162}
except (OSError,ValueError,KeyError) as error:
    from flask import Flask
    app=Flask(__name__);server=app
    configuration_error=str(error)
    @server.get('/healthz')
    def health():return {'status':'configuration-error','mode':'recorded-replay','detail':configuration_error},503
    @server.route('/',defaults={'path':''})
    @server.route('/<path:path>')
    def missing_assets(path):
        return ('<!doctype html><title>Driving Scene Lab — configuration error</title>'
                '<h1>Recorded replay assets are not configured</h1>'
                '<p>The original four-scene replay cannot start. No procedural replacement is provided.</p>'
                f'<pre>{escape(configuration_error)}</pre>'),503

if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT','7860')),debug=False)
