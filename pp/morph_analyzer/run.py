# -*- coding: utf-8 -*-
import sys
import tornado.web
from tornado import web
from bson.json_util import loads, dumps
from .models import AnalysisScorerModel


class BaseHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type, Authorization")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Content-Type', 'application/json')

    def options(self, *args):
        self.set_status(200)
        self.finish()


class APIHandler(BaseHandler):

    _morph_anlyzer = AnalysisScorerModel.create_from_existed_model("lookup_disambiguator_wo_suffix")

    def post(self):
        try:
            data = loads(self.request.body.decode('utf-8'))
            tokens = data["tokens"]
            res = APIHandler._morph_anlyzer.predict(tokens)
            self.finish(dumps(res))
        except Exception as e:
            self.set_status(500, str(e))
            self.finish()


def make_app():
    handlers = [
        (r"/analyze/?", APIHandler,),
    ]
    return web.Application(handlers)


if __name__ == "__main__":
    app = make_app()
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = "8081"
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()

