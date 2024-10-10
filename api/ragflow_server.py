import os
import signal
import logging
import traceback
from api import utils
from api.apps import app
from werkzeug.serving import run_simple
from concurrent.futures import ThreadPoolExecutor
from api.settings import HOST, HTTP_PORT, access_logger, stat_logger


if __name__ == '__main__':
    print(r"""
    ____                 ______ __               
   / __ \ ____ _ ____ _ / ____// /____  _      __
  / /_/ // __ `// __ `// /_   / // __ \| | /| / /
 / _, _// /_/ // /_/ // __/  / // /_/ /| |/ |/ / 
/_/ |_| \__,_/ \__, //_/    /_/ \____/ |__/|__/  
              /____/                             

    """, flush=True)
    stat_logger.info(f'project base: {utils.file_utils.get_project_base_directory()}')

    thr = ThreadPoolExecutor(max_workers=1)

    # start http server
    try:
        stat_logger.info("RAG Flow http server start...")
        werkzeug_logger = logging.getLogger("werkzeug")
        for h in access_logger.handlers:
            werkzeug_logger.addHandler(h)
        run_simple(hostname=HOST, port=HTTP_PORT, application=app, threaded=True)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGKILL)