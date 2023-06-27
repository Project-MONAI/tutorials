# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

from monailabel.config import settings
from monailabel.interfaces.exception import MONAILabelError, MONAILabelException
from monailabel.utils.others.class_utils import get_class_of_subclass_from_file

logger = logging.getLogger(__name__)
apps: Dict[str, Any] = {}


def app_instance(app_dir=None, studies=None, conf=None):
    app_dir = app_dir if app_dir else settings.MONAI_LABEL_APP_DIR
    studies = studies if studies else settings.MONAI_LABEL_STUDIES
    cache_key = f"{app_dir}{studies}"

    global apps
    app = apps.get(cache_key)
    if app is not None:
        return app

    conf = conf if conf else settings.MONAI_LABEL_APP_CONF
    logger.info(f"Initializing App from: {app_dir}; studies: {studies}; conf: {conf}")

    main_py = os.path.join(app_dir, "main.py")
    if not os.path.exists(main_py):
        raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, "App Does NOT have main.py")

    c = get_class_of_subclass_from_file("main", main_py, "MONAILabelApp")
    if c is None:
        raise MONAILabelException(
            MONAILabelError.APP_INIT_ERROR,
            "App Does NOT Implement MONAILabelApp in main.py",
        )

    app = c(app_dir=app_dir, studies=studies, conf=conf)
    apps[cache_key] = app
    return app


def clear_cache():
    global apps
    apps.clear()


def save_result(result, output):
    logger.info(f"Result: {json.dumps(result)}")
    if output:
        with open(output, "w") as fp:
            json.dump(result, fp, indent=2)


def run_main():
    logging.basicConfig(
        level=(logging.INFO),
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--app", type=str, default=None)
    parser.add_argument("-s", "--studies", type=str, default=None)
    parser.add_argument("-m", "--method", required=True, choices=["infer", "train", "info", "batch_infer", "scoring"])
    parser.add_argument("-r", "--request", type=str, default="{}")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    for arg in vars(args):
        logger.debug(f"USING:: {arg} = {getattr(args, arg)}")
    logger.debug("")

    logger.debug("------------------------------------------------------")
    logger.debug("SETTINGS")
    logger.debug("------------------------------------------------------")
    logger.debug(json.dumps(settings.dict(), indent=2))
    logger.debug("")

    app_dir = args.app if args.app else settings.MONAI_LABEL_APP_DIR
    studies = args.studies if args.studies else settings.MONAI_LABEL_STUDIES
    logger.debug(f"++ APP_DIR: {app_dir}")
    logger.debug(f"++ STUDIES: {studies}")

    sys.path.append(app_dir)
    sys.path.append(os.path.join(app_dir, "lib"))

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format="[%(asctime)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    )

    a = app_instance(app_dir=app_dir, studies=studies)
    request = json.loads(args.request)
    result = None

    if args.method == "infer":
        res_img, res_json = a.infer(request=request)
        result = {"label": res_img, "params": res_json}
    elif args.method == "train":
        request["local_rank"] = args.local_rank
        result = a.train(request)
    elif args.method == "info":
        result = a.info(request)
    elif args.method == "batch_infer":
        result = a.batch_infer(request)
    elif args.method == "scoring":
        result = a.scoring(request)
    else:
        parser.print_help()
        exit(-1)

    save_result(result, args.output)


if __name__ == "__main__":
    run_main()
