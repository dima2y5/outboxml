import sys
sys.path.append("../../../")
sys.path.append("../../../outboxml")

import time
from typing import Callable, Any

import schedule
from outboxml import config
from examples.titanic.titanic_basic import titanic_example
from outboxml.automl_utils import check_for_new_data


def main(auto_ml_script: Callable, config: Any, waiting_time: float):
    params = {'script': auto_ml_script, 'config': config, 'waiting_time': waiting_time}
    # schedule.every(2).minutes.do(check_for_new_data, **params)

    while True:
        check_for_new_data(**params)
        # schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main(auto_ml_script=titanic_example,
         config=config,
         waiting_time=2 * 60
         )
