# coding: utf-8

import abc
from typing import Union, Dict, Any

from pydantic import BaseModel


class Task(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, conf: Union[Dict[str, Any], BaseModel] = None, *args, **kwargs):
        if isinstance(conf, BaseModel):
            conf = conf.model_dump()
        self.conf = conf
        self.daemon_target = kwargs.get('daemon_target')
        self._use_demon = False if not conf else conf.get('use_demon', False)
        self._exception = None

    def before_run(self):
        pass

    def after_run(self):
        pass

    @abc.abstractmethod
    def run(self):
        """Raise exception if not success."""

    def start(self) -> Any:
        try:
            self.before_run()
            self._daemon_run()
            ret = self.run()
            return 0 if ret is None else ret
        except BaseException as ex:
            self._exception = ex
            # do record or report
            raise ex
        finally:
            self.after_run()

    def _daemon_run(self) -> None:
        if self._use_demon:
            import threading
            t = threading.Thread(target=self.daemon_target, name="daemon")
            t.setDaemon(True)
            t.start()
