# https://stackoverflow.com/questions/40935074/how-to-use-all-cpus-with-pathos-using-nested-pools?noredirect=1&lq=1

import pathos
import multiprocess


class NoDaemonProcess(multiprocess.Process):

    """NoDaemonProcess class.

    Inherit from :class:`multiprocessing.Process`.
    ``daemon`` attribute always returns False.

    """

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(pathos.multiprocessing.Pool):

    """NestedPool class.

    Inherit from :class:`pathos.multiprocessing.Pool`.
    Enable nested process pool.

    """

    Process = NoDaemonProcess