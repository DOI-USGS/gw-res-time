import numpy as np

class imeth(object):
    '''parse modflow budget data according to imeth'''
    def __init__(self, nlay, nrow, ncol):
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.rxc = nrow * ncol
        self.nn = nlay * self.rxc

    def imeth1(self, BUFF):

        return BUFF.ravel()

    def imeth2(self, BUFF):

        tmp = np.zeros((self.nn, 7))

        tseqnum = np.array(BUFF.node - 1)
        iflow = np.array(BUFF.q)
        tmp[tseqnum, 0] = iflow

        return tmp

    def imeth3(self, BUFF):

        tmp = np.zeros((self.nn , 7))

        lays = BUFF[0]
        modarray = BUFF[1]

        cols, rows = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow))
        arr = np.zeros((self.nlay, self.nrow, self.ncol))
        arr[lays - 1, rows, cols] = modarray

        tmp[: , 6] = arr.ravel()

        return tmp

    def imeth4(self, BUFF):

        tmp = np.zeros((self.nn, 7))
        tmp[0 : self.rxc , 6] = BUFF.ravel()

        return tmp

    def imeth5(self, BUFF):

        tmp = np.zeros((self.nn, 7))

        p = BUFF.dtype.names
        BUFF.dtype.names = [item.lower().strip() for item in p]

        if 'iface' in BUFF.dtype.names:
            tseqnum = np.array(BUFF.node - 1)
            iflow = np.array(BUFF.q)
            iface = BUFF.iface.astype(int)
            tmp[tseqnum, iface] = iflow

        else:
            tseqnum = np.array(BUFF.node - 1)
            iflow = np.array(BUFF.q)
            tmp[tseqnum, 0] = iflow

        return tmp