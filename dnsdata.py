import pandas as pd
import numpy as np
from math import floor, pi



def read_dnsin(fdir):
# Reads dns.in as a dictionary. Syntax:
# dictionary = read_dnsin(path/to/folder/with/dnsin/)

    with open(fdir + 'dns.in') as dnsin:
        temp = dnsin.readlines()

    tmpln = ''

    spltmark = '\n'

    for line in temp:
        if 'restart_file' in line or 'time_from_restart' in line:
            continue
        line = line.replace('\t',' ')
        foundequal = False
        foundvalue = False
        for char in line:
            tmpln += char
            if not foundequal:
                if char == '=':
                    foundequal = True
            elif not foundvalue:
                if not char == ' ':
                    foundvalue = True
            else:
                if char == ' ':
                    tmpln += spltmark
                    foundequal = False
                    foundvalue = False
    tmpln = tmpln.replace(' ', '') # remove white spaces
    tmpln = tmpln.replace('\n\n','\n') # remove multiple endlines

    # now tmpln is a single string with endlines, where each line reads "something=somevalue" without spaces
    # it's going to be turned into a dictionary

    tmpln = tmpln.replace('=',':') # replace equals with :
    tmpln = tmpln.replace('\n',',') # replace endlines with commas
    tmpln = '{"' + tmpln # open square brackets and brackets
    tmpln = tmpln.replace(',', ',"') # open brackets after comma
    tmpln = tmpln.replace(':', '":') # close brackets before colon

    # now remove last two characters (which will be ,") and close brackets
    tmpln = tmpln[:-2]
    tmpln = tmpln + '}'

    # one final trick: replace key "ni" with "re" for better clarity
    tmpln = tmpln.replace('"ni"', '"re"')

    return eval(tmpln)



class mesh():
# Extracts mesh and properties. All mesh properties here calculated are meant
# as in the standard scaling units (which is, the ones used for the simulation).
# Syntax:
# mesh_obj = mesh(dns_dictionary)
# where dns_dictionary can be extracted from dns.in with read_dnsin()

    def __init__(self, dnsdata):

        self.nx = dnsdata['nx']
        self.ny = dnsdata['ny']
        self.nz = dnsdata['nz']
        self.alfa0 = dnsdata['alfa0']
        self.beta0 = dnsdata['beta0']
        self.a = dnsdata['a']

        # find nxd,nzd
        self.nxd=3*self.nx//2-1
        self.nzd=3*self.nz-1

        # set grid
        self.y = np.tanh(self.a*(2*np.arange(-1,self.ny+2)/self.ny-1))/np.tanh(self.a)+1
        self.kx = self.alfa0*np.arange(0,self.nx+1)
        self.kz = self.beta0*np.arange(0,self.nz+1)

        # calculate domain size
        self.lx_pih = 2 / self.alfa0
        self.lz_pih = 2 / self.beta0

        # calculate resolution
        self.dx = self.lx_pih/(2*self.nx) * pi
        self.dz = self.lz_pih/(2*self.nz) * pi
        self.dyw = abs(self.y[1] - self.y[2]) # watch out for ghost cell!
        idxy = floor((self.ny + 3) / 2)
        self.dyc = abs(self.y[idxy] - self.y[idxy+1])