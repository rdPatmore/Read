import numpy  as np
import xarray as xr
import config as paths
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import config
import os

class Read(object):

    def __init__(self, case):
        self.case = case
        self.readPath = config.readPath(case)

    def test_nc_exists(func):
        files = inspect.getargspec(func)[3]
        new_files = {}
        print ('files',files)
        def func_wrapper(self, **kwargs):
            print ('READ PATH', self.readPath)
            l = []
            for k, file in files[0].items():
                l.append(Path(self.readPath + self.maskPath + file).is_file())
                new_files[k] = self.maskPath + file
            if all(l):
                print ('file exists', l)
                pass
            else:
                func(self, **kwargs, files=new_files)
        return func_wrapper

#    def readMeta(self, filename):
#        
#        with open(filename, 'r') as f:
#            lines = f.readlines()  
#        
#        ndim = int(lines[0].split()[3])
#        xdim = int(lines[2].split()[0][:-1])
#        ydim = int(lines[3].split()[0][:-1])
#        prec = str(lines[5].split()[3].strip('\''))
#        
#        if ndim == 2:
#            flddim = int(lines[6].split()[3])
#            increm = 0
#            zdim = 1
#        elif ndim == 3:
#            flddim = int(lines[7].split()[3])
#            self.zdim = int(lines[4].split()[0][:-1])
#            m = 1
#        else:
#        	print("unsupported number of dimensions")
#        return ndim, xdim, ydim, zdim, flddim, prec

    def readMeta(self,filename):
        ''' read a 2 or 3 dimentional MITgcm meta file'''

        #print ('FILENAME ',filename)
    
        meta = {}
        n = 0
    
        with open(filename, 'r') as f:
            lines = f.readlines()
    
        meta['ndims'] = int(lines[0].split()[3])
    
        meta['X'] = int(lines[2].split()[0][:-1])
        meta['Y'] = int(lines[3].split()[0][:-1])
    
        if meta['ndims'] == 3:
            meta['Z'] = int(lines[4].split()[0][:-1])
            n = 1
        else:
            meta['Z'] = 1
    
        meta['dType'] = lines[5 + n].split()[3].strip('\'')
    
        try:
            meta['nFld'] = int(lines[10 + n].split()[3])
            meta['fldList'] = lines[12 + n].replace('\'',' ').split()
        except:
            meta['nFld'] = 1
        #print ('meta ', meta)
        return meta


    def read_binary(self, file, meta=None):
        if meta == None:
            meta = self.readMeta(file.replace('.data', '.meta'))
        if meta['ndims'] == 2:
            file = np.fromfile(file, dtype=meta['dType'],
               count=meta['X']*meta['Y']*meta['nFld']).byteswap()
            if meta['nFld'] > 1:
                bin = file.reshape((meta['nFld'], meta['Y'], meta['X']))
            else:
                bin = file.reshape((meta['Y'], meta['X']))
        if meta['ndims'] == 3:
            file = np.fromfile(file, dtype=meta['dType'],
               count=meta['X']*meta['Y']*meta['Z']*meta['nFld']).byteswap()
            if meta['nFld'] > 1:
                bin = file.reshape((meta['nFld'], meta['Z'], meta['Y'],
                                                  meta['X']))
                # reset z according to ocean coords (-z)
                bin = bin [:,::-1,:,:]
            else:
                bin = file.reshape((meta['Z'], meta['Y'],
                                                  meta['X']))
                # reset z according to ocean coords (-z)
                bin = bin [::-1,:,:]
        return bin, meta

    # 07/11/18
    # binary removed for BL stuff 
    #def readBin(self, file, x=1, y=1, z=1, fld=1, dtype='float64'):
    #    file = np.fromfile(self.readPath + file, dtype=dtype).byteswap()
    #    matrix = np.squeeze(file.reshape((fld,z,y,x)))
    #    return matrix

    def readBin(self, file, x=1, y=1, z=None, dtype='float64'):
        #file = np.fromfile(self.readPath + file,dtype=dtype).byteswap()
        file = np.fromfile(file,dtype=dtype).byteswap()
        print ('FILE SHAPE', file.shape)
        if z == None:
            matrix = file.reshape((y,x))
        else:
            matrix = file.reshape((z,y,x))
        print ('MATRICX SHAPE', matrix.shape)
        return matrix


    def read_box_meta(self, case):
        path = paths.readPath(case) + 'model.meta'
        meta = {}
        with open(path , 'r') as file:
            for line in file:
                info = line.rstrip('\n').replace(' ', '').split('=')
                meta[info[0]] = info[1]
        print (meta)
        return meta

    
    def readMIT(self):
        path = config.readPath('SPBC_717') + 'vels_snap.0000000060.meta'
        data_path =  'vels_snap.0000000060.data'
        self.readMeta(path)
        m = self.readBin(data_path, x=self.xdim, y=self.ydim,
                         z=self.zdim, fld=self.flddim)
        mcori =  m[0]
        print (mcori)
        mx, my = np.meshgrid(np.arange(self.xdim), np.arange(self.ydim))
        plt.figure(1)  
        plt.pcolor(mx, my,mcori)
        plt.show()


    def load_dataset_all(self, chunks=None):
        '''
        Load all NetCDF and .data files in a directory. Loads to a xarray
        dataset. 
        Note: Under construction, not finished
        '''

        nc_files, data_files = [], []
        for root, dirs, files in os.walk(self.readPath):
            for file in files:
                if file.endswith('.nc'):
                    nc_files.append(self.readPath + file)
                if file.endswith('.data'):
                    data_files.append(self.readPath + file.rstrip('.data'))

        # Load netCDF files

        #nc_file = nc_files.pop('state2D.nc')
        
        print ('ncfiles', nc_files)
        try:
            print ('TRY')
            self.ds = xr.open_mfdataset(nc_files, chunks=chunks, 
                                        combine='by_coords')
        except:
            print ('EXCEPT')
            self.ds = xr.open_mfdataset(nc_files, chunks=chunks,
                                        concat_dim='TIME')

        LAT = self.ds.coords['Y'].values
        LON = self.ds.coords['X'].values
        Z   = self.ds.coords['Z'].values
        if Z[0] > Z[-1]:
            Z = Z[::-1]
        
        coordinates = {'Z': Z, 'Y': LAT, 'X': LON}
        #coordinates = {'X': LON, 'Y': LAT, 'Z': Z}
        dimensions  = ['Z','Y', 'X']#[::-1]

        # Combine .data files to nc file xarray dataset
        for file in data_files:
            data, meta = self.read_binary(file + '.data')
            print ('file strip', file.split("/")[-1])
            try:
                dArray = xr.DataArray(data, coords=coordinates, dims=dimensions)
            except Exception as e:
                try:
                    shape = (Z.shape[0], LAT.shape[0], LON.shape[0])
                    #print ('BROADCAST SHAPE', shape)
                    if file.split("/")[-1] == 'DRC':
                        data = data[1:]
                    data = np.broadcast_to(data, shape)
                    print (data.shape)
                    dArray = xr.DataArray(data, coords=coordinates, 
                                                  dims=dimensions)
                except Exception as e:
                    #print (e)
                    #print ('brken3', file)
                    continue
            self.ds = xr.merge([ self.ds, 
                                 dArray.to_dataset(name=file.split('/')[-1])])
        print (self.ds)


    def load_dataset(self, nc_files=None, data_files=None):
        '''
        Load all NetCDF files listed
        '''

        # Load netCDF files
        file_paths = [self.readPath + file for file in nc_files]
        try:
            self.ds    = xr.open_mfdataset(file_paths, combine='by_coords')
        except:
            self.ds    = xr.open_mfdataset(file_paths, concat_dim='TIME')

        if data_files == None: 
            print ('no data')
        else:
            coordinates = {}
            for file in data_files:
                data, meta = self.read_binary(self.readPath + file + '.data')
                dims = {k: meta[k] for k in ('Z', 'Y', 'X')}
                dimensions = [k for (k,v) in dims.items() if v > 1]


                print ('dims', dimensions)
                # check z type
                if 'new_Z' in self.ds.coords.keys():
                    dimensions = ['new_Z'] + dimensions
                    dimensions.remove('Z')
                print ('self', self.ds)
                print ('dims', dimensions)
                print ('meta', meta)


                coordinates = self.ds[dimensions]
                
                if file.split("/")[-1] == 'DRC':
                    data = data[1:]

                dArray = xr.DataArray(np.squeeze(data), coords=coordinates,
                                                        dims=dimensions)
                print (file, dArray)

                self.ds = xr.merge([self.ds, dArray.to_dataset(name=file)])

                if 'new_Z' in self.ds.coords.keys():
                    # assert ascending Z
                    self.ds = self.ds.sortby('new_Z', ascending=True)
                else:
                    # assert ascending Z
                    self.ds = self.ds.sortby('Z', ascending=True)

        # below is post SPBC stuff, uncomment for SPBC i.e postProcess.py 
        
        #for file in data_files:
        #    data, meta = self.read_binary(self.readPath + file + '.data')
        #    data = data[0]
        #    dArray  = xr.DataArray(data, coords=coordinates, dims=dimensions)
        #    self.ds = xr.merge([ self.ds, dArray.to_dataset(name=file)])
        
        # Add a 2D/3D DRF field - why is this necesarry?
        #depths  = np.full(self.ds['DXC'].shape, 5000.0) 
        #drf     = xr.DataArray(depths, coords=coordinates, dims=dimensions)
        #self.ds = xr.merge([ self.ds, drf.to_dataset(name='DRF')])


    def momentum_jig(self):
        self.ds = self.ds.rename({'UBotDrag': 'u_tauB',
                                  'Um_Ext':   'u_tauW',
                                  'Um_Cori':  'u_cori',
                                  'SDIAG9':   'u_etaGrad',
                                  'AB_gU':    'u_AB',
				  'VBotDrag': 'v_tauB',
                                  'Vm_Ext':   'v_tauW',
                                  'Vm_Cori':  'v_cori',
                                  'SDIAG10':  'v_etaGrad',
                                  'AB_gV':    'v_AB'})
 
        self.ds['u_viscH'] = self.ds['Um_Diss']-self.ds['u_tauB']
        self.ds['v_viscH'] = self.ds['Vm_Diss']-self.ds['v_tauB']
        

    def load_additional_ncfile(self, nc_file):
        '''
        Load an addisional NetCDF file and merge to current dataset
        '''
        
        array = xr.open_dataset(self.readPath + nc_file)
        self.ds = xr.merge( [self.ds, array] )
