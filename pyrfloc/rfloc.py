import numpy as np
import json

class RFloc3D(object):
    """
    Class representing the RFloc3D method
    
    Written by Oct 5, 2022
    """
    def __init__(self, 
                fpicks,                #file of Picked traveltimes
                nevent=10000,          #number of synthetic events
                mlon=None,			   #lon limit (min and max)
                mlat=None, 			   #lat limit (min and max)
                mdep=None,			   #dep limit (min and max)
                nlon=101,			   #number of grid samples in lon
                nlat=101,			   #number of grid samples in lat
                ndep=101, 			   #number of grid samples in dep
                nsta=None):    		   #number of stations in the area of interest
        """
        
            nns = RFloc3D(
                fpicks='stations_times.txt',
                nevent=10000,nst=9)
                
        
        """
        
        f = open(fpicks,'r');
        lines=f.readlines();
        
        lons=[]
        lats=[]
        ptimes=[]
        for iline,line in enumerate(lines):
            line=line.rstrip("\n").split(" ")
            print(iline,line)
            lons.append(float(line[0]))
            lats.append(float(line[1]))
            ptimes.append(float(line[2]))
            
            
    
        
        if nsta == None:
            nsta=9;
        self.nsta=nsta;
        self.nlon=nlon;
        self.nlat=nlat;
        self.ndep=ndep;
        self.nx=self.nlon;
        self.ny=self.nlat;
        self.nz=self.ndep;
        
        self.nevent = nevent
        self.vp = None
        self.vs = None
        self.model = None
        
        lons=lons[0:nsta]
        lats=lats[0:nsta]
        ptimes=ptimes[0:nsta]
        
        self.lons=lons;
        self.lats=lats;
        self.ptimes=ptimes;
        
        if mlon == None:
            minlon=min(lons)
            maxlon=max(lons)
        else:
            minlon=mlon[0]
            maxlon=mlon[1]
        
        if mlat == None:
            minlat=min(lats)
            maxlat=max(lats)
        else:
            minlat=mlat[0]
            maxlat=mlat[1]
            
        if mdep == None:
            mindep=0
            maxdep=25
        else:
            mindep=mdep[0]
            maxdep=mdep[1] 
        
        print('minlon,maxlon:',minlon,maxlon)
        print('minlat,maxlat:',minlat,maxlat)
        
        #In Cartesian coordinates
        minx=0;miny=0;minz=0;
        maxx=(maxlon-minlon)*111.1949
        maxy=(maxlat-minlat)*111.1949
        maxz=maxdep;
        
        dx=(maxx-minx)/(self.nx-1);
        dy=(maxy-miny)/(self.ny-1);
        dz=(maxz-minz)/(self.nz-1);
        
        print('minx,maxx:',minx,maxx)
        print('miny,maxy:',miny,maxy)
        print('minz,maxz:',minz,maxz)

        print('dx:',dx)
        print('dy:',dy)
        print('dz:',dz)
        
        self.minlon=minlon
        self.maxlon=maxlon
        self.minlat=minlat
        self.maxlat=maxlat
        
        self.minx=minx
        self.maxx=maxx
        self.miny=miny
        self.maxy=maxy
        self.minz=minz
        self.maxz=maxz
        
        self.dx=dx
        self.dy=dy
        self.dz=dz
        
        ## coordinates transformation
        self.lons_x=(np.array(lons)-minlon)*111.1949 #station longitudes in X
        self.lats_y=(np.array(lats)-minlat)*111.1949 #station latitudes in Y
        
        
        
    def load_vel(self,vpfile=None,vsfile=None):
        '''
        Load velocity
        '''


        fd = open(vpfile,'rb')
        vp = np.fromfile(fd, dtype = np.float32) 
        vp=vp.reshape([self.nz,self.nx,self.ny],order='F') #zxy
        
        self.vp=vp
        
        return vp


    
    def calc_traveltime(self,shots):
        '''
        Calculate traveltime using pyekfmm
        
        A substitute of load_traveltime
        '''
        import pyekfmm as fmm
        
#         vel=np.swapaxes(vp,0,2).reshape([101*101*101,1],order='F'); #transpose to [xyz]
        # t=fmm.eikonal_surf(vel,xyz=shots[0:4],ax=[0,0.9413,101],ay=[0,0.8031,101],az=[0,0.25,101],order=1) #xy-nevets 
        # fmmtime=t.reshape(101,101,nevent,order='F'); #[x,y]
        # memory issue stopped
        
        vp=self.vp; #zxy
        vp=np.swapaxes(vp,0,2).reshape([self.nx*self.ny*self.nz,1],order='F'); #transpose to [xyz]
        
        fmmtimes=np.zeros([self.nx,self.ny,self.nevent],dtype='float32') #x,y,nevent
        for ii in range(self.nevent):
            fmmtimes[:,:,ii]=fmm.eikonal_surf(vp,
                                              xyz=np.expand_dims(shots[ii,:],0),
                                              ay=[self.minx,self.dx,self.nx],
                                              ax=[self.miny,self.dy,self.ny],
                                              az=[self.minz,self.dz,self.nz],
                                              order=1,verb=0).reshape(self.nx,self.ny,order='F');
            #a little inconsistent here (due to C-version X-Y inconsistency)
            if np.mod(ii,50)==0:
                print('event id=',ii,' is done\n');
        self.fmmtimes=fmmtimes;
        
    def load_traveltime(self,tfile='./times.bin'):
        '''
        Load traveltime from disk
        
        A substitute of calc_traveltime
        
        INPUT
        tfile: traveltime file name (binary)
        
        OUTPUT
        
        '''
        
        fd = open(tfile,'rb')
        self.stimes = np.fromfile(fd, dtype = np.float32) #synthetic time

        ##read event labels (100000)
        nevent=self.nevent
        nx=self.nx
        ny=self.ny
        self.stimes=self.stimes.reshape([nx,ny,nevent],order='F');
        
        return self.stimes
        
    def load_events(self,evfile='./locations.txt'):
        '''
        Load event locations from disk
        
        A substitute of create_events
        
        INPUT
        evfile: event file name (ASCII)
        
        OUTPUT
        
        '''
        f=open(evfile,'r');
        self.evlabels=[line.rstrip("\n").split(" ") for line in f.readlines()];
        self.evlabels=np.array(self.evlabels,dtype='float32').transpose()
        
        return self.evlabels
        
    def prepare_data(self):
        '''
        prepare the data for on stations
        '''
        print(self.lons_x)
        print(self.lats_y)
        dx=self.dx
        dy=self.dy
        dz=self.dz
        print(dx,dy,dz)

        indx=np.int16(self.lons_x/dx)
        indy=np.int16(self.lats_y/dy)
        print(indx)
        print(indy)

        lons_x2=indx*dx
        lats_y2=indy*dy
        data=np.zeros([3,self.nsta,self.nevent])
        for ii in range(self.nevent):
            for jj in range(self.nsta):
                data[0,jj,ii]=lons_x2[jj] #use grid position
                data[1,jj,ii]=lats_y2[jj] #use grid position
                data[2,jj,ii]=self.stimes[indx[jj],indy[jj],ii]
        for ii in range(self.nevent):
            ind=np.argsort(data[2,:,ii])
            data[:,:,ii]=data[:,ind,ii]
    
        self.data=data
        
        return data
        
    def prepare_train_data(self):
        '''
        prepare the training data (differential traveltime) for random forest model
        '''
        # Reading Input Lat and Long for the Stations and Events.
        lab = self.evlabels
        lat=[]
        long=[]
        timdif=[]
        laball=[]
        loc=[]
        depall=[]
        lad = []
        lod = []
        for i in range(0,np.shape(self.data)[-1]):
            # Time difference betweeen the first arrival time and the 9 stations.
            t = self.data[2,:,i] - self.data[2,0,i] 
            t = t[1:]
            timdif.append(t)
            
            # The target is the difference between the lat/long of the first station 
            # and the excat lat/long of the event. The depth is the excat value.
            lad.append(lab[0,i] - self.data[0,0,i])
            lod.append(lab[1,i] - self.data[1,0,i])
            
            # The lat/long of the 10 stations
            lat.append(self.data[0,:,i])
            long.append(self.data[1,:,i])
    
            # The lat/long and depth of the events.
            laball.append(lab[:,i])
            depall.append(lab[2,i])
    
        lat    = np.array(lat)
        long   = np.array(long)
        timdif = np.array(timdif)
        laball = np.array(laball)
        depall = np.array(depall)
        lad = np.array(lad)
        lod = np.array(lod)

        lad    = np.reshape(lad,(lad.shape[0],1))
        lod   = np.reshape(lod,(lod.shape[0],1))
        depall = np.reshape(depall,(depall.shape[0],1))

        # Concatenating the input features.
        p_all = np.concatenate([lat,long,timdif],axis=-1)
        
        # We randomly split the data, so I saved the random index for reproducing and comparing.
        np.random.seed(2021)
        indrand=np.random.permutation(self.nevent)
        # Reorder according to the randomaization.
        p_all = p_all[indrand]
        d = np.concatenate([lad,lod,depall],axis=-1) 
        d = d[indrand]
        
        np.shape(p_all)
        
        self.p_all=p_all;
        self.d=d;
        self.laball=laball;
        self.indrand=indrand;
        
        return p_all,d

    def train(self):
        '''
        train a random forest model
        '''
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(max_depth=1000,
               max_features=26,
               n_estimators=1000,
               oob_score=False,
               random_state=43,
               verbose=1,
               criterion='mse',
               n_jobs=-1)
        # Training RF using 90% of the data.
        ix = int(len(self.p_all)*0.9)
        self.model.fit(self.p_all[0:ix],self.d[0:ix])  
        
    def train_check(self):
        '''
        check training result
        '''
        indrand=self.indrand;
        ix = int(len(self.p_all)*0.9);
        
        # Predicting the difference lat/long and excat depth.
        outtrain = self.model.predict(self.p_all[0:ix])

        # Retriving the excat lat/long
        outtrain[:,0] = self.data[0,0,indrand][0:ix] + outtrain[:,0]
        outtrain[:,1] = self.data[1,0,indrand][0:ix] + outtrain[:,1]
        dx = self.laball[indrand][0:ix]

        # Error estimatin (MAE)
        errlattrain  = np.mean(np.abs( (outtrain[:,0]) - (dx[:,0]) ))
        errlongtrain = np.mean(np.abs( (outtrain[:,1]) - (dx[:,1]) ))
        errsepthtrain = np.mean(np.abs( (outtrain[:,2]) - (dx[:,2]) ))

        print('Error in lat,lon,dep:',errlattrain,errlongtrain,errsepthtrain)
        
    
    def save_model(self,filename='./models/model.joblib'):
        """
        Save model to disk
        """ 
        import joblib
        joblib.dump(self.model, filename) 
		
    def load_model(self,filename='./models/model.joblib'):
        """
        Load model from disk
        """ 
        import joblib
        self.model = joblib.load(filename)
        
    
    def plot_stations(self,evs=None):
        """
        plot stations used in the RFloc3D class
        
        evs: event list of lon,lat,dep (lon,lat,dep can be a list for each)
        """ 
        
        import matplotlib.pyplot as plt
        plt.plot(self.lons,self.lats,'v',color='b',markersize=15)
        
        if evs is not None:
            plt.plot(evs[0],evs[1],'*',color='r',markersize=15)
        
        plt.show()
        
    def plot_vel(self):
        """
        plot vel used in the RFloc3D class
        """ 
        
        import matplotlib.pyplot as plt
        
        ii=int(self.ny/2)
        fig = plt.figure(figsize=(12, 8))
        ax=plt.imshow(self.vp[:,:,ii],extent=[self.minx,self.maxx,self.maxz,self.minz],aspect='auto');
        plt.jet();
        plt.xlabel('X (km)');plt.ylabel('Z (km)');
        plt.colorbar(orientation='horizontal',shrink=0.6,label='Velocity (km/s)');
        plt.show()

    def apply(self):
        """
        Apply trained model to picked arrivals
        """ 
        ptimes=np.array(self.ptimes)
        inds=np.argsort(ptimes)
        ptimes=ptimes[inds]
        lons_x=self.lons_x[inds]
        lats_y=self.lats_y[inds]
        
        t_all = np.concatenate([lons_x,lats_y,ptimes[1:]-ptimes[0]],axis=-1) #here use grid positions of stations
        t_all=np.expand_dims(t_all, axis=0)
        
        outtest_xy = self.model.predict(t_all)

        # Retriving the excat lat/long
        outtest_xy[:,0] = lons_x[0] + outtest_xy[:,0] #use grid positions of stations (after binning)
        outtest_xy[:,1] = lats_y[0] + outtest_xy[:,1] #use grid positions of stations (after binning)

        outtest=outtest_xy
        outtest[:,0]=self.minlon+outtest[:,0]/111.1949
        outtest[:,1]=self.minlat+outtest[:,1]/111.1949

        print(outtest)
        print('Predicted',outtest[:,0],outtest[:,1],outtest[:,2])
        print('catalog:',-104.05,31.7,7.1)
        
        self.outtest=outtest;
        
    def plot_result(self,evs=None):
        """
        plot result
        
        evs: event list of lon,lat,dep (lon,lat,dep can be a list for each)
        """ 
        
        import matplotlib.pyplot as plt

        plt.plot(self.lons,self.lats,'v',color='b',markersize=15)
        if evs is not None:
            plt.plot(evs[0],evs[1],'*',color='r',markersize=15)
        plt.plot(self.outtest[:,0],self.outtest[:,1],'p',color='g',markersize=15)
        
        plt.show()
        
        
        
		

