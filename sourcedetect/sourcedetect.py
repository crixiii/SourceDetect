import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

import keras
import keras.ops as ops
import tensorflow as tf
from photutils.aperture import CircularAperture as CA

import os

from .prfbuild import PrfBuild
from .prfmodel import PrfModel



class SourceDetect:
    """Performs object detection and analysis on a set of real TESS images"""

    def __init__(self,flux,Xtrain='default',ytrain='default',savepath=None,model='default',train=False,run=False,do_cut=False,
                 precheck=False,batch_size=32,epochs=50,validation_split=0.1,optimizer=tf.keras.optimizers.Adam,learning_rate=0.003,
                 metrics=["categorical_accuracy"],monitor='loss'):
        """
        Initialise
        ------
        Parameters
        ------
        flux : array or str
            TESS image (array of fluxes): either the array itself or its filepath to its file; the flux array must be 2D with dimensions (y,x) or 3D with dimensions (n_frames,y,x)
        Xtrain : str or array (default 'default')
            TESS prf arrays (npy file) to be added into the training/test sets; if 'default' then load premade set  
        ytrain : str or array (default 'default')
            labels of the TESS prf arrays (npy file) to be added into the training/test sets; if 'default' then load premade labels (positive/negative sources can either share a label or have different labels)
        savepath : str or None (default None)
            location to save any output models, tables, and figures will be saved
        model : str or keras.Model (default 'default')
            ML model or model savepath; if 'default' then a prebuilt model is defined
        train : bool (default False)
            if true then the ML model is trained on the Xtrain dataset (only use if calling an untrained custom model)
        run : bool (default False)
            if True then the entire source detection and analysis pipeline is run with default settings
        do_cut : bool (default False)
            if True then the TESS images are cut via a user prompt to specify the desired region to keep
        precheck : bool (default False)
            if True then display the first frame prior to object detection to check for any formatting issues 
        batch_size : int (default 32)
            batch size used when training the ML model 
        epochs : int (default 50)
            number of training epochs performed by the ML model 
        validation_split : float (default 0.1)
            the ratio of training set images used in the validation set
        optimizer : tensorflow.keras.optimizers instance (default Adam)
            tensorflow optimiser method to be used by the ML model
        learning_rate : float (default 0.003)
            learning rate during the training process
        metrics : list (default ["categorical_accuracy"])
            list of metrics (strings and/or keras.metrics.Metric instances) to be evaluated by the model during training/testing (using tensorflow naming conventions)
        monitor : str (default 'loss')
            name of metric to monitor during training (using tensorflow naming conventions)
        """
        if type(flux) == str:
            flux = np.load(flux)
        try:
            type(flux) == np.ndarray
        except:
            raise TypeError('flux has to be either a numpy array or a string (filename/path of a numpy array)')
        self.flux = flux
        if savepath == None:
            self.savepath = '.'
        else:
            self.savepath = savepath
        self.directory = os.path.dirname(os.path.abspath(__file__)) + '/'
        if type(model) == str:
            if model == 'default':
                model = self.directory+'default_model.keras'
            self.model = keras.saving.load_model(model,compile=False)
        else:
            self.model = model
        self.precheck = precheck
        self.train = train

        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.metrics = metrics
        self.issues = False

        if Xtrain == 'default':
            self.Xtrain = np.load(self.directory+'training_data.npy',allow_pickle=True)
            self.ytrain = np.load(self.directory+'training_labels.npy',allow_pickle=True)
        else:
            self.Xtrain = Xtrain
            self.ytrain = ytrain
            if type(Xtrain) == str:
                self.Xtrain = np.load(self.Xtrain,allow_pickle=True)
            if type(ytrain) == str:
                self.ytrain = np.load(self.ytrain,allow_pickle=True)

        if len(self.flux.shape) == 2:
            self.flux = np.expand_dims(self.flux,0)

        if run == True:
            self.SourceDetect(train=train,do_cut=do_cut)


    def cut(self,xrange=None,yrange=None):
        """Cut the TESS images down to a specified subregion or via a user prompt 
        ------
        Parameters
        ------
        xrange : None or tuple (default None)
            upper and lower bound for the x-axis of the image cut; if None then this is defined by a user prompt
        yrange : None or tuple (default None)
            upper and lower bound for the y-axis of the image cut; if None then this is defined by a user prompt
        """
        plt.figure()
        if yrange == None or xrange == None:
            ranges = input('Enter the bounds for the cut in the form "ymin,ymax,xmin,xmax": ').split(',')
            yrange, xrange = [int(ranges[0]),int(ranges[1])], [int(ranges[2]),int(ranges[3])]

        plt.imshow(self.flux[0,yrange[0]:yrange[1],xrange[0]:xrange[1]],cmap='gray')
        plt.colorbar()
        plt.show()

        confirm = ':)'
        while confirm.upper() not in ('YES',''):
            confirm = input('Would you like to make the above cut? Type "yes" to confirm or leave blank to cancel: ')
        if confirm == "yes":
            self.flux = self.flux[:,yrange[0]:yrange[1],xrange[0]:xrange[1]]
        

    def apply_model(self,train=False):
        """Apply the desired model to perform object detection on the image. This funciton can also build, compile and train a ML model using find_sources.PrfModel if desired
        ------
        Parameters
        ------
        train : bool (default False)
            if True then build and train the model with find_sources.PrfModel
        """
        if self.train == True or train == True:
            _model = PrfModel(self.Xtrain,self.ytrain,savepath=self.savepath,model=self.model,loss_func='default')
            self.model = _model.model
            self.flux = np.expand_dims(self.flux,-1)
     
        #Prevents issue where model doesn't like datasets with shapes different to the training set:
        _ = self.model.predict(np.ones((1,16,16,1)))

        print('Applying model:')
        if len(self.flux.shape) == 3:
            self.flux = np.expand_dims(self.flux,-1)
        self.y = self.model.predict(self.flux*np.ones(self.flux.shape))


    def close_detect(self):
        """Identifies any groups of object detections that are in close proximity to one another (potential double detections or close sources)
        ------
        Outputs
        ------
        close_sources : list
            positions (as tuples) of detections within close proximity to each other across all images
        close_sources_by_frame : list
            lists of positions (as tuples) of detections within close proximity to each other detected per image
        """
        self.close_sources, self.close_sources_by_frame = [], []

        for s in range(0,len(self.sources_by_frame)):
            close_objs = [[] for _ in range(len(self.sources_by_frame[s]))]

            for i in range(0,len(self.sources_by_frame[s])):
                for j in range(0,len(self.sources_by_frame[s])):
                    if i != j:
                        if np.abs(self.sources_by_frame[s][i][0]-self.sources_by_frame[s][j][0]) <= 4 and np.abs(self.sources_by_frame[s][i][1]-self.sources_by_frame[s][j][1]) <= 4:
                            close_objs[i].append(self.sources_by_frame[s][j])

            _close_sources = []
            for i in range(0,len(self.sources_by_frame[s])):
                for j in range(0,len(close_objs)):
                    if i != j:
                        if self.sources_by_frame[s][i] in close_objs[j]:
                            for k in range(0,len(close_objs[i])):
                                if close_objs[i][k] not in close_objs[j]:   
                                    close_objs[j].append(close_objs[i][k])

            for i in range(0,len(close_objs)):
                if len(close_objs[i]) > 0 and sorted(close_objs[i]) not in _close_sources:
                    _close_sources.append(sorted(close_objs[i]))

            for i in range(0,len(_close_sources)):
                if len(_close_sources[i]) > 0:
                    self.close_sources.append(_close_sources[i])

            self.close_sources_by_frame.append(_close_sources)


    def unique_detect(self):
        """Identifies all guaranteed unique detections in each TESS image (detections not in close proximity to any others plus one detection from each grouping from SourceDetect.close_detect) 
        ------
        Outputs
        ------
        unique_sources : list
            list of positions (as tuples) of every guaranteed unique detection in each image
        """
        try:
            self.close_sources_by_frame = self.close_sources_by_frame
        except:
            self.close_detect()

        self.unique_sources = []
        for s in range(0,len(self.sources_by_frame)):
            if len(self.close_sources_by_frame[s])>0:
                _unique_sources = []

                for i in range(0,len(self.sources_by_frame[s])):
                    add = True

                    for j in range(0,len(self.close_sources[s])):
                        if self.sources_by_frame[s][i] in self.close_sources_by_frame[s][j][1:]:
                            add = False

                    if add == True:
                        _unique_sources.append(self.sources_by_frame[s][i])

                self.unique_sources.append(_unique_sources)

            else:
                self.unique_sources.append(self.sources_by_frame[s])


    def detect(self,threshold=0.8,grid_size=4,close=True,unique=False):
        """Documents all potential sources detected by the ML model in each image, their potential variability and whether they are in close proximity to any other detections
        ------
        Parameters
        ------
        threshold : float (default 0.8)
            minimum probability of being a true source (according to the ML model) required for detections to be boxed in the plot
        grid_size : int (default 4)
            ratio between image size and mask size (pixel area of potential detection sites)
        close : bool (default True)
            if True then document all grouping of detections in each frame in close proximity to other detections (keep True if you want analysis tables to work)
        unique : bool (default False)
            if True then document all guaranteed unique detections (detections not in close proximity to any others plus one detection from each grouping from SourceDetect.close_detect)
        ------
        Outputs
        ------
        sources : list
            positions (as tuples) of every source detected in every image
        sources_by_frame : list
            lists of positions (as tuples) of every source detected per image
        to_plot : list
            list of tuples that can be used to plot detection results
        num_sources : list
            number of sources detected in each image
        frames : list
            the frame (image index) each detection is found in (list of equal length to sources)
        sourceID : dictionary
            pairs each unique detection (by position) to a unique ID number
        flux_sign : list
            indicates whether each detection has positive or negative flux
        variable_flag : dictionary
            documents whether each unique detection (by position) was flagged for variability
        """
        self.sources, self.sources_by_frame = [], []
        self.to_plot = []
        self.num_sources, self.frames = [], []
        self.flux_sign, self.variable_flag, variable_flag_counter = [], {}, {}

        print('Performing object detection:')
        for a in range(0,len(self.flux)):
            i, j = self.y[a].shape[0], self.y[a].shape[1]
            numb_sources = 0
            positions = []
            to_plot_ = []
            for mx in range(j):
                for my in range(i):
                    channels = self.y[a][my][mx]
                    prob, x1, y1, x2, y2, bright, dim, trash, fake = channels

                    #Ignore all detections with low probability and those likely to be 'trash' or 'fake' sources
                    if prob < threshold:
                        continue
                    if fake > 0.8:
                        continue
                    if bright < 0.6 and dim < 0.6:
                        if bright < 0.5 and dim < 0.5:
                            continue
                        elif fake > 0.2:
                            continue
                    if trash > bright and trash > dim:
                        continue
                    if fake > bright and fake > dim:
                        #For 'close calls' which are usually genuine detections
                        if fake - bright > 0.2 or fake - dim > 0.2:
                            continue
                    if abs(x2) > 6 or abs(y2) > 6:
                        continue

                    px, py = (mx * grid_size) + x1, (my * grid_size) + y1
                    while int(py) < 2:
                        py += 2
                    while int(px) < 2:
                        px += 2
                    while int(py) > self.flux.shape[1]-3:
                        py -= 2
                    while int(px) > self.flux.shape[2]-3:
                        px -= 2

                    #Final filters for any obvious false detections on background fluctuations or overexposures
                    if self.flux[a][int(py),int(px)] > -1.5 and self.flux[a][int(py),int(px)] < 1.5:
                        continue 
                    if np.max(self.flux[a][int(py)-2:int(py)+3,int(px)-2:int(px)+3]) > 5 and np.min(self.flux[a][int(py)-2:int(py)+3,int(px)-2:int(px)+3]) < -5:
                        continue

                    numb_sources += 1
                    smax = np.where(np.abs(self.flux[a][int(py)-1:int(py)+2,int(px)-1:int(px)+2,0])==np.max(np.abs(self.flux[a][int(py)-1:int(py)+2,int(px)-1:int(px)+2,0])))
                    # smax = np.where(np.abs(self.flux[a][int(py-y2/2):int(py+y2/2+1),int(px-x2/2):int(px+x2/2+1),0])==np.max(np.abs(self.flux[a][int(py-y2/2):int(py+y2/2+1),int(px-x2/2):int(px+x2/2+1),0])))
                    smax_i = (int(py)+smax[0][0]-1,int(px)+smax[1][0]-1)
                    if smax_i not in positions:
                        to_plot_.append((prob,smax_i[1],smax_i[0],x2,y2))
                        positions.append(smax_i)
                        self.sources.append(smax_i)
                        self.frames.append(a)

                        if smax_i not in self.variable_flag:
                            self.variable_flag[smax_i] = 1*(bright>dim)
                            variable_flag_counter[smax_i] = 0
                        else:
                            if self.variable_flag[smax_i] != 1*(bright>dim):
                                variable_flag_counter += 1
                                self.variable_flag[smax_i] = 1*(bright>dim)

                        if self.flux[a][smax_i] > 0:
                            self.flux_sign.append('positive')
                        else:
                            self.flux_sign.append('negative')
            self.to_plot.append(to_plot_)
            self.sources_by_frame.append(sorted(positions))
            self.num_sources.append(numb_sources)

        for i in self.variable_flag.keys():
            if variable_flag_counter[i] >= 5:
                self.variable_flag[i] = True
            else:
                self.variable_flag[i] = 'unsure'
            if type(self.variable_flag[i]) != bool:
                self.variable_flag[i] = False
            self.variable_flag = self.variable_flag

        self.sourceID = {}
        id = 0
        for s in self.sources:
            if s not in self.sourceID:
                self.sourceID[s] = id
                id += 1

        if close == True:
            self.close_detect()
        if unique == True:
            self.unique_detect()
        
        print('Object detection complete')


    def get_flux(self,analyse=False,position=None,frame=None):
        """Creates a list of the fluxes of each potential source detected by the model or provides the flux at a specified position
        ------
        Parameters
        ------
        analyse : bool (default False)
            if True then document flux or every detection in every image
        position : None or tuple (default None)
            if analyse is False then this specifies the position of the detection
        frame : None or int (default None)
            if analyse if False then this specifies the frame (image index) with the detection
        ------
        Outputs
        ------
        source_flux : list
            flux of every detection in every image (only output if analyse is True)
        """
        if analyse == True:
            self.source_flux = []
            for s in range(0,len(self.sources)):
                # ind = np.where(np.array(self.to_plot[self.frames[s]])[1:3] == [self.sources[s][1],self.sources[s][0]])[0]
                # aper = RA(positions=(self.sources[s]),w=int(self.to_plot[self.frames[s]][ind][3]+1),h=int(self.to_plot[self.frames[s]][ind][4]+1))
                # self.source_flux.append(aper.do_photometry(self.flux[self.frames[s]][:,:,0])[0][0])
                self.source_flux.append(np.sum(self.flux[self.frames[s]][self.sources[s][0]-1:self.sources[s][0]+2,self.sources[s][1]-1:self.sources[s][1]+2]))

        else:
            try:
                print(f'Flux at position {position} in frame{frame}: {self.flux[frame,position[0],position[1]]}')
            except:
                print('Please specify the position and frame of the object when calling this method')


    def group_and_id(self):
        """Identify the full groups of detections in close proximity across all images, assign each group a number, 
           and document the number of detecitons at each (x,y) position across all images
        ------
        Outputs
        ------
        groups : list
            lists of positions (as tuples) of detections in close proximity to each other across all frames 
        groupID : dictionary
            mapping between each potential detection and a group ID number (ID of -1 for detections not in a group)
        n_detections : dictionary
            mapping between the position of each potential detected source and the number of detections across all images    
        """
        def find(parent, x):
            """Subprocess of SourceDetect.group_and_id.group"""
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        def union(parent, rank, x, y):
            """Subprocess of SourceDetect.group_and_id.group"""
            rootX = find(parent, x)
            rootY = find(parent, y)
            if rootX != rootY:
                if rank[rootX] > rank[rootY]:
                    parent[rootY] = rootX
                elif rank[rootX] < rank[rootY]:
                    parent[rootX] = rootY
                else:
                    parent[rootY] = rootX
                    rank[rootX] += 1

        def group(close_sources):
            """Similar to SourceDetect.close_detect but across all images to obtain complete groupings of potential sources
            ------
            Parameters
            ------
            close_sources : list
                lists of positions (as tuples) of detections in close proximity to each other in each frame separately
            ------
            Returns
            ------
            result : list
                lists of positions (as tuples) of detections in close proximity to each other across all frames 
            """
            parent = {}
            rank = {}

            for sublist in close_sources:

                for coord in sublist:
                    if coord not in parent:
                        parent[coord] = coord
                        rank[coord] = 0

                for i in range(len(sublist) - 1):
                    union(parent, rank, sublist[i], sublist[i+1])

            grouped = {}
            for sublist in close_sources:
                root = find(parent, sublist[0])

                if root not in grouped:
                    grouped[root] = set()
                grouped[root].update(sublist)

            result = [list(group) for group in grouped.values()]
            return result
        
        def IDassign(groups,positions):
            """Assign a group (close proximity detections) ID number, or lack thereof, to every detections across all images
            ------
            Parameters
            ------
            groups : list
                list of lists containing positions of detections in close proximity to each other across all images
            positions : list 
                positions (as tuples) of every detection across all images
            ------
            Returns
            ------
            result_dict : dictionary
                mapping between each potential detection and a group ID number (ID of -1 for detections not in a group)
            """
            id_map = {tuple(sorted(group)): idx for idx, group in enumerate(groups)}
            position_to_id = {}

            for group, group_id in id_map.items():
                for coord in group:
                    position_to_id[coord] = group_id

            result_dict = {}
            for coord in positions:
                result_dict[coord] = position_to_id.get(coord, -1)

            return result_dict

        def get_num_detections(detections):
            """Determines the number of detections for potential sources across each image
            ------
            Parameters
            ------
            detections : list 
                positions (as tuples) of every detection across all images
            ------
            Returns
            ------
            n_detections : dictionary
                mapping between the position of each potential detected source and the number of detections across all images    
            """
            n_detections = {}
            for i in detections:
                if i in n_detections:
                    n_detections[i] += 1
                else:
                    n_detections[i] = 1
            return n_detections

        self.groups = group(self.close_sources)
        self.groupID = IDassign(self.groups,self.sources)
        self.n_detections = get_num_detections(self.sources)


    def preview(self,frame=0,do_cut=False):
        """Plots a specified image prior to object detection; can also apply a cut to the images via SourceDetect.cut
        ------
        Parameters
        ------
        frame : int (default 0)
            frame (image index) to be plotted 
        do_cut : bool (default False)
            if True then provide user prompt to apply a cut to the images via SourceDetect.cut 
        ------
        Outputs
        ------
        issues : bool
            if True then there are user identifies issues with the flux file and the object detection process will be cancled
        """
        plt.figure()
        plt.imshow(self.flux[frame],vmin=-10,vmax=10)
        plt.show()
        issues = ''
        while issues.lower() not in ['yes','no']:
            input('Are there any issues with the image below (reply "yes" or "no"): ')
        if issues.lower() == 'yes':
            self.issues = True
        if do_cut == True:
            cut = input('Would you like to make a cut to the images (type "yes" to cut): ')
            if cut.lower() == 'yes':
                self.cut()


    def resultdf(self,update=False):
        """Creates output tables summarising the full object detection process
        ------
        Parameters
        ------
        update : bool (default False)
            if True then update the events table rather than redefining both (only used from with in SourceDetect.combine_groups)
        ------
        Outputs
        ------
        result : pd.Dataframe
            summary table of detections across all images
        events : pd.Dataframe
            summary table of potential sources (detections unique to each position) across all images
        """
        if update == False:
            print('Collecting results:')
            self.result = pd.DataFrame(data={'x_centroid':np.array(self.sources)[:,1],'y_centroid': np.array(self.sources)[:,0],'flux': self.source_flux,'frame':self.frames})
            self.result['n_detections'] = self.result.apply(lambda row:self.n_detections[(row['y_centroid'],row['x_centroid'])],axis=1) 
            self.result['objid'] = self.result.apply(lambda row:self.sourceID[(row['y_centroid'],row['x_centroid'])],axis=1)
            self.result['group'] = self.result.apply(lambda row:self.groupID[(row['y_centroid'],row['x_centroid'])],axis=1)
            self.result['flux_sign'] = self.flux_sign
            self.result.drop(self.result[(self.result.flux>0) & (self.result.flux_sign=='negative')].index)
            self.result.drop(self.result[(self.result.flux<0) & (self.result.flux_sign=='positive')].index)

        self.events = self.result.drop_duplicates(subset='objid').drop(columns=['flux','frame','flux_sign']).reset_index()
        self.result['abs_target'] = self.result['flux'].abs()
        self.events['variability'] = self.events.apply(lambda row:self.variable_flag[(row['y_centroid'],row['x_centroid'])],axis=1)

        max_flux = self.result.groupby('objid')['abs_target'].idxmax().to_dict()
        self.result = self.result.drop(columns=['abs_target'])
        self.events['max_flux'] = self.events.apply(lambda row:max_flux[row['objid']],axis=1)

        framei,framef = self.result.groupby('objid')['frame'].min().to_dict(),self.result.groupby('objid')['frame'].max().to_dict()
        self.events['start_frame'] = self.events.apply(lambda row:framei[row['objid']],axis=1)
        self.events['end_frame'] = self.events.apply(lambda row:framef[row['objid']],axis=1)
        self.events = self.events.drop(columns=['index'])

        self.result.to_csv(f'{self.savepath}/detected_sources')
        self.events.to_csv(f'{self.savepath}/detected_events')

    
    def combine_groups(self):
        """Reduce the result and events tables from SourceDetect.resultdf to only include the detection/source from each group with the largest flux 
           (shows the minimum unique detections across all images)
        ------
        Outputs
        ------
        result : pd.Dataframe
            summary table containing positions, flux, object ID, etc of all guaranteed unique detections across all images
        events : pd.Dataframe
            summary table containing positions, flux, frame duration, etc of all guaranteed unique potential sources (detections unique to each position) across all images
        """    
        self.result['abs_target'] = self.result['flux'].abs()
        mask = self.result['group'].isin([-1])
        excluded = self.result[mask]
        included = self.result[~mask]

        idx = included.groupby(['group','frame'])['abs_target'].idxmax()
        filtered = self.result.loc[idx]
        df_result = pd.concat([filtered, excluded])
        df_result = df_result.drop(columns=['group','abs_target'])

        self.result = df_result.reset_index(drop=True)
        self.resultdf(update=True)

        to_plot = []
        for i in range(0,np.max(self.frames)+1):
            to_plot_ = []
            for c in range(0,len(self.to_plot[i])):
                if self.to_plot[i][c][1:3] in self.result[['x_centroid','y_centroid']].values:
                    to_plot_.append(self.to_plot[i][c])
            to_plot.append(to_plot_)
        self.to_plot = to_plot 


    def analyse(self,train=False,threshold=0.8):
        """Build, compile and train a ML model then perform full object detection analysis and tablate the results
        ------
        Parameters
        ------
        train : bool (default False)
            if true then the ML model is trained on the Xtrain dataset (only use if calling an untrained custom model)
        threshold : float (default 0.8)
            minimum probability of being a true source (according to the ML model) required for detections to be boxed in the plot
        """
        self.apply_model(train=train)
        self.detect(threshold=threshold)
        self.get_flux(analyse=True)
        self.group_and_id()
        self.resultdf()
        self.combine_groups()


    def check_region(self,xrange,yrange,frame=0):
        """Establishes a subregion of the images to be zoomed in on when plotting with SourceDetect.plot
        ------
        Parameters
        ------
        xrange : tuple
            lower and upper x-axis limits for the zoomed region desired for plotting 
        yrange : tuple
            lower and upper y-axis limits for the zoomed region desired for plotting 
        frame : int (default 0)
            desired frame (image index) to be plotted in the specified zoomed region
        ------
        Outputs
        ------
        zoom : list
            positions (as tuples) of detections within the specified frame and zoomed region
        zoom_range : list
            bounds of the zoomed region: of the form [ymin,ymax,xmin,xmax]
        """
        self.zoom = []
        self.zoom_range = [yrange[0],yrange[1],xrange[1],xrange[0]]
        for i in range(0,len(self.sources_by_frame[frame])):
            if self.sources_by_frame[frame][i][0] in range(yrange[0],yrange[1]) and self.close_sources[frame][i][1] in range(xrange[0],xrange[1]):
                self.zoom.append(self.sources_by_frame[frame][i])


    def plot(self,which_plots=['sources'],frame=0,compare=False,zoom=False,saveplots=False,savename=''):
        """Produces output plots illustrating the object detection process with identification boxes.
           Zoom currently only available for the close and unique sources plots
        ------
        Parameters
        ------
        which_plots : list (default ['sources'])
            one or more types of image plots to be produced out of the following options for identifying detections:
                - "sources" : all detections within a particular frame 
                - "close" : all detections in close proximity to other detections within a particular frame 
                - "unique" : all guaranteed unique detections (only one from each grouping included) within a particular frame  
                - "nobox" : the specified frame without any object detection boxes plotted on top
        frame : int (default 0)
            the frame (image index) to be plotted with speficied detections identified (see which_plots above)
        compare : bool (default False)
            if True then the plots specified by 'which_plots' will be displayed using matplotlib.pyplot.subplots rather than individually
        zoom : bool (default False)
            if True then only plot the region specified when calling SourceDetect.zoom (this must be done first)            
        saveplots : bool (default False)
            if True then save all plots to the location defined by SourceDetect.savepath
        savename : str (default '')
            beginning of the savename for any plots made (the plot types specified by 'which_plots' are appended to the end)
        """
        if zoom == True:
            try:
                self.zoom = self.zoom
            except:
                print('self.zoom, the region to be plotted, needs to be defined first by calling SourceDetect.check_region')
                

        def get_color_by_probability(p):
            """Specifies the box colour plotted for each source detection based on the confidence of the model.
               This is only important when the threshold is lowered to display uncertain objects in the plots
            ------
            Parameters
            ------
            p : float
                the probability of a detection being a real source
            ------
            Returns
            ------
            colour : str
                colour of the box to be plotted over the detected source (red, yellow, orange, or green)
            """
            if p < 0.3:
                return 'red'
            if p < 0.8:
                return 'yellow'
            return 'green'
        
        
        def plotter(ax,p,zoom=False):
            """Performs the plotting for SourceDetect.plot
            ------
            Parameters
            ------
            ax : matplotlib.axes._axes.Axes
                the subplot to be plotted on for this iteration
            p : str
                ones of the arguments from 'which_plots' defined when calling SourceDetect.plot
            zoom : bool (default False)
                if True then only plot the region specified when calling SourceDetect.zoom (this must be done first)            
            """
            if p == 'sources':
                ax.set_title('Detected Sources')

                for prob, px, py, x2, y2 in self.to_plot[frame]:
                    color = get_color_by_probability(prob)
                    ax.add_patch(Rectangle((int(px-x2/2),int(py-y2/2)),int(x2+2),int(y2+2),edgecolor=color,fill=False,lw=1))

                if zoom == True:
                    ax.set_ylim(self.zoom_range[0],self.zoom_range[1])
                    ax.set_xlim(self.zoom_range[2],self.zoom_range[3])

                plt.show()

            elif p == 'close':
                ax.set_title('Close Proximity Detections')

                for i in range(0,len(self.close_sources)):
                    for j in range(0,len(self.close_sources[i])):
                        ax.add_patch(Rectangle((int(self.close_sources[i][j][1]-2), int(self.close_sources[i][j][0]-2)), 4, 4, edgecolor='green', fill=False, lw=1))
                
                if zoom == True:
                    ax.set_ylim(self.zoom_range[0],self.zoom_range[1])
                    ax.set_xlim(self.zoom_range[2],self.zoom_range[3])
                
                plt.show()

            elif p == 'unique':
                ax.set_title('Definite Unique Detections')

                for i, j in self.unique_sources[frame]:
                    ax.add_patch(Rectangle((int(j-2),int(i-2)),4,4,edgecolor='green', fill=False, lw=1))
                
                if zoom == True:
                    ax.set_ylim(self.zoom_range[0],self.zoom_range[1])
                    ax.set_xlim(self.zoom_range[2],self.zoom_range[3])
                                    
                plt.show()

            elif p == 'nobox':
                ax.set_title('Original Frame')
                plt.show()

            else:
                print('Invalid plot type (options are "sources", "close", "unique", "nobox", or a combination of these as a list or tuple)')
    

        if compare == False:
            plotnames = {'sources':'detected_sources','close':'close_objects','unique':'unique_objects','nobox':'original_frame'}
            for p in which_plots:
                fig, ax = plt.subplots()
                im = ax.imshow(self.flux[frame],vmin=-10,vmax=10)
                fig.colorbar(im)
                plotter(ax,p,zoom)
                if saveplots == True:
                    if zoom == True:
                        plt.savefig(f'{self.savepath}/{savename}_{plotnames[p]}_zoomed', dpi=750)
                    else:
                        plt.savefig(f'{self.savepath}/{savename}_{plotnames[p]}', dpi=750)

        else:
            fig, ax = plt.subplots(1,len(which_plots),figsize=(10,10))
            for p in range(0,len(which_plots)):
                im = ax[p].imshow(self.flux[frame],vmin=-10,vmax=10)
                plotter(ax[p],which_plots[p],zoom)
            cbar_ax = fig.add_axes([0.93, 0.3, 0.022, 0.38])
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig(f'{self.savepath}/{savename}_object_detection')


    def SourceDetect(self,flux=None,train=False,do_cut=False,plot=False):
        """Perform object detection on a collection of TESS images/frames
        ------
        Parameters
        ------
        flux : array or None (defualt None)
            use this parameter to change the set of images being analysed rather than having to create a brand new SourceDetect object
        train : bool (default False)
            if true then the ML model is trained on the Xtrain dataset (only use if calling an untrained custom model)
        do_cut : bool (default False)
            if True then the TESS images are cut via a user prompt to specify the desired region to keep
        plot : bool (default False)
            if True then plot and save an example object detection plot on the first image/frame
        """
        self.issues = False
        if self.precheck == True:
            self.preview(do_cut=do_cut)
        if self.issues == False:
            if flux != None:
                self.flux = flux
                if len(self.flux.shape) == 2:
                    self.flux = np.expand_dims(self.flux,0)
            self.analyse(train=train)
        print('Collection complete')
        if plot == True:
            self.plot(which_plots=['sources'])
