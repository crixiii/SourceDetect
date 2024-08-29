import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import keras 
import keras.ops as ops
from keras import layers
import tensorflow as tf

from .prfbuild import PrfBuild



class PrfModel:
    """Builds and/or trains a ML model to perform object detection on simulated datasets from PrfBuild
       or real reduced TESS images then analyses the results"""

    def __init__(self,Xtrain='deault',ytrain='default',savepath=None,model='default',save_model=True,loss_func='default',optimizer=tf.keras.optimizers.Adam,learning_rate=0.003,
                 metrics=["categorical_accuracy"],monitor='loss',batch_size=32,epochs=50,validation_split=0.1):
        """
        Initialise
        
        Parameters
        ----------
        Xtrain : str (default 'default')
            TESS prf arrays to be added into the training/test sets; if 'default' then load premade set    
        ytrain : str (default 'default')
            labels of the TESS prf arrays (npy file) to be added into the training/test sets; if 'default' then load premade labels (positive/negative sources can either share a label or have different labels)
        savepath : str or None (default None)
            location to save any output modelsand plots
        model : str (default 'default')
            ML model savepath; if 'default' then a prebuilt model is defined
        loss_func : str or keras.losses (default 'default')
            loss function used by the model; if 'default' then a prebuilt model is defined
        optimizer : tensorflow.keras.optimizers instance (default Adam)
            tensorflow optimiser method to be used by the ML model
        learning_rate : float (default 0.003)
            learning rate during the training process
        metrics : list (default ["categorical_accuracy"])
            list of metrics (strings and/or keras.metrics.Metric instances) to be evaluated by the model during training/testing (using tensorflow naming conventions)
        monitor : str (default 'loss')
            name of metric to monitor during training (using tensorflow naming conventions)
        batch_size : int (default 32)
            batch size used when training the ML model 
        epochs : int (default 50)
            number of training epochs performed by the ML model 
        validation_split : float (default 0.1)
            the ratio of training set images used in the validation set 
        """
                     
        self.dataset = PrfBuild(Xtrain,ytrain)
        if savepath == None:
            self.savepath = '.'
        else:
            self.savepath = savepath
        self.model = model
        self.loss_func = loss_func
        if self.loss_func == 'default':
            try:
                loss = self.model.loss
                if loss != None:
                    self.loss_func = loss
            except:
                pass
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.monitor = monitor
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        self.savename = 'object_detection'
        self.save_model = save_model

        self.__main__()
    

    def get_color_by_probability(self,p):
        """
        Specifies the box colour plotted for each source detection based on the confidence of the model.
        This is only important when the threshold is lowered to display uncertain objects in the plots
        
        Parameters
        ----------
        p : float
            the probability of a detection being a real source
        
        Returns
        -------
        colour : str
            colour of the box to be plotted over the detected source (red, yellow or green)
        """
        
        if p < 0.3:
            return 'red'
        if p < 0.8:
            return 'yellow'
        return 'green'


    def show_predict(self,threshold=0.1,saveplot=False,skipbox=False):
        """
        Plots the object detection result for a real or simulated TESS image and detected positions of the sources
        and whether they are bright or dim sources
        
        Parameters
        ----------
        threshold : float (default 0.1)
            minimum probability of being a valid detection (according to the ML model) required for detections to be boxed in the plot
        saveplot : bool (default False)
            if True then the detections plot is saved to the savepath defined when calling find_sources.PrfModel
        skipbox : bool (default False)
            if True then the image is plotted without the detection boxes
        
        Outputs
        -------
        detections : list
            positions (as tuples) of all potential sources detected by the model 
        num_sources : int
            number of sources detected by the model
        bright_or_dim : dictionary
            documents whether each detection was flagged as a variable star
        """
        
        i,j = self.dataset.y[0].shape[0],self.dataset.y[0].shape[1]
        num_sources, prob_sources = 0, 0

        fig, ax = plt.subplots()
        im = ax.imshow(self.X[0],vmin=-10,vmax=10)

        positions = []
        bright_or_dim = {}

        for mx in range(j):
            for my in range(i):
                output = self.dataset.y[0][my][mx]
                prob, x1, y1, x2, y2 = output[:5]
                bright, dim, trash, fake = output[5:]

                if prob < threshold:
                    continue
                if trash > bright and trash > dim:
                    continue
                if fake > bright and fake > dim:
                    continue

                color = self.get_color_by_probability(prob)
                px, py = (mx * self.dataset.grid_size) + x1, (my * self.dataset.grid_size) + y1
                if self.dataset.y_shape != (4,4):
                    px, py = px/4, py/4

                if self.X[0][int(py),int(px)] > -1.5 and self.X[0][int(py),int(px)] < 1.5:
                    continue 
                if np.max(self.X[0][int(py)-2:int(py)+3,int(px)-2:int(px)+3]) > 7.5 and np.min(self.X[0][int(py)-2:int(py)+3,int(px)-2:int(px)+3]) < -7.5:
                    continue

                if skipbox == False:
                    ax.add_patch(Rectangle((int(px-x2/2), int(py-y2/2)), int(x2+1), int(y2+1), edgecolor=color, fill=False, lw=1))

                num_sources += 1
                if threshold < 0.8:
                    if prob > 0.8:
                        prob_sources += 1
                
                positions.append((int(py),int(px)))
                if bright > dim:
                    bright_or_dim[(int(py),int(px))] = 1
                elif bright < dim:
                    bright_or_dim[(int(py),int(px))] = -1

        if num_sources == 1:
            print('1 potential source detected above the threshold')
        elif num_sources > 1:
            print(f'{int(num_sources)} potential sources detected above the threshold')
        else:
            print('No sources')

        if prob_sources > 0:
            if prob_sources != num_sources:
                print(f'(Probably {int(prob_sources)} actual sources)')

        fig.colorbar(im)

        if saveplot == True:
            plt.savefig(f'{self.savepath}/prf_plot', dpi=750)
        plt.show()

        self.detections = sorted(positions)
        self.num_sources = num_sources
        self.bright_or_dim = bright_or_dim


    def add_model(self,summary=True):
        """
        Call to build the default ML model to be trained and/or display a model summary (for either the default or own custom model)
        
        Parameters
        ----------
        model : str or tensorflow.keras.models.Model (default 'default')
            ML model; if 'default' then a default model is defined.
        summary : bool (default True)
            if True then prints a visual summary of the ML model 
        
        Outputs
        -------
        model : ML model to be used for object detection
        """
        
        if self.model == 'default':
            x = x_input = tf.keras.layers.Input(shape=(self.dataset.x_shape[0], self.dataset.x_shape[1], 1))

            x = tf.keras.layers.Conv2D(12, kernel_size=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.MaxPool2D()(x)
            x = tf.keras.layers.BatchNormalization()(x) 

            x = tf.keras.layers.Conv2D(12, kernel_size=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Conv2D(12, kernel_size=2, padding='same', activation='relu')(x)
            x = tf.keras.layers.MaxPool2D()(x)
            x = tf.keras.layers.BatchNormalization()(x) 

            x_prob = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='x_prob')(x)
            x_boxes = tf.keras.layers.Conv2D(4, kernel_size=3, padding='same', name='x_boxes')(x)
            x_cls = tf.keras.layers.Conv2D(4, kernel_size=3, padding='same', activation='sigmoid', name='x_cls')(x)

            gate = ops.where(x_prob > 0.5, ops.ones_like(x_prob), ops.zeros_like(x_prob))
            x_boxes = x_boxes * gate
            x_cls = x_cls * gate

            x = tf.keras.layers.Concatenate()([x_prob, x_boxes, x_cls]) 
            self.model = keras.Model(x_input, x, name="ObjectDetector")

        if summary == True:
            self.model.summary()


    def add_loss_func(self):
        """
        Build the default loss function used to quantify the performance of the training process
        
        Returns
        -------
        loss_func : function
            loss function to be used by the ML model
        """
        
        if self.loss_func == 'default':
            idx_p = [0]
            idx_bb = [1, 2, 3, 4]
            idx_cls = [5, 6, 7, 8]

            @tf.function
            def loss_bb(y_true, y_pred):
                y_true = tf.gather(y_true, idx_bb, axis=-1)
                y_pred = tf.gather(y_pred, idx_bb, axis=-1)

                loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
                return tf.reduce_mean(loss[loss > 0.0])

            @tf.function
            def loss_p(y_true, y_pred):
                y_true = tf.gather(y_true, idx_p, axis=-1)
                y_pred = tf.gather(y_pred, idx_p, axis=-1)
                
                loss = tf.losses.binary_crossentropy(y_true, y_pred)
                return tf.reduce_sum(loss)

            @tf.function
            def loss_cls(y_true, y_pred):
                y_true = tf.gather(y_true, idx_cls, axis=-1)
                y_pred = tf.gather(y_pred, idx_cls, axis=-1)
                
                loss = tf.losses.binary_crossentropy(y_true, y_pred)
                return tf.reduce_sum(loss)

            @tf.function
            def loss_func(y_true, y_pred):
                return loss_bb(y_true, y_pred) + loss_p(y_true, y_pred) + loss_cls(y_true, y_pred)

            self.loss_func = loss_func


    def compile_model(self,savename='object_detection'):
        """
        Compile the ML model with the loss function, an optimiser and evaluation metrics
        
        Parameters
        ----------
        savename : str (default 'object_detection')
            savename for the object detection plot
        """
        
        self.optimizer = self.optimizer(learning_rate=self.learning_rate)
        self.savename = savename
        self.monitor = self.monitor
        self.metrics = self.metrics
        self.callbacks = [
        keras.callbacks.ModelCheckpoint(
                f"{self.savepath}/{savename}.keras", save_best_only=True, monitor=self.monitor
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=self.monitor, factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor=self.monitor, patience=50, verbose=1),
        ]
        self.model.compile(loss=self.loss_func, optimizer=self.optimizer, metrics=self.metrics)


    def build(self,summary=True):
        """
        Defines and compiles a ML model for training/testing then creates a dataset and corresponding labels incase that hasn't already been done
        
        Parameters
        ----------
        loss : str or func (default 'default')
            loss function; if 'default' then a default loss function is defined
        summary : bool (default True)
            if True then prints a visual summary of the ML model 
        """
        
        self.add_model(summary=summary)
        self.add_loss_func()
        self.compile_model()
        
        self.dataset.make_data(size=1,num=2)
        self.dataset.y = self.model.predict(self.dataset.X)


    def preview(self,num=2,threshold=0.1,saveplot=False,skipbox=False):
        """
        Crates a training set and labels then plots a single 16x16 training image (detection boxes will not be useful if this is called before training)
        
        Parameters
        ----------
        num : int (default 2)
            maximum number of true/false sources in each image
        threshold : float (default 0.1)
            minimum probability of being a true source (according to the ML model) required for detections to be boxed in the plot
        saveplot : bool (default False)
            if True then the detections plot is saved to the savepath defined when calling find_sources.PrfModel
        skipbox : bool (default False)
            if True then the image is plotted without the detection boxes
        """
        
        self.dataset.make_data(size=1,num=num)
        self.dataset.y = self.model.predict(self.dataset.X)
        self.show_predict(threshold=threshold,saveplot=saveplot,skipbox=skipbox)


    def train(self,batch_size=None,epochs=None,validation_split=None):
        """
        Train the ML model
        
        Parameters
        ----------
        batch_size : None or int (default None)
            batch size used when training the ML model; if None then the batch size defined when calling PrfModel is used
        epochs : None or int (default None)
            number of training epochs performed by the ML model; if None then the number of epochs defined when calling PrfModel is used
        validation_split : None or float (default None)
            the ratio of training set images used in the validation set; if None then the validation split defined when calling PrfModel is used
        
        Outputs
        -------
        history : tf.keras.callbacks.History object
            Record of events during the training process
        """
        
        if self.batch_size == None:
            self.batch_size = 32
        if self.epochs == None:
            self.epochs = 50
        if self.validation_split == None:
            self.validation_split = 0.1
        if batch_size != None:
            self.batch_size = batch_size
        if epochs != None:
            self.epochs = epochs
        if validation_split != None:
            self.validation_split = validation_split
        self.dataset.make_data(size=self.batch_size * 400)

        self.history = self.model.fit(
            self.dataset.X,
            self.dataset.y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            shuffle=True,
            validation_split=self.validation_split,
            verbose=1)
        if self.save_model:
            self.model.save("object_detection_model.keras")


    def sim_detect(self):
        """
        Analyses the result of applying an ML model to a simulated image to quantify the effectiveness of the model:
        finds the number of sources detected/undetected, any close groupings of sources (potential double detections
        or close objects), the overall number of detections made by the model, and the percentage of detections that 
        actually correspond to a 'true' source
        
        Outputs
        -------
        matches : list
            positions (as tuples) of detections corresponding to each source in the image 
        correct : float
            lower limit for the proportion of sources in the image correctly identified by the model
        missed_sources : list
            positions (as tuples) of any sources not detected by the model
        close_sources : list
            lists of positions (as tuples) of sources within close proximity to each other 
        """
        
        result = [[] for _ in range(len(self.dataset.sources))]
        ind, corr = 0, 0

        for i in self.dataset.sources:

            for j in self.detections:
                if np.abs(i[0]-j[0]) <= 3 and np.abs(i[1]-j[1]) <= 3:
                    result[ind].append(j)
            
            if len(result[ind]) == 1:
                corr += 1
            ind += 1

        missed_sources = []

        for i in range(0,len(result)):
            if len(result[i]) == 0:
                missed_sources.append(self.dataset.sources[i])

        close_sources = []

        for i in range(0,len(result)):
            if len(result[i]) > 1:
                if sorted(result[i]) not in close_sources:
                    close_sources.append(sorted(result[i]))

        if len(missed_sources) > 0:
            if len(missed_sources) == 1:
                print('There is 1 unidentified source at:')
            else:
                print(f'There are {len(missed_sources)} unidentified sources at:')

            for i in missed_sources:
                print(i)
            print()

        if len(close_sources) > 0:
            print(f'This leaves {len(self.dataset.sources)-len(missed_sources)} potentially detected sources in the image with {self.num_sources} potential detections above the threshold')
            print()

            if len(close_sources) == 1:
                print('There is 1 set of close proximity detections at:')
            else:
                print(f'There are {len(close_sources)} sets of close proximity detections at:')
            
            for i in close_sources:
                string = ''
                for j in i:
                    string += str(j) + ', '
                print(string[0:-2])
            
            print('These may be multiple detections for the same sources or multiple sources very close together. Each set of detections above are over at least one source so each set has at least one correct detection')
            print()
            print(f'Therefore, at least {int(corr)} (likely {len(self.dataset.sources)-len(missed_sources)}) of the {len(self.dataset.sources)} sources ({100*int(corr)/len(self.dataset.sources):.2f}-{100*(len(self.dataset.sources)-len(missed_sources))/len(self.dataset.sources):.2f}%) were correctly identified by the model')
        
        else:
            if len(missed_sources) == 0:
                if len(self.dataset.sources) == 1:
                    print('The 1 source was corectly identified by the model')
                else:
                    print(f'All {len(self.dataset.sources)} of the sources were correctly identified by the model')
            
            else:
                print(f'{int(corr)} of the {len(self.dataset.sources)} sources ({100*int(corr)/len(self.dataset.sources):.2f}%) were correctly identified by the model')
        
        if len(self.detections) > len(self.dataset.sources):
            print(f'Hence, there were also at least {len(self.detections)-len(self.dataset.sources)} false or non-unique detections made by the model so {int(corr)} to {len(self.dataset.sources)-len(missed_sources)} of the {len(self.detections)} detections ({100*(int(corr)/len(self.detections)):.2f}-{100*(len(self.dataset.sources)-len(missed_sources))/len(self.detections):.2f}%) were correct and unique classifications')      
        
        self.matches = result
        self.correct = int(corr)/len(self.dataset.sources)
        self.sources = self.dataset.sources
        self.missed_sources = missed_sources
        self.close_sources = close_sources


    def show_result(self,x_shape=(512,512),y_shape=(128,128),num=20,threshold=0.8,saveplot=False,skipbox=False):
        """
        Complete ML model build and evaluation of a simulated full TESS CCD image
        
        Parameters
        ----------
        x_shape : tuple (default (512,512))
            shape of the training/test images 
        y_shape : tuple (default (128,128))
            shape of the object position/size/label output 
        num : int (default 20)
            maximum number of true/false sources in each image
        threshold : float (default 0.8)
            minimum probability of being a true source (according to the ML model) required for detections to be boxed in the plot
        saveplot : bool (default False)
            if True then the detections plot is saved to the savepath defined when calling find_sources.PrfModel
        skipbox : bool (default False)
            if True then the image is plotted without the detection boxes
        """
        
        self.dataset.make_data(x_shape=x_shape,y_shape=y_shape,size=1,num=num)
        self.dataset.y = self.model.predict(self.dataset.X)
        self.show_predict(threshold=threshold,saveplot=saveplot,skipbox=skipbox)
        self.sim_detect()


    def __main__(self):
        """Applies the model building and training process"""
        if self.model == 'default':
            print('Building Model:')
            self.build(summary=False)
            print('Building Complete')
        print('Training Model:')
        self.train()
        print('Training complete')
