import numpy as np
import os


class PrfBuild:
    """Create a training or test set of simulated TESS images containing both 'real' and 'false' sources"""

    def __init__(self,Xtrain,ytrain,run=True):
        """
        Initialise
        
        Parameters
        ----------
        Xtrain : str
            TESS prf arrays to be added into the training/test sets  
        ytrain : str
            TESS prf arrays to be added into the training/test sets (positive/negative sources can either share a label or have different labels)
        
        Options
        -------
        run : bool
            if true (default) then the training/test set will be built upon calling PrfBuild rather 
            than just defining the prfs and corresponding labels 
        """
        print('made it to prf build')
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.directory = os.path.dirname(os.path.abspath(__file__)) + '/'
        print('a')
        if type(self.Xtrain) == str:
            self.Xtrain = np.load(self.directory+'training_data.npy',allow_pickle=True)
        if type(self.ytrain) == str:
            self.ytrain = np.load(self.directory+'training_labels.npy',allow_pickle=True)
        if run==True:
            print('b')
            self.make_data()


    def make_labels(self,X,y,num):
        """
        Places true/false sources into the training/test array with randomly assigned positions and updates
        the label arrays accordingly (this is called once per training/test image)
        
        Parameters
        ----------
        X : array
            training/test dataset template with background but no sources 
        y : array
            training/test dataset labels template (np.zeros array)
        num : int
            maximum number of true/false sources in each image
        
        Returns
        -------
        positions : list
            list of tuples corresponding to the coordinates of the true/false sources
        """
        
        positions = []

        for _ in range(num):
            print('aa')
            placed = False
            while placed == False:
                # print('aaa')
                try:
                    idx = np.random.randint(len(self.Xtrain))
                    number = self.Xtrain[idx]
                    class_, offset_x, offset_y, w, h = int(self.ytrain[idx][0]), int(self.ytrain[idx][1]), int(self.ytrain[idx][2]), int(self.ytrain[idx][3]), int(self.ytrain[idx][4])
                    px, py = np.random.randint(2,int(self.x_shape[0]-2)), np.random.randint(2,int(self.x_shape[1]-2))
                    # mx, my = (px+2) // self.grid_size, (py+2) // self.grid_size
                    mx, my = px//self.grid_size, py//self.grid_size
                    output = y[my][mx]

                    #prevent multiple generations from overlapping
                    if output[0] > 0:
                        continue
                    overlap = False
                    for i,j in positions:
                        if i in range(py+-2,py+3) and j in range(px-2,px+3):
                        # if i in range(py+offset_y+2,py+offset_y+3) and j in range(px+offset_x-2,px+offset_x+3):
                            overlap = True
                    if overlap == True:
                        continue

                    # X[py-h//2:py+(h-h//2),px-w//2:px+(w-w//2),0] = number
                    # if class_ != 3:
                    #     X[py-h//2:py+(h-h//2),px-w//2:px+(w-w//2),0] *= np.max((np.random.rand()*2.5)+0.5)
                    # px, py = px + offset_x, py + offset_y
                    
                    output[0] = 1.0
                    # output[1] = px - (mx * self.grid_size)  # x1
                    # output[2] = py - (my * self.grid_size)  # y1
                    output[1] = px % self.grid_size  # Object grid x index
                    output[2] = py % self.grid_size  # Object grid y index
                    output[3] = w # Object width
                    output[4] = h # Object height
                    output[5 + class_] = 1.0

                    X[py-h//2:py+(h-h//2),px-w//2:px+(w-w//2),0] = number
                    print('b')
                    if class_ != 3:
                        X[py-h//2:py+(h-h//2),px-w//2:px+(w-w//2),0] *= np.max((np.random.rand()*2.5)+0.5) #0.75+np.random.rand()*(3*np.std(X[:,:,0])-0.75)
                    if class_ < 2:
                        positions.append((py,px))
                    placed = True
                except:
                    pass
        print('aaaa')
        return positions    
        

    def make_data(self,x_shape=(100,100),y_shape=(4,4),size=64,num=2):
        """
        Creates the template training/test dataset and label arrays and saves the positions of the real/false sources.
        
        Parameters
        ----------
        x_shape : tuple (default (16,16))
            shape of the training/test images 
        y_shape : tuple (default (4,4))
            shape of the object position/size/label output 
        size : int (default 64) 
            number of training/test images
        num : int (default 2)
            maximum number of true/false sources in each image
        
        Outputs
        -------
        sources : list
            positions (as tuples) of the sources in the image
        X : array
            training/test images
        y : array
            labels corresponding to the training/test images (labels for object position, size, likelihood, and probability of positive/negative/false source)
        """
        
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.grid_size = int(x_shape[0]/y_shape[0])
        print('d')
        X = np.zeros((size, self.x_shape[0], self.x_shape[1], 1), dtype=np.float32)
        y = np.zeros((size, self.y_shape[0], self.y_shape[1], 9), dtype=np.float32)
        positions = []
        print('e')
        for i in range(size):
            print(i)
            X[i] += np.random.normal(0,np.random.rand()*0.2+0.4,(x_shape[0], x_shape[1], 1))
            positions.append(self.make_labels(X[i],y[i],num=num))
            print('done', i)
        print('f')
        self.sources = sorted(positions)[0]
        self.X = X
        self.y = y
