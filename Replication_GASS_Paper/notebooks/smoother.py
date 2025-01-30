import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal.weights as sw

class ConstantTerm():
    
    def __init__(self, n):
        self._x = np.ones(n).reshape(-1,1)     
        
    @property
    def X(self):
        return self._x
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)
    
class LinearTerm():
    
    def __init__(self, df, *idx, standard = False, log = False):
        self._df = df
        self._idx = list(idx) 
        X = self._df.iloc[:, self._idx].apply(pd.to_numeric).values
        
        if log == True:
            X = np.log(X)
            
        if standard == True:
            X = (X-np.mean(X))/np.std(X)
        
        self._x = X
    
    @property
    def X(self):
        return self._x
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)

class DistanceWeighting():
    
    def __init__(self, df, geomap, poidf, uids = [0,0,0,1], standard = False, log = False):
        """
        Initialize the DistanceWeighting class.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the observations, target support.
        geomap : GeoDataFrame
            GeoDataFrame used for spatial distance calculation, must include all observations in 'df' and all entities in 'poidf'.
        poidf : DataFrame
            DataFrame containing other spatial entities that have a different spatial support from the observations.
        uids : list of int, optional
            List of column indices to specify which columns to use from the above DataFrames. The default is [0, 0, 0, 1].

        uids element-wise explanation
        -----------------------------
        uids[0] : int
            Index of the column in 'df' DataFrame containing the Unique Identifiers for each observation.
        uids[1] : int
            Index of the column in 'geomap' DataFrame containing the Unique Identifiers that correspond to those in 'df'.
        uids[2] : int
            Index of the column in 'poidf' DataFrame containing the Unique Identifiers for each entity.
        uids[3] : int
            Index of the column in 'poidf' DataFrame containing the attributes of interest for each entity.
       
       Equations
       ---------
        Currently, this class uses the Power-Law Distance-Decay Weighted Sum for calculations:
            DDWS_{ij} = DDWS_j = sum over j (sum over k (m_k * d_jk^sigma))
            j is the index running over all observations in 'df' DataFrame
            k is the index running over all entities in 'poidf' DataFrame
            m_k represents the attribute of the k-th entity in 'poidf' DataFrame
            d_jk is the distance between the j-th observation and the k-th entity 
            
        Future extensions of this class may include exponential distance-decay weighted sum and weighted averages.

        """
        # 0. by default
        self.lower_bound = -5.0
        self.upper_bound = 0.0
        self.initial_sigma = -1.0
        self.standard = standard
        self.log = log
        
        # 1. get the UID for each observation in 'df'
        indices = list(uids)
        tdf = pd.DataFrame(np.array(df.iloc[:, indices[0]]), columns = ['UID'])  
        
        # 2. get the UID and attribute of all entities in 'poidf'
        tpoidf = poidf.iloc[:, list([indices[2], indices[3]])] 
        tpoidf = tpoidf.sort_values(by = [tpoidf.columns[0]]) # make sure mk can correspond to djk
        pois = tpoidf.iloc[:, 0].values.flatten()
        
        # 3. get the attribute of each entity
        def get_mk(uid):
            
            index_mask = tpoidf.iloc[:,0] != uid # exclude the observation from 'poidf' in case the poidf includes observations in 'df'
            ttpoidf = tpoidf[index_mask]
            mk = ttpoidf.iloc[:,1].values.reshape((1,-1))

            return mk
        
        # 4. generate a unique geo-map including one observation in 'df' and all entities in 'poidf'; 
        #    then calcualte distances between this specific observation and all entities;
        #    repeat this process 'get_djk()' for each observation in 'df' 
        def get_djk(uid):
            
            # The first element in 'pt_pois' represents the specific observation in 'df'.
            # All subsequent elements in 'pt_pois' represent all entities in 'poidf' to which we will calculate distances.
            pt_pois = list([uid])
            tpois = [x for x in pois if x != uid]
            pt_pois.extend(tpois)

            # 'tmp': the temporary map for one specific observation and entities
            # The rows of 'tmp' should follow the same order as in 'pt_pois' to ensure that 'mk' and 'djk' align element-wise for subsequent calculations.
            tmp = geomap[geomap.iloc[:, indices[1]].isin(pt_pois)] 
            tmp = tmp.assign(ncat = pd.Categorical(tmp.iloc[:, indices[1]], categories=pt_pois, ordered=True))
            tmp = tmp.sort_values(by = 'ncat').reset_index()
            
            # Calculate the Euclidean distances from the first geometry to all other geometries in 'tmp'.
            dk = np.array(tmp.iloc[1:,].distance(tmp.geometry.iloc[0]))
            dk[dk < 1] += 1
            return dk

        # 5. Compute the Power-law Distance Decay Weighted Sum for each observation in 'df'
        findf = tdf.assign(mk = tdf.iloc[:,0].apply(lambda x: get_mk(x)),
                           djk = tdf.iloc[:,0].apply(lambda x: get_djk(x)))

        self._mk = findf.mk
        self._djk = findf.djk
        self._df = findf
    
    def cal(self, sigma):
        
        def cal_PDDWS(a, b, sigma):
            ddws = np.dot(a, np.power(b, sigma))
            return ddws

        pddws = np.vectorize(cal_PDDWS)(self._mk, self._djk, sigma)
        pddws = pd.to_numeric(pddws).reshape((-1,1))
        
        if self.log == True:
            pddws = np.log(pddws)
        
        if self.standard == True:
            pddws = (pddws-np.mean(pddws))/np.std(pddws)
        
        return pddws
    
    def set_searching_range(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        pass
    @property
    def mk(self):
        return self._mk
    
    @property
    def dkj(self):
        return self._djk
    
    @property
    def df(self):
        return self._df
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)
    
class SpatialWeightSmoother():
    def __init__(self, df, geomap, uids = [0,0,1], standard = False, average = False, log = False):
        
        # 0. by default
        self.lower_bound = -3.0
        self.upper_bound = 0.0
        self.initial_sigma = -1.0
        self.standard = standard
        self.average = average
        self.log = log
        
        indices = list(uids)
        tdf = pd.DataFrame(np.array(df.iloc[:, list((indices[0], indices[2]))]), columns = ['UID', 'X'])
        pts = tdf.iloc[:, 0].values.flatten()
        tmap = geomap[geomap.iloc[:, indices[1]].isin(pts)]
        tmap = tmap.assign(ncat = pd.Categorical(tmap.iloc[:, indices[1]], categories=pts, ordered=True))
        tmap = tmap.sort_values(by = 'ncat').reset_index()
        distmat = sw.distance.DistanceBand.from_dataframe(tmap, threshold=99999, binary=False, alpha = 1)
        distmat = distmat.full()[0]
        self._distmat  = distmat.copy()
        np.fill_diagonal(self._distmat, 1) # sp that we can include self-neighboring values
        self._distmat[self._distmat < 1] += 1
        
        self._tmap = tmap
        self._X = tdf.X
    
    @property
    def X(self):
        return self._X
    
    @property
    def tmap(self):
        return self._tmap
    
    def cal(self, sigma):
        sws_W = np.power(self._distmat, sigma)
        sws = pd.to_numeric(self.X @ sws_W ).reshape(-1,1)
        
        if self.average == True:
            sws = sws/sws.shape[0]
        
        if self.log == True:
            sws = np.log(sws) 
    
        if self.standard == True:
            sws = (sws-np.mean(sws))/np.std(sws)
            
        return sws
    
    def set_searching_range(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        pass
    
    #in case you want to check the weights
    def sws_W(self, sigma):
        sws_W = np.power(self._distmat, sigma)
        
        return sws_W
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)