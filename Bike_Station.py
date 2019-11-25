from argparse import ArgumentParser
import os
import logging
import pandas as pd
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class Bike(object):
        def __init__(self, jsonfile, nbcluster ,model,destinationfolder):
            self.jsonfile = jsonfile
            self.nbcluster = nbcluster 
            self.model =  model
            self.destinationfolder = destinationfolder
            
        def cluster(self, df, n_clusters, model):
            if model == "kmeans":
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(df.loc[:, ["longitude","latitude"]].notnull())
                labels = kmeans.labels_
            else:
                hc = AgglomerativeClustering(n_clusters=n_clusters, affinity = 'euclidean', linkage = 'ward')
                labels = hc.fit_predict(df.loc[:, ["longitude", "latitude"]].notnull())
            return labels


        def run(self):
             logger.info('Processing %s', self.jsonfile)
             df = pd.read_json(self.jsonfile)
             labels = self.cluster(df, self.nbcluster, self.model)
             df['label'] = labels
             logger.info('Generation output file')
             df.to_json(os.path.join(self.destinationfolder ,self.model+"_bike_stations.json"))
             
def usage():
    print("Usage : " + __file__ + " jsonfile nbcluster model destinationfolder")
    sys.exit(1)

if __name__ == '__main__':
    try:
        parser = ArgumentParser(usage='', description="")
        parser.add_argument("jsonfile", metavar="jsonfile", help="the input file json")
        parser.add_argument("nbcluster",type=int, metavar="nbcluster", help="number of clusters")
        parser.add_argument("model", metavar="model", help="clustring model")
        parser.add_argument("destinationfolder", metavar="destinationfolder",help="target folder for clustring")
        options = parser.parse_args()
        Bike(options.jsonfile, options.nbcluster, options.model, options.destinationfolder).run()
    except:
        print("astare process raise an exception")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        sys.exit(1)
