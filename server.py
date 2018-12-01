import os
import math
import datetime
import pickle
import numpy as np
import pandas as pd
import json
import html


from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from collections import defaultdict, OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from cgi import parse_header, parse_multipart
from lineitRequestHandler import LineitRequestHandler

os.chdir(os.path.dirname(__file__))

#Load static datasets
print("LoadingData")
js = open('sourceData/featureVals.json')
cats = json.load(js, encoding="utf-8")

try:
    dtSuspects = pd.read_csv("sourceData/suspects.txt", sep=';', header=0, dtype=str)
    dtSuspects = dtSuspects["id"].tolist()
except pd.io.common.EmptyDataError:
    dtSuspects = []
try:
    dtFeatures = pd.read_csv("createdFeatures.csv", sep=';', header=None, dtype=str)
except pd.io.common.EmptyDataError:
    dtFeatures = pd.DataFrame()


df = pd.read_csv("sourceData/features.csv", sep=';', header=0, dtype={"id": str})
d = pd.read_csv("sourceData/personsData.csv", sep=';', header=0)
        
cbFeatures = pd.read_csv("sourceData/personsCBVectors.csv", sep=';', header=None)       
     


# HTTPRequestHandler class
class LineitV2Server(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):                
        self.cats =  cats     
        self.dtSuspects = dtSuspects
        self.dtFeatures =  dtFeatures
        self.df = df
        self.d = d        
        self.cbFeatures = cbFeatures   
           
        with open("sourceData/cbCoefs.csv", "r") as file:  
            data = file.read() 
            self.cbCoefs = [float(i) for i in data.split(";")]
          
        with open("sourceData/cbVals.csv", "r") as file:  
            self.cbVal = float(file.read())    
            
        print("LineitServerInit")
        """
        js = open('sourceData/featureVals.json')
        self.cats = json.load(js, encoding="utf-8")

        try:
            self.dtSuspects = pd.read_csv("sourceData/suspects.txt", sep=';', header=0, dtype=str)
            self.dtSuspects = self.dtSuspects["id"].tolist()
        except pd.io.common.EmptyDataError:
            self.dtSuspects = []



        try:
            self.dtFeatures = pd.read_csv("createdFeatures.csv", sep=';', header=None, dtype=str)
        except pd.io.common.EmptyDataError:
            self.dtFeatures = pd.DataFrame()


        self.df = pd.read_csv("sourceData/features.csv", sep=';', header=0, dtype={"id": str})
        self.d = pd.read_csv("sourceData/personsData.csv", sep=';', header=0)
        
        self.cbFeatures = pd.read_csv("sourceData/personsCBVectors.csv", sep=';', header=None)
        #self.cbCoefs = pd.read_csv("sourceData/cbCoefs.csv", sep=';', header=None).values()
        
        with open("sourceData/cbCoefs.csv", "r") as file:  
            data = file.read() 
            self.cbCoefs = [float(i) for i in data.split(";")]
        """
        self.lrh = LineitRequestHandler(self.cbFeatures, self.cbCoefs, self.cbVal)
        
        
        super(LineitV2Server, self).__init__(request, client_address, server)


    def do_GET(self):
        self.send_response(200)
        # Send headers
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        params = parse_qs(urlparse(self.path).query)
        #print(params)
        
        try:
            self.dtFeatures = pd.read_csv("createdFeatures.csv", sep=';', header=None, dtype=str)
        except pd.io.common.EmptyDataError:
            self.dtFeatures = pd.DataFrame()

        
        for i in params.keys():
          if type(params[i]) == list:
              params[i] = params[i][0]

        if params.get("ajaxRequest","") != "":
            print("AJAX request")
            request =  params.get("ajaxRequest","")
            
            if request == "addToLineupAndRecommend":
                 lineupSuspect = params.get("lineupSuspect", "")
                 fmembers = params.get("featureMembers", "").split(",")          
                 fmembers = list(set(fmembers))
                 addedCBVal =  float(params.get("addedCBVal", 0.0))
                 addedVISVal =  float(params.get("addedVISVal", 0.0))
                                  
                 if (addedCBVal - addedVISVal) != 0.0:                  
                    self.lrh.updateRecommenderWeights(addedCBVal, addedVISVal)
                    #TODO: implement
                 
                 html = self.lrh.getRecommendedCandidates(lineupSuspect, fmembers, self.cbFeatures)
                 
                 
                 message = html
                 self.send_response(200)
                 self.wfile.write(bytes(message, "utf8"))                 
            
            


        elif params.get("uid", "") != "":  # it is a valid request, no favicon etc
            print("GET request")
            #print(params["featureMembers"])
            #print(type(params["featureMembers"]))
            ids = params.get("ids", "")            
            ids = ids.split(",")
            ids = [x for x in ids if x] #remove empty strings
            dsc = params.get("descriptor", "")
            fweight = params.get("weighting", "")
            qagg = params.get("aggregation", "")
            qLagg = params.get("local_aggregation", "")
            glW = params.get("globalLocalWeights", "")
            qRel = self.lrh.getQueryRelevance()
            wpR = params.get("whitePatchRemoval", "-1")

            fmembers = params.get("featureMembers", "").split(",")
            fmembers = [x for x in fmembers if x] #remove empty strings
            
            fmembers = list(set(fmembers))
            #print(fmembers)
            
            lineupSuspect = params.get("lineupSuspect", "")
            lineupNote = params.get("lineupNote", "")
            lineupAuthor = params.get("lineupAuthor", "")             
            lineupID = params.get("lineupID", "")

            exampleID = params.get("exampleID", "0")
            exampleQueries = self.lrh.getExampleQueries(int(exampleID))

            cart = list(set(params.get("cartMembers", [])))
            recPos = params.get("recPos", "")

            actionType = params.get("actionType", "")
            navSubmit = params.get("navSubmit", "")
            
            if isinstance(actionType, list):  # sometimes action type appears to be a list - strange, maybe because of not yet send requests from the slider?
                actionType = actionType[len(actionType) - 1] 
                
                
            if actionType == "Update / Create lineup":
                lineupID = self.lrh.saveFeature(fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID)
                self.dtFeatures = pd.read_csv("createdFeatures.csv", sep=';', header=None, dtype=str)   
                if len(ids) > 0:
                    actionType = "Run Search"                                     

            features = ""
            lof = []
            try:
                for i in range(0, self.dtFeatures.shape[0]):
                    features += self.lrh.createFeatureOption(i, self.dtFeatures[1][i], self.dtFeatures[0][i], lineupID)
                    lof.append(self.lrh.createFeatureRecord(i, self.dtFeatures[0][i], self.dtFeatures[1][i], self.dtFeatures[2][i], self.dtFeatures[3][i]))
                listOfFeatures = "{\n" + ",".join(lof) + "\n}"
            except:
                listOfFeatures = "{}"


            if actionType == "Save Query":
                querySummary = params.get("querySummary", "")
                ipAddr = html.escape(os.environ["REMOTE_ADDR"])
                self.lrh.saveQuery(ipAddr + ";" + querySummary + ";'" + recPos + "'")

            if navSubmit != "":                
                # non-empty CB search, filter candidates and run visual Search if existing
                categoryIds = self.lrh.getObjectsByCategory(navSubmit, self.df)
                searchIds = ids
                catNames = self.lrh.getNamesFromCategories(navSubmit, self.cats)
                self.lrh.updateCBWeights(self.cbFeatures, self.cats, navSubmit)
                
                resultsTxt = self.lrh.cbsearch(self.dtSuspects, self.df, self.d, self.cbFeatures, self.cats, categoryIds, searchIds, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID, features, listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries,
                               cart, navSubmit, catNames, recPos)
                               
                               
            elif (actionType == "Multiquery Search" or actionType == "Run Search") and len(ids) > 0:
                #self.lrh.updateCBWeights(self.cbFeatures, self.cats)
                resultsTxt = self.lrh.search(self.dtSuspects, self.df, self.d, self.cbFeatures, self.cats, ids, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID, features, listOfFeatures, fweight,
                                    qLagg, qagg, glW, wpR, qRel, exampleQueries, cart, recPos)

            elif actionType == "Save Query":
                if len(ids) > 0:
                    resultsTxt = self.lrh.search(self.dtSuspects, self.df, self.d, self.cbFeatures, self.cats, ids, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID, features,
                                 listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries, cart, recPos)
                else:
                    resultsTxt = self.lrh.main(self.dtSuspects, self.df, self.d, self.cbFeatures, ids, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID, features,
                               listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries, cart)          

            elif actionType == "Load example":
                name, ids, dsc, fweight, qagg = self.lrh.loadExample(exampleID)
                resultsTxt = self.lrh.search(self.dtSuspects, self.df, self.d, self.cats, ids.split(","), dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID, features, listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel,
                             exampleQueries, cart, cbFeatures)


            else:
                resultsTxt = self.lrh.main(self.dtSuspects, self.df, self.d, self.cbFeatures, self.cats, ids, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID, features, listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries, cart)

            message = resultsTxt
            self.send_response(200)
            # Send headers
            #self.send_header('Content-type', 'text/html')
            #self.end_headers()

            # store the query and response to the logfile
            # Send message back to client
            # Write content as utf-8 data
            self.wfile.write(bytes(message, "utf8"))
        return


def run():
    print('starting server...')  # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('', 50001)
    httpd = HTTPServer(server_address, LineitV2Server)
    print('running server...')
    httpd.serve_forever()

run()



