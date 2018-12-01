import numpy as np
import pandas as pd
import random
from scipy.spatial import distance
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import normalize
from flask import Flask, request, render_template, Markup
from datetime import datetime
import math
import os
import jinja2
import re                                     
import json

class LineitRequestHandler:    
    
    def __init__(self, cbFeatures, cbCoefs, cbVal):
        self.menuShape = {
            "Gender": "GENDER",
            "Age": "AGE",
            "Nationality": "NATIONALITY",
            "Body shape": "BODY SHAPE",
            "Hairs": {
                "Hair color": "HAIR COLOR",
                "Hair shape": "HAIR SHAPE",
            },
            "Beards": {
                "Beard color": "BEARD COLOR",
                "Beard thickness": "BEARD THICKNESS",
                "Beard style": "BEARD STYLE",                                   
                "Beard shape": "BEARD SHAPE",
            },
            "Scars": "SCARS"  
            
        }
        
        self.recAlgCBWeight = cbVal
        self.recAlgCBFeatureWeight = normalize(np.asarray(cbCoefs).reshape(1, -1))[0]
        self.recAlgSuspectWeight = 0.5
        self.learningRate = 0.1
        
        #print(self.recAlgCBFeatureWeight[0:5])
    

    def storeLog(self, logEntry):
        ip = ""#str(os.environ['REMOTE_ADDR'])
        dttime = str(datetime.now())
        with open("log.csv", "a") as procFile:
            procFile.writelines(["%s;%s;%s \n" % (dttime, ip, logEntry)])




    def render(self, tpl_path, context):
        path, filename = os.path.split(tpl_path)
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(path or './')
        ).get_template(filename).render(context)

    def getRandObjects(self, df, k=120, seed = 1):
        data = df["id"].tolist()


        #print(data[0:10])
        random.seed(seed)
        random.shuffle(data)
        return data[0:k]

    def getSelectedObjects(self, ids, k=20):
        d = pd.read_csv("sourceData/personsData.csv", sep=';', header=0)
        dt_cb = np.asarray(d)
        data = dt_cb[:,0].tolist()

        #print(data[0:10])
        #random.seed(seed)
        #random.shuffle(data)
        return data[0:k] #0:k


    # method defining similmilarity of patches based on their metric distance
    def getDistanceWeightVector(self, patchID, patchSize="3x4", imgWidth=140, imgHeight=175):
        import sklearn.metrics.pairwise as dst
        maxDist = dst.euclidean_distances([[0, 0]], [[imgWidth, imgHeight]])

        patchGrid = np.mgrid[0:4, 0:3]

        stepX = imgWidth // 3
        stepY = imgHeight // 4

        patchGrid[0] = patchGrid[0] * stepY + round(stepY / 2)
        patchGrid[1] = patchGrid[1] * stepX + round(stepX / 2)

        x = patchGrid[0].flatten()
        y = patchGrid[1].flatten()
        patchCentres = np.array(list(zip(x, y)))

        patchIDCentre = patchCentres[int(patchID)]
        dstVector = dst.euclidean_distances([patchIDCentre], patchCentres)

        patchSim = 1 - (dstVector / maxDist)
        return patchSim.T


    # method for defining similarity of object with other objects
    def aggregateSimilarity(self, simMetric, method="max"):
        # print(len(simMetric))
        if method == "mean":  # global average
            sim = []
            for pos1, positions in simMetric.items():
                sim.extend(list(positions.values()))
            arr = np.asarray(sim)
            # print(arr.shape)
            return np.mean(arr, axis=0)

        elif method == "mean_of_max":  # average of maximums
            sim = []
            for pos1, positions in simMetric.items():
                listOfSimilarities = np.asarray(list(positions.values()))
                #print(listOfSimilarities.shape)
                amax = np.amax(listOfSimilarities, axis=0)
                #print(amax.shape)
                sim.append(amax)

            # print(np.mean(np.asarray(sim), axis=0).shape)

            return np.mean(np.asarray(sim), axis=0)

        elif method == "max":  # global maximum
            sim = []
            for pos1, positions in simMetric.items():
                sim.extend(list(positions.values()))
            arr = np.asarray(sim)
            # print(arr.shape)
            return np.amax(arr, axis=0)

        elif method == "distwise_mean_of_max":  # average of maximums weighted by the distance from the source patch

            sim = []
            for pos1, positions in simMetric.items():
                weightVector = self.getDistanceWeightVector(patchID=pos1)

                listOfSimilarities = np.asarray(list(positions.values()))
                listOfSimilarities = listOfSimilarities * weightVector  # weight examples by distance

                # print(listOfSimilarities.shape, weightVector.shape)
                amax = np.amax(listOfSimilarities, axis=0)
                # print(amax.shape)
                sim.append(amax)
            # print(np.mean(np.asarray(sim), axis=0).shape)
            return np.mean(np.asarray(sim), axis=0)


    # method returns similarity vector for all images based on global descriptors
    def getObjectsByGlobal(self, objects, similaritiesDir, ids, descriptor="alexnet_fc6.csv", k=20, method="sum", recPos="", suspectWeight=0.5):
        sumSim = np.zeros((len(objects), len(ids)))
        #print(objects)
        #print(sumSim.shape)
        i = 0
        for f in ids:            
            
            # print simFile
            # print isfile(join(similaritiesDir, simFile))
            try:
                fid = objects.index.get_loc(np.int64(f))            
                simFile = f + ".npy"
                simMetric = np.load(join(similaritiesDir, simFile))
                simMetric[fid] = 0.0  # remove query from results
                #print("SimMetricGlobal shape")
                #print(simMetric.shape)
                sumSim[:, i] = simMetric[:, 0]
                i = i + 1
            except:
                pass

        if method == "sum":
            sumSim = np.mean(sumSim, axis=1)
        elif method == "max":
            sumSim = np.amax(sumSim, axis=1)
        elif method == "min":
            sumSim = np.amin(sumSim, axis=1)
        elif method == "suspectWeightRecommendation":
            othersWeight = (1 - suspectWeight) / len(ids)
            sumSim[0] = sumSim[0] *  suspectWeight
            if(len(ids) > 1):
                sumSim[1:] = sumSim[1:] * othersWeight                            
            sumSim = np.mean(sumSim, axis=1)
        sumSim = np.nan_to_num(sumSim, copy=False)
        
        #print(sumSim)
        return sumSim


    #method returns similarity vector for all images based on selected patches
    def getObjectsByPatches(self, objects, ids, k=20, localAggregationMethod = "mean_of_max", recPos = "", whiteRemovalThreshold=-1 ):
        from collections import defaultdict

        # recPos is string in shape oid1:patch1;oid2:patch2

        recPosArr = recPos.split(";")
        relPos = defaultdict(list)
        
        for rp in recPosArr:
            try:
                rp = rp.split(":")
                relPos[rp[0]].append(rp[1])
            except:
                pass
                
        #print(relPos)
        
        sumSim = np.zeros((len(objects), len(ids)))
        i = 0
        allPos = range(12)
        # recPos is a dictionary of fid -> [list of relevant positions]
        cropedSimDir = "descriptorsPatchesSIM"
        whitePercDir = "WhitePercentage"
        for f in ids:
            #objects.index.astype(np.str, copy=False)
            
            simMetric = {}          
            fid = objects.index.get_loc(np.int64(f))
            relposFID = relPos[f]
            #print([relposFID, fid, f])
  
            """
            if whiteRemovalThreshold != -1:
                  dfThr = pd.DataFrame({"ObjectId":[], "PatchId":[]} )
                  #------------filter out similarities based on white patch removal
                  for t1 in range(whiteRemovalThreshold, 105, 5):
                      t2 = t1+5
                      if t2 > 100:
                          t2 = 100
                      #print (t1, t2)
                      dfThr = dfThr.append(pd.read_csv(whitePercDir+"/_percentage-of-white-from-"+str(t1)+"-to-"+str(t2)+".csv", header=0, sep=",", usecols=[1,2] ), ignore_index = True)
  
                  dfThr["oids"] = [objects.index(obj) if obj in objects else None for obj in dfThr.ObjectId]
                  dfThr = dfThr.loc[dfThr['oids'] != None]
                  dfThr.dropna(inplace=True)
                  dfThr[["oids", "PatchId"]] = dfThr[["oids", "PatchId"]].astype(int, copy=False)
                  #print(dfThr.head())
              """
              #------------get local similarities for each patch
            for pos1 in relposFID:
                  simMetric[pos1] = {}
                  for pos2 in allPos:  
                      simFile = str(pos2) + "_" + str(pos1) + "_" + f + ".npy"  # change filename                                                             
                      try:                      
                          simMetric[pos1][pos2] = np.load(join(cropedSimDir, simFile))  # obalit vyjimkou
                          # np.savetxt(str(f)+"_"+str(pos1)+"_"+str(pos2)+"test.csv",simMetric[pos1][pos2], fmt='%.6f', delimiter=';' )
                          simMetric[pos1][pos2][fid] = 0.0  # remove query from results
                          """if whiteRemovalThreshold != -1:
                              dfThrMet = dfThr.loc[dfThr['PatchId'] == pos2]
                              simMetric[pos1][pos2][dfThrMet.oids] = 0.0
                          """
                      except:
                          #print(simFile+" not found") 
                          pass

            simVectorForFID = self.aggregateSimilarity(simMetric, method=localAggregationMethod)
            sumSim[:, i] = simVectorForFID

            #np.savetxt(str(f) + "test.csv", simVectorForFID, fmt='%.6f', delimiter=';')
            i = i + 1

        sumSim = np.mean(sumSim, axis=1)
        sumSim = np.nan_to_num(sumSim, copy=False)
        #print(sumSim)
        return sumSim


    def getRecommendedVisual(self, lineupSuspect, fmembers, suspectWeight, k=50):
        similaritiesDir = "descriptorsSIM/"
        ids = []
        
        if len(lineupSuspect) > 0 :
            ids.append(lineupSuspect)
        
        if (len(fmembers)>0) and (fmembers != [""]):
            ids.extend(fmembers)
        
        if len(ids) > 0:
        
            objects = pd.read_csv("objects.csv", sep=';', header=None, index_col=0, dtype=np.str) #order of objectIDs
            sumSim = self.getObjectsByGlobal( objects, similaritiesDir, ids, "", k, "suspectWeightRecommendation", "", suspectWeight)
            sumSim = normalize(np.expand_dims(sumSim, axis=0))[0,:]
            
            ind_vis = np.argsort(sumSim)[-k:]
            recIDs = [str(objects.index.values[idU]) for idU in ind_vis]
            recSim = [round(sumSim[idU], 5) for idU in ind_vis]
        
            return recIDs, recSim, list(range(1,k+1)) 
        return [], [], []   

    def getRecommendedCB(self, lineupSuspect, fmembers, suspectWeight, cbFeatures, k=20):
        ids = []
        
        fmembers = [f for f in fmembers if f != "" ]
        
        if len(lineupSuspect) > 0 :
            ids.append(lineupSuspect)
                
        if (len(fmembers)>0) and (fmembers != [""]):
          ids.extend(fmembers)  
                        
        objects = pd.read_csv("objects.csv", sep=';', header=None, index_col=0, dtype=np.str)
        #print(type(cbFeatures))
        allObjectsArray = cbFeatures.values * self.recAlgCBFeatureWeight

        
        results = np.zeros((len(ids), len(objects)))
        
        for (i, idx) in enumerate(ids):
        
          idsVector = cbFeatures.loc[objects.index.get_loc(np.int64(idx)),:].values * self.recAlgCBFeatureWeight
          if len(idsVector.shape) > 1: #check for duplicate keys
              idsVector = idsVector[0,:]
          idsVector = np.expand_dims(idsVector, axis=0)
          
          #print(idsVector[0:100])
          #TODO: add importance (features)
          
          
          result = 1 - distance.cdist(allObjectsArray, idsVector, "cosine")
          #result =  np.dot(allObjectsArray, idsVector) 
          results[i,:] =  np.squeeze(result)
        
          print(result.shape)
        
        
        othersWeight = (1 - suspectWeight) / len(ids)
        results[0] = results[0] *  suspectWeight
        if(len(ids) > 1):
           results[1:] = results[1:] * othersWeight 
        
        sumSim = np.mean(results, axis=0)
        sumSim = normalize(np.expand_dims(sumSim, axis=0))[0,:]
        
        print(sumSim.shape)  
        ind_vis = np.argsort(sumSim)[-k:]
        recIDs = [str(objects.index.values[idU]) for idU in ind_vis]
        recSim = [round(sumSim[idU], 5) for idU in ind_vis]
        
        return recIDs, recSim, list(range(1,k+1))    

        
         
        
        

    # return top-k most similar items to the currently selected ones
    def getSimilarObjects(self, ids, descriptor="alexnet_fc6.csv", k=20, method="sum", localAggregationMethod="mean_of_max",
                          localGlobalRatio="0.1", whiteRemovalThreshold="100", recPos=""):
        descriptorDir = "descriptorsCSV/"
        similaritiesDir = "descriptorsSIM/"
        additionalHTML = ""
        # descriptors = pd.read_csv(descriptorDir+descriptor, sep=';', header=None)
        whiteRemovalThreshold = int(whiteRemovalThreshold)
        lgr = float(localGlobalRatio)
        if lgr <= 0.01:
            method = "local"
        elif lgr >= 0.99:
            method = "global"

        if method == "local":
            objects = pd.read_csv("objects_0.csv", sep=';', header=None, index_col=0, dtype=np.str) #order of objectIDs
            sumSim = self.getObjectsByPatches(objects, ids, k, localAggregationMethod, recPos, whiteRemovalThreshold)

        elif method == "global":
            objects = pd.read_csv("objects.csv", sep=';', header=None, index_col=0, dtype=np.str) #order of objectIDs
            sumSim = self.getObjectsByGlobal( objects, similaritiesDir, ids, "alexnet_fc6.csv", k, "sum", recPos)

        elif method == "sum_and_crop":
            lgr = float(localGlobalRatio)

            objects = pd.read_csv("objects_0.csv", sep=';', header=None, index_col=0, dtype=np.str) #order of objectIDs
            sumSim1 = self.getObjectsByPatches(objects, ids, k, localAggregationMethod, recPos, whiteRemovalThreshold)
            #print(sumSim1.shape)

            objects2 = pd.read_csv("objects.csv", sep=';', header=None, index_col=0, dtype=np.str) #order of objectIDs
            sumSim2 = self.getObjectsByGlobal(objects2, similaritiesDir, ids, "alexnet_fc6.csv", k, "sum", recPos)
            #print(sumSim2.shape)

            sumSim = []
            for (idx, obj) in enumerate(objects.index.values.tolist()):
                o1Loc = objects2.index.get_loc(obj)
                if type(o1Loc) == int:
                    sumSim.append(((1 - lgr) * sumSim1[idx] + lgr * sumSim2[o1Loc]))
                else:
                    sumSim.append(((1 - lgr) * sumSim1[idx] + lgr * sumSim2[o1Loc][0]))              
            
            
            #print(type(sumSim))
            #print(sumSim)
            sumSim = np.asarray(sumSim)
            # np.savetxt("test2.csv", sumSim, fmt='%.6f', delimiter=';')
            #print(sumSim.shape)
        if k <= 0:
            ind_vis = np.argsort(sumSim)
        else:
            ind_vis = np.argsort(sumSim)[-k:]
            # print(ind_vis)
            # np.savetxt("test3.csv", ind_vis, fmt='%.6f', delimiter=';')
            # np.savetxt("test4.csv", sumSim[ind_vis], fmt='%.6f', delimiter=';')

        # print(objects)
        foto_vis = [str(objects.index.values[idU]) for idU in ind_vis]

        # print(ind_vis.shape, len(foto_vis))

        # print(foto_vis)
        if method == "sum":
            foto_sim = [str(round(sumSim[idU] / len(ids), 5)) for idU in ind_vis]
        else:
            foto_sim = [str(round(sumSim[idU], 5)) for idU in ind_vis]

        # print foto_vis
        # print foto_sim

        # print([ind_vis, foto_vis, foto_sim])

        return foto_vis, foto_sim, additionalHTML


    def getPatchSelectionForImage(self, cls, itClass, o, htmlParams, chck):
        scr = """  
            <script>   
            /*  
                $(document).ready(function () {           
                    //Canvas
                    var canvas = document.getElementById('canvas_""" + o + """');
                    var ctx = canvas.getContext('2d');
                    //Variables
                    var canvasx = $(canvas).offset().left;
                    var canvasy = $(canvas).offset().top;
                    var mousedown = false;                        
                    //Mousedown
                    $(canvas).on('mousedown', function(e) {
                        var canvasx = $(canvas).offset().left;
                        var canvasy = $(canvas).offset().top;     
    
                        var xOffset=Math.max(document.documentElement.scrollLeft,document.body.scrollLeft);
                        var yOffset=Math.max(document.documentElement.scrollTop,document.body.scrollTop);
    
                        var mousex = parseInt(e.clientX-canvasx + xOffset);
                        var mousey = parseInt(e.clientY-canvasy + yOffset);                    
    
                        var boxX = Math.floor(mousex / stepSizeX);
                        var boxY = Math.floor(mousey / stepSizeY);
    
                        var box = boxX + (boxNoX * boxY)
                        if(box < 0){
                            //error by wrong calculation of the offset - add box 0 to be able to recover from error
                            box = 0;
                        }
    
                        // check whether the box was already selected. If so, deselect and remove color frame
                        rpVal = document.getElementById("recPos").value;
    
                        if(rpVal.indexOf(\"""" + o + """:\" + box.toString() + ";") != -1){
                            //already selected, deselect
                            rpVal = rpVal.replace(\"""" + o + """:\" + box.toString() + ";", "");
                            color = 'rgba(255, 255, 255, 1.0)';
                            line = 2;
                        }else{
                            //newly selected patch
                            rpVal =  rpVal + \"""" + o + """:\" + box.toString() + ";"; 
                            color = 'rgba(255, 0, 0, 1.0)';
                            line = 1.5;
                        }
    
                        document.getElementById("recPos").value = rpVal;
    
                        boxPosX = Math.round(boxX * stepSizeX);
                        boxPosY = Math.round(boxY * stepSizeY);              
                        alpha = 1.0;
    
                        drawRectangle(canvas, ctx, alpha, color, line, boxPosX, boxPosY, stepSizeX, stepSizeY);
    
    
                        $('#output').html('current: '+mousex+', '+mousey+'<br/>box: '+boxX+', '+boxY+' => '+box);
    
                    });                                
    
                }); 
                */
                selectedObjects.push(\"""" + o + """\");                       
            </script>        
        """

        imgHTML = """<canvas id='canvas_""" + o + """' class="canvas" width="140" height="175" data-image="http://herkules.ms.mff.cuni.cz/lineit_v2/static/foto/""" + o + """.jpg"></canvas>"""

        html = scr + "<div class='itemCell" + cls + "' id='" + itClass + o + "'>" + imgHTML + htmlParams + " <br/><input type='checkbox' name='ids' value='" + o + "' " + chck + " title='Keep object in the search query'/>Keep in Query \n"
        #<input type='button' class='addButton nodisplay' value='Add to VA' id='va_" + o + "' /> <input type='button' class='addToOutfitButtonMain nodisplay' value='Add to Cart' id='outfit_" + o + "' /><input type='button' class='showDetailsButton' value='Details' id='details_" + o + "' />
        return html


    def getHTML(self, df, d, objects, selected=False, simList = [], selectedItems = [], type="resultList", recPos = ""):
        ids = df["id"].tolist()
        fNames = list(df)

        i = 0

        html = ""
        additionalHTML = ""

        itClass = "item_"
        if selected == True:
            itClass = "itemSelected_"
            boxNoX = 3
            boxNoY = 4
            globalScript = """
                <script>
                var boxNoX = """ + str(boxNoX) + """;
                var boxNoY = """ + str(boxNoY) + """;
                var imgWidth = 140;
                var imgHeight = 175; 
    
                var stepSizeX = imgWidth / boxNoX;
                var stepSizeY = imgHeight / boxNoY;
    
                </script>           
            """
            html += globalScript

            additionalHTML = """            
                <div id="output"></div>
                <input type="hidden" id="recPos" name="recPos" value='""" + recPos + """'/>       
            """

        if  type == "outfitCatList":
            itClass = "itemInOutfit_"
        for o in objects:
            dtID = ids.index(o)
            dtRow = df.iloc[[dtID]].values.tolist()
            dtRow = sum(dtRow, [])    #remove outer list, make it 1D

            trueIndeces = [index for index, value in enumerate(dtRow) if float(value) > 0.0]
            objectFeatures = [fNames[i] for i in trueIndeces]
            #print(i)
            #print(objectFeatures)
            tags =  ", ".join(objectFeatures)
            tg = ""


            if tags != "nan":
              tg = "<p>Tags: " +  tags + "</p>"

            htmlParams = """ 
              <div class="hide info">
                    """+tg+"""
              </div>      
            """

            cls = ""
            if o in  selectedItems:
                cls = " queryIdentity"
            if selected:
                chck =   " checked='checked'"
            else:
                chck =   ""


            if selected == True:
                # pass #TODO show code here
                html += self.getPatchSelectionForImage(cls, itClass, o, htmlParams, chck)
            elif type == "resultList":
                #<input type='checkbox' name='ids' value='"+o+"' "+chck+" title='Add object into the search query'/><input type='button' class='addToOutfitButtonMain nodisplay' value='Add to Cart' id='outfit_"+o+"' /><input type='button' class='showDetailsButton' value='Details' id='details_"+o+"' />
                html += "<div class='itemCell"+cls+"' id='"+itClass+o+"'><image class='obj' width=155 height=198 src='http://herkules.ms.mff.cuni.cz/lineit_v2/static/foto/"+o+".jpg' />"+htmlParams+" <br/> <input type='button' id='addID_"+o+"' class='AddToQueryButton' value='Query' title='Add person into the search query'/> <input type='button' class='addButton nodisplay' value='Add to Lineup' id='va_"+o+"' title='Add this person to the lineup as filler' /><br/><input type='button' id='suspectLink_"+o+"' class='suspectLink' value='Select as Suspect' title='Select this person as suspect' style='margin-top:3px;'/> \n"
            elif type == "outfitCatList":
                #<input type='checkbox' name='ids' value='"+o+"' "+chck+" title='Add object into the search query'/><input type='button' class='addToOutfitButton' value='Add to Cart' id='"+o+"' /><br/><br/> <input type='button' class='showDetailsButtonOutfit' value='Details' id='details_"+o+"' />
                html += "<div class='itemCell"+cls+"' id='"+itClass+o+"'><image class='objOutfit' width=80 height=105 src='http://herkules.ms.mff.cuni.cz/lineit_v2/static/foto/"+o+".jpg' />"+htmlParams+" <br/><input type='button' id='addID_"+o+"' class='AddToQueryButton' value='Query' title='Add object into the search query'/>\n"



            if len(simList)>0:
                html += "<br/> sim:"+str(simList[i])+""
            html += "</div>"
            i = i+1
        html += additionalHTML
        return Markup(html)

    def loadExample(self, exID):
        """d = pd.read_csv("queryExamples.csv", sep=';', header=0, dtype=str)
        dt = np.asarray(d)
        exID = int(exID)
        return (dt[exID, 0], dt[exID, 2], dt[exID, 1], dt[exID, 3], dt[exID, 4])
        """
        return ""

    def getExampleQueries(self, selectedID = 0):
        """d = pd.read_csv("queryExamples.csv", sep=';', header=0, dtype=str)
        dt = np.asarray(d)
        names = dt[:,0].tolist()
        i = 0
        html = ""
        for n in names:
            if i == selectedID:
                html += "<option value='"+str(i)+"' selected='selected'>"+str(n)+"</option>\n"
            else:
                html += "<option value='"+str(i)+"'>"+str(n)+"</option>\n"
            i = i+1
        return Markup(html)
        """
        return ""

    def getDescriptors(self, dsc = ""):
      """
        descriptorsDir = "descriptorsCSV/"
        descriptors = [f for f in listdir(descriptorsDir) if isfile(join(descriptorsDir, f))]
        html = ""
        for d in descriptors:
            if d == dsc:
                html += "<option value='"+d+"' selected='selected'>"+d+"</option>\n"
            else:
                html += "<option value='"+d+"'>"+d+"</option>\n"
        return Markup(html)
      """
      return ""
      
    def processMenuOptions(self, options, outputIDs, df, sc, catName):
                html1 = ""
                
                lastOption = ""
                selected = ""
                liClass = ""
                catMembers = 0   
                optionIDs = [] 
                
                
                
                for o in options:
                    if (o[1] != lastOption) and (lastOption != ""):                                                
                        html1 += "<li class='checkbox"+liClass+"'><input" + selected + " type='checkbox' class='catsBox " + catName.replace(" ",
                          "_") + "' name='"+str("|".join(optionIDs))+"' value='1'><a href='#'  class='navLink' id='" + str(
                          "|".join(optionIDs)) + "'>" + lastOption + "</a> (" + str(catMembers) + ")</li>\n"
                          
                    if (o[1] != lastOption):    
                        lastOption = o[1]
                        selected = ""
                        liClass = ""
                        catMembers = 0   
                        optionIDs = []                 

                
                    cm = list(set(outputIDs[(df.ix[:, int(o[0])] > 0)]))
                    catMembers = catMembers + len(cm)
                    optionIDs.append(str(o[0]))
                    if str(o[0]) in sc:
                        selected = " checked='checked'"
                        liClass = " checked"
                
                
                html1 += "<li class='checkbox"+liClass+"'><input" + selected + " type='checkbox' class='catsBox " + catName.replace(" ",
                          "_") + "' name='"+str(o[0])+"' value='1'><a href='#'  class='navLink' id='" + str(
                          "|".join(optionIDs)) + "'>" + lastOption + "</a> (" + str(catMembers) + ")</li>\n"  
                          
                return html1
                

    def getCategories(self, df, cats, selectedCats=""):

        html = ""

        sc = re.split('[,|]', selectedCats)

        groupNames = list(cats.keys())

        outputIDs = df["id"]


        outputLev1Names = list(self.menuShape.keys())
        for l1Name in outputLev1Names:
            html += "<h3>" + l1Name + "</h3>\n"

            if(isinstance(self.menuShape[l1Name], str)):#simple menu item
                options = cats[self.menuShape[l1Name]]
                options = sorted(options, key=lambda x: x[1])
                
                html1 = self.processMenuOptions(options, outputIDs, df, sc, l1Name)
                
                     

                html1 = "<ul>" + html1 + "</ul>\n"

            else: #composed menu item - iterate over lev2 dict
                html1 = ""
                l2Names = list((self.menuShape[l1Name]).keys())
                for l2Name in l2Names:
                    html1 += "<h4>" + l2Name + "</h4>\n<ul>\n"
                    options = cats[self.menuShape[l1Name][l2Name]]
                    options = sorted(options, key=lambda x: x[1])
                    
                    html1 += self.processMenuOptions(options, outputIDs, df, sc, l2Name)
   
                    html1 += "</ul>\n"

                html1 = "<div>" + html1 + "</div>\n"
            html += html1

        html = "<div class='menuAccordion'>"\
               +html+\
               "</div>" \
               "<input type='submit' id='submit_cbSearch' name='actionType' value='Filter Candidates'>"

        return html

    def updateCBWeights(self, cbFeatures, cats, selectedCats):
        #normalize((np.zeros(cbFeatures.shape[1]) +1).reshape(1, -1) )[0]
        sc = re.split('[,|]', selectedCats)
        sc = [int(c) for c in sc]
        
        self.recAlgCBFeatureWeight[sc] = (1 + self.learningRate) * self.recAlgCBFeatureWeight[sc]
        self.recAlgCBFeatureWeight = normalize(self.recAlgCBFeatureWeight.reshape(1, -1) )[0]
        
        with open("sourceData/cbCoefs.csv", "w") as f:
            f.write(";".join([str(i) for i in self.recAlgCBFeatureWeight]))
        
        print(self.recAlgCBFeatureWeight[sc])
        print(self.recAlgCBFeatureWeight[0:5])

    def getNamesFromCategories(self, selectedCats, cats):
        #js = open('sourceData/featureVals.json')
        #cats = json.load(js, encoding="utf-8")

        html = ""

        sc = re.split('[,|]', selectedCats)

        html1 = []
        outputLev1Names = list(self.menuShape.keys())
        for l1Name in outputLev1Names:
            if(isinstance(self.menuShape[l1Name], str)):#simple menu item
                options = cats[self.menuShape[l1Name]]
                options = sorted(options, key=lambda x: x[1])

                selectedOptions = []
                for o in options:
                    if str(o[0]) in sc:
                        selectedOptions.append(o[1])
                
                if(len(selectedOptions)>0):
                    selectedOptions = list(set(selectedOptions))
                    html1.append(l1Name+" = {"+", ".join(selectedOptions)+"}")

            else: #composed menu item - iterate over lev2 dict

                l2Names = list((self.menuShape[l1Name]).keys())
                for l2Name in l2Names:
                    options = cats[self.menuShape[l1Name][l2Name]]
                    options = sorted(options, key=lambda x: x[1])

                    selectedOptions = []
                    for o in options:
                        if str(o[0]) in sc:
                            selectedOptions.append(o[1])

                    if (len(selectedOptions) > 0):
                        selectedOptions = list(set(selectedOptions))
                        html1.append(l2Name + " = {" + ", ".join(selectedOptions)+"}")


        html += ";<br/>".join(html1)
        return html

    def getObjectsByCategory(self, cats, df):

        # print(df.ix[0:10, 0:10])

        outputIDsOrig = df["id"]
        outputIDs = outputIDsOrig.copy()
        catAND = cats.split(",")
        for ca in catAND:
            catOR = ca.split("|")
            if len(catOR) == 1:
                outputIDs = list(set(outputIDs).intersection(set(outputIDsOrig[(df.ix[:, int(ca)] > 0)])) )
            else:
                currOutput = []
                for co in catOR:
                    currOutput.extend(outputIDsOrig[(df.ix[:, int(co)] > 0)])
                outputIDs = list(set(outputIDs).intersection(set(currOutput)))
        print(len(outputIDs))
        return outputIDs


    def getSelectWeight(self, fweight):
        vals = ["plain", "max","min","std"]
        names = ["No weighting", "Max","Min","Std. dev."]
        html = ""
        for i in range(0, len(vals)):
            if fweight == vals[i]:
                html += "<option selected='selected' value='"+vals[i]+"'>"+names[i]+"</option>"
            else:
                html += "<option value='"+vals[i]+"'>"+names[i]+"</option>"
        return html

    def getSelectAgg(self, fweight):
        vals = ["sum", "max","min"]
        names = ["Mean", "Max","Min"]
        html = ""
        for i in range(0, len(vals)):
            if fweight == vals[i]:
                html += "<option selected='selected' value='"+vals[i]+"'>"+names[i]+"</option>"
            else:
                html += "<option value='"+vals[i]+"'>"+names[i]+"</option>"
        return html

    def getSuspects(self, ids = []):
        html = "<h3>Predefined Suspects</h3><ul>"
        for id in ids:
            html += "<li><a class='suspectLink' id='suspect_"+id+"' href='#'><img src='http://herkules.ms.mff.cuni.cz/lineit_v2/static/foto/"+id+".jpg' height=20>"+id+"</a>\n"

        html += "</ul>\n"
        return Markup(html)


    def getSelectLocalAgg(self, fweight):

        vals = ["mean_of_max", "max","mean","distwise_mean_of_max"]
        names = ["Mean of max", "Max","Mean","Mean of distance-weighted max"]
        html = ""
        for i in range(0, len(vals)):
            if fweight == vals[i]:
                html += "<option selected='selected' value='"+vals[i]+"'>"+names[i]+"</option>"
            else:
                html += "<option value='"+vals[i]+"'>"+names[i]+"</option>"
        return html

    def getSelectGlobalLocalWeight(self, fweight):

        vals = ["0.0", "0.1", "0.25","0.5","0.75", "0.9", "1.0"]
        names = ["0.0", "0.1", "0.25","0.5","0.75", "0.9", "1.0"]
        html = ""
        for i in range(0, len(vals)):
            if fweight == vals[i]:
                html += "<option selected='selected' value='"+vals[i]+"'>"+names[i]+"</option>"
            else:
                html += "<option value='"+vals[i]+"'>"+names[i]+"</option>"
        return html

    def getSelectWhitePatchRemoval(self, fweight):

        vals = ["-1","100","90","80","70","60","50","40","30","20","10"]
        names = ["-1","100","90","80","70","60","50","40","30","20","10"]
        html = ""
        for i in range(0, len(vals)):
            if fweight == vals[i]:
                html += "<option selected='selected' value='"+vals[i]+"'>"+names[i]+"</option>"
            else:
                html += "<option value='"+vals[i]+"'>"+names[i]+"</option>"
        return html



    def asort(self, d):
         return sorted(d.items(), key=lambda x: x[1])
    
    def updateRecommenderWeights(self, addedCBVal, addedVISVal):
        
        self.recAlgCBWeight = self.recAlgCBWeight + self.learningRate * (addedCBVal - addedVISVal)
        
        #set borders of the weight
        if self.recAlgCBWeight > 0.95:
            self.recAlgCBWeight = 0.95
        if self.recAlgCBWeight < 0.05:
            self.recAlgCBWeight = 0.05
            
        with open("sourceData/cbVals.csv", "w") as f:
            f.write(str(self.recAlgCBWeight))
      
    
    
    def getRecommendedCandidates(self, lineupSuspect, fmembers, cbFeatures):
        #TODO: get data about suspect and other members
        print(lineupSuspect)
        print(fmembers)
        ids = []
        if len(lineupSuspect) > 0 :
            ids.append(lineupSuspect)
        
        if (len(fmembers)>0) and (fmembers != [""]):
            ids.extend(fmembers)
        
        if len(ids) > 0:  #We have some objects to base our query on
            
            alpha = self.recAlgCBWeight
            suspectWeight = self.recAlgSuspectWeight
            cbFeatureWeights = self.recAlgCBFeatureWeight
            
            #TODO: update data on user action
            
            #TODO: calculate similarity
            html = ""
            candidateID, candidateWeight, bordaCount = self.getRecommendedVisual(lineupSuspect, fmembers, suspectWeight, k=500)  
            
            candidateID2, candidateWeight2, bordaCount2 = self.getRecommendedCB(lineupSuspect, fmembers, suspectWeight, cbFeatures, k=500) 
     
            
            candidates = pd.DataFrame(data={"sim":candidateWeight,"bc":bordaCount, "id":candidateID})
            candidates = candidates.drop_duplicates(subset='id')
            candidates.set_index("id", inplace=True)
            candidates2 = pd.DataFrame(data={"sim":candidateWeight2,"bc":bordaCount2, "id":candidateID2})
            candidates2 = candidates2.drop_duplicates(subset='id')
            candidates2.set_index("id", inplace=True)
            
            #print(candidates)
            #print(candidates2)
            
            mergedCandidates = ((1-alpha) * candidates).add(alpha*candidates2, fill_value=0.0)
            mergedCandidates["CB_sim"] = candidates2.sim
            mergedCandidates["VIS_sim"] = candidates.sim
            mergedCandidates["CB_bc"] = candidates2.bc
            mergedCandidates["VIS_bc"] = candidates.bc
            mergedCandidates.fillna(0.0, inplace=True)
            #print(mergedCandidates)
            
            currVisVotes =  (1-alpha)
            currCBVotes = alpha
            #d'hondt voting scheme, for candidates from multiple parties (mandate is divided by position)
            
            k = 20
            cbID = 1.0
            visID = 1.0 
            html = ""     
            for i in range(k):
                 mergedCandidates["res"] =  (currCBVotes * mergedCandidates["CB_sim"]) + (currVisVotes * mergedCandidates["VIS_sim"])
                 idx = mergedCandidates["res"].argmax()            
                 cbFraction =  mergedCandidates["CB_sim"][idx] / (mergedCandidates["CB_sim"][idx] + mergedCandidates["VIS_sim"][idx])
                 visFraction = 1- cbFraction
                 cbID += cbFraction
                 visID += visFraction
                 currVisVotes =  (1-alpha) / visID
                 currCBVotes = alpha / cbID
                 #TODO: past weights of CB and Vis sim 
                 html += """<div class='itemCell' >
                              <image class='objOutfit' width=94 height=120 src='http://herkules.ms.mff.cuni.cz/lineit_v2/static/foto/"""+str(idx)+""".jpg' />
                              <br/><input type='button' id='addToLineupID_"""+str(idx)+"""' class='RecAddToLineupButton' value='Add to Lineup' title='Add object into the current lineup'/>
                              <br/><input type='button' id='addRecID_"""+str(idx)+"""' class='RecAddToQueryButton' value='Query' title='Add object into the search query'/>
                              <input type='hidden' id='recCBSim_"""+str(idx)+"""' value='"""+str(mergedCandidates["CB_sim"][idx])+"""' />
                              <input type='hidden' id='recVISSim_"""+str(idx)+"""' value='"""+str(mergedCandidates["VIS_sim"][idx])+"""' />
                            </div>\n"""
                 
                 mergedCandidates["CB_sim"][idx] = 0
                 mergedCandidates["VIS_sim"][idx] = 0           
            return html#+str(alpha)
        return ""

    def main(self, dtSuspects, df, d, cbFeatures, cats, ids, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, featureID, features,
             listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries, cart):
        selectWeight = self.getSelectWeight(fweight)
        selectAgg = self.getSelectAgg(qagg)
        selectLocalAgg = self.getSelectLocalAgg(qLagg)
        globalLocalWeight = self.getSelectGlobalLocalWeight(glW)
        whitePatchRemoval = self.getSelectWhitePatchRemoval(wpR)

        html = self.getHTML(df, d, self.getRandObjects(df))
        descriptors =""# getDescriptors(dsc)
        categories = self.getCategories(df, cats)
        suspects = self.getSuspects(dtSuspects)
        
        if len(lineupSuspect) > 0:
          recommendedCandidates = self.getRecommendedCandidates(lineupSuspect, fmembers, cbFeatures)
        else:
          recommendedCandidates = ""

        self.storeLog("Homepage view")
        return self.render('templates/index.html',
                      {"suspects": suspects, "cart": cart, "exampleQueries": exampleQueries, "fmembers": fmembers,
                       "lineupSuspect": lineupSuspect, "lineupNote": lineupNote, "lineupAuthor": lineupAuthor,
                       "featureID": featureID, "local_aggregation": selectLocalAgg, "aggregation": selectAgg,
                       "globalLocalWeight": globalLocalWeight, "whitePatchRemoval": whitePatchRemoval, "qRel": qRel, "weighting": selectWeight,
                       "listOfFeatures": listOfFeatures, "descriptors": descriptors,  "selected": self.getHTML(df, d,[], True),"objects": html,
                       "objectCategories": categories, "recommendedCandidates":recommendedCandidates })


    def search(self, dtSuspects, df, d, cbFeatures, cats, ids, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, featureID, features, listOfFeatures,
               fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries, cart, recPos):

        if isinstance(recPos, list):
            recPos = recPos[0]

        k = 50  # volume of shown images

        fotoVis, fotoSim, segmentsHTML = self.getSimilarObjects(ids, dsc, k, qagg, qLagg, glW, wpR, recPos)

        objAll = [i for i in reversed(fotoVis)]
        simAllFloat = [float(i) for i in reversed(fotoSim)]
        simAll = [i for i in reversed(fotoSim)]
        #print simAllFloat[0:10]
        meanSim = np.mean(simAllFloat)
        #showing outfit options
        idRes = []
        dt_cb = np.asarray(d)
        data = dt_cb[:,4].tolist()
        dataIDs = dt_cb[:,0].tolist()

        htmlOutfit = ""

        #showing top-K best fits
        obj = objAll[0:k]
        sim = simAll[0:k]

        selectWeight = self.getSelectWeight(fweight)
        selectAgg = self.getSelectAgg(qagg)
        selectLocalAgg = self.getSelectLocalAgg(qLagg)
        globalLocalWeight = self.getSelectGlobalLocalWeight(glW)
        whitePatchRemoval = self.getSelectWhitePatchRemoval(wpR)
        
        if len(lineupSuspect) > 0:
          recommendedCandidates = self.getRecommendedCandidates(lineupSuspect, fmembers, cbFeatures)
        else:
          recommendedCandidates = ""

        #print(obj)
        return self.render('templates/index.html', {"cart":cart, "featureSegments":segmentsHTML, "exampleQueries":exampleQueries,
                                               "outfitHTML":htmlOutfit,  "fmembers":fmembers,
                                               "lineupSuspect": lineupSuspect,"lineupNote": lineupNote, "lineupAuthor": lineupAuthor,
                                               "featureID": featureID,
                                               "features":features, "local_aggregation":selectLocalAgg,
                                               "aggregation":selectAgg, "globalLocalWeight":globalLocalWeight,
                                               "whitePatchRemoval":whitePatchRemoval, "qRel":qRel, "weighting":selectWeight,
                                               "listOfFeatures":listOfFeatures, "descriptors":self.getDescriptors(dsc),
                                               "selected":self.getHTML(df, d, ids, True, recPos = recPos), "objects":self.getHTML(df,  d, obj, False, sim, ids), "objectCategories":self.getCategories(df, cats), 
                                               "recommendedCandidates":recommendedCandidates })


    def cbsearch(self, dtSuspects, df, d, cbFeatures, cats, categoryIds, searchIds, dsc, fmembers, lineupSuspect, lineupNote, lineupAuthor, featureID, features,
                 listOfFeatures, fweight, qLagg, qagg, glW, wpR, qRel, exampleQueries, cart, navSubmit="", catNames="", recPos = ""):
        
        k = 50
                 
        if isinstance(recPos, list):
            recPos = recPos[0]  
        
        if len(searchIds) > 0:   
            fotoVis, fotoSim, segmentsHTML = self.getSimilarObjects(searchIds, dsc, -1, qagg, qLagg, glW, wpR, recPos)
            #objAll = [i for i in reversed(fotoVis)]
            #simAllFloat = [float(i) for i in reversed(fotoSim)]
            reversedFotoVis = [i for i in reversed(fotoVis)]
            reversedFotoSim = [i for i in reversed(fotoSim)]
            
            catRestrIDX = [idx for (idx, obj) in enumerate(reversedFotoVis) if obj in categoryIds]
            
            catRestrOBJ  = [reversedFotoVis[idx] for idx in catRestrIDX]
            catRestrSIM  = [reversedFotoSim[idx] for idx in catRestrIDX] 
            
            obj = catRestrOBJ[0:k]
            sim = catRestrSIM[0:k]
            
            html = self.getHTML(df, d, obj, False, sim, searchIds)                     
             
        else:
            html = self.getHTML(df, d, categoryIds, False)
            
        descriptors = ""  
        categories = self.getCategories(df, cats, navSubmit)
        suspects = self.getSuspects(dtSuspects)

        selectWeight = self.getSelectWeight(fweight)
        selectAgg = self.getSelectAgg(qagg)
        selectLocalAgg = self.getSelectLocalAgg(qLagg)
        globalLocalWeight = self.getSelectGlobalLocalWeight(glW)
        whitePatchRemoval = self.getSelectWhitePatchRemoval(wpR)
        
        if len(lineupSuspect) > 0:
          recommendedCandidates = self.getRecommendedCandidates(lineupSuspect, fmembers, cbFeatures)
        else:
          recommendedCandidates = ""
          
        self.storeLog("Category search:" + navSubmit)

        return self.render('templates/index.html',
                      {"suspects" : suspects, "navSubmit" : navSubmit, "navHeader" : catNames, "cart": cart, "exampleQueries": exampleQueries, "fmembers": fmembers,
                       "lineupSuspect": lineupSuspect,"lineupNote": lineupNote, "lineupAuthor": lineupAuthor, "featureID": featureID,
                       "features": features, "local_aggregation":selectLocalAgg, "aggregation":selectAgg, "globalLocalWeight":globalLocalWeight,
                       "whitePatchRemoval":whitePatchRemoval, "qRel":qRel,  "weighting": selectWeight,
                       "listOfFeatures": listOfFeatures, "descriptors": descriptors, "selected":self.getHTML(df, d, searchIds, True, recPos = recPos), "objects": html,
                       "objectCategories": categories, "recommendedCandidates":recommendedCandidates })



    def saveFeature(self, fmembers, lineupSuspect, lineupNote, lineupAuthor, lineupID):
        #print("lineupID:"+lineupID)

        if lineupID != "" and int(lineupID) >= 0 :
              lineupID = int(lineupID)
              dt = pd.read_csv("createdFeatures.csv", sep=';', header=None, dtype=str)
              dt[0][lineupID]  = lineupAuthor
              dt[1][lineupID]  = lineupSuspect
              dt[2][lineupID]  = lineupNote
              dt[3][lineupID]  = ",".join(fmembers)
              dt.to_csv(path_or_buf="createdFeatures.csv", sep=';', header=False, index=False)
              lineupID = str(lineupID + 1)
        else:
           with open("createdFeatures.csv", "a") as procFile:
              procFile.writelines(["%s;%s;%s;%s \n" % (lineupAuthor, lineupSuspect, lineupNote, ",".join(fmembers))  ])
           dt = pd.read_csv("createdFeatures.csv", sep=';', header=None, dtype=str)
           lineupID = str(dt.shape[0]-1)

        self.storeLog("Stored lineup:" + lineupID)
        return lineupID

    def saveQuery(self, querySummary):
        if querySummary != "":
            with open("queryResultsSummary.csv", "a") as procFile:
                procFile.writelines(["%s\n" % (querySummary)])

    def getQueryRelevance(self):
        dt = pd.read_csv("queryResultsSummary.csv", sep=';', header=None, dtype=str, quotechar="'")

        dt.columns = ['UID', 'featureID', 'glWeight', 'whiteRemoval', 'aggMethod', 'queryList', 'responseList',
                      'responseRelevance', 'patches']
        dt_group = dt.groupby(['featureID'])[["responseList", "responseRelevance"]].agg(lambda x: ','.join(x))
        dt_group.reset_index(0, inplace=True)
        dt_group["responseList"] = dt_group.responseList.str.split(",")
        dt_group["responseRelevance"] = dt_group.responseRelevance.str.split(",")

        rowTxt = []
        for i in range(dt_group.shape[0]):
            dt_group["responseList"][i] = [str(v) for idx,v in enumerate(dt_group["responseList"][i]) if dt_group["responseRelevance"][i][idx] == "1"]
            dt_group["responseList"][i] = list(set(dt_group["responseList"][i]))

            rowTxt.append("'"+str(dt_group["featureID"][i])+"': ['" +"','".join(dt_group["responseList"][i])+"']")

        return "{"+",\n".join(rowTxt)+"}"


    def createFeatureOption(self, id, name, author,  current):
        if current ==  id:
            return "<option value='"+str(id)+"' selected='selected'>Author: "+str(author)+", Suspect: "+str(name)+"</option>\n"
        return "<option value='"+str(id)+"'>Author: "+str(author)+", Suspect: "+str(name)+"</option>\n"
        return ""

    def createFeatureRecord(self, id, author, suspect, notes, members):
        return " '"+str(id)+"': ['"+str(author)+"','"+str(suspect)+"','"+str(notes)+"','"+str(members)+"']\n"
   