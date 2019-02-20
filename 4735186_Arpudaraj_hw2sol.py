# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 12:35:51 2019

@author: helena arpudaraj
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
import string
from string import punctuation

class nGram():
    def __init__ (self,file):
         s = open(file, "r").read().lower()
         self.st=re.sub("[^a-zA-Z0-9' \n]", " ", s)
         self.sentences=self.st.splitlines()
       
         for i in range(len(self.sentences)):
             self.sentences[i]=' '.join(self.sentences[i].split(' ')[:-1])
             
         stt=s.splitlines()
         self.actual=[]
         for sentence in stt:
             self.actual.append((int) (sentence.split()[-1]))
            
##         
    def findngram(self):

        self.no_of_total_sentences = len(self.sentences) 
        #1gram
        self.unique_words = []

          # generate unique words
        for sentence in self.sentences:
            words=[]
            words=sentence.split()
            for i in range(len(words)):
                if words[i] not in self.unique_words:
                    self.unique_words.append(words[i])
        self.unique_wordpairs=[]
        self.unique_3words=[]
        self.unique_4words=[]
        self.unique_5words=[]
        for sentence in self.sentences:
            words=[]
            words=sentence.split()
            for i in range(len(words)-1):
                if ([(words[i],words[i+1])]) not in self.unique_wordpairs:
                    self.unique_wordpairs.append([(words[i],words[i+1])])
            for i in range(len(words)-2):
                if ([(words[i],words[i+1],words[i+2])]) not in self.unique_3words:
                    self.unique_3words.append([(words[i],words[i+1],words[i+2])])
            for i in range(len(words)-3):
                if ([(words[i],words[i+1],words[i+2],words[i+3])]) not in self.unique_4words:
                    self.unique_4words.append([(words[i],words[i+1],words[i+2],words[i+3])])
            for i in range(len(words)-4):
                if ([(words[i],words[i+1],words[i+2],words[i+3],words[i+4])]) not in self.unique_5words:
                    self.unique_5words.append([(words[i],words[i+1],words[i+2],words[i+3],words[i+4])])
        
        self.unique_words.sort()
        self.unique_wordpairs.sort()
        self.unique_3words.sort()
        self.unique_4words.sort()
        self.unique_5words.sort()
        
        #n-gram
    def generate_cooccurrence_matrix(self):
       self.cooccurrence_matrix1=np.zeros((self.no_of_total_sentences ,len(self.unique_words)))
       self.cooccurrence_matrix2=np.zeros((self.no_of_total_sentences ,len(self.unique_wordpairs)))
       self.cooccurrence_matrix3=np.zeros((self.no_of_total_sentences ,len(self.unique_3words)))
       self.cooccurrence_matrix4=np.zeros((self.no_of_total_sentences ,len(self.unique_4words)))
       self.cooccurrence_matrix5=np.zeros((self.no_of_total_sentences ,len(self.unique_5words)))
       for i, sentence in enumerate(self.sentences):
           words=[]
           words=sentence.split()
           for k in range(len(words)):
               j=self.unique_words.index(words[k])
               self.cooccurrence_matrix1[i][j]=self.cooccurrence_matrix1[i][j]+1
           for k in range(len(words)-1):
               j=self.unique_wordpairs.index([(words[k],words[k+1])])
               self.cooccurrence_matrix2[i][j]=self.cooccurrence_matrix2[i][j]+1
           for k in range(len(words)-2):
               j=self.unique_3words.index([(words[k],words[k+1],words[k+2])])
               self.cooccurrence_matrix3[i][j]=self.cooccurrence_matrix3[i][j]+1
           for k in range(len(words)-3):
               j=self.unique_4words.index([(words[k],words[k+1],words[k+2],words[k+3])])
               self.cooccurrence_matrix4[i][j]=self.cooccurrence_matrix4[i][j]+1
           for k in range(len(words)-4):
               j=self.unique_5words.index([(words[k],words[k+1],words[k+2],words[k+3],words[k+4])])
               self.cooccurrence_matrix5[i][j]=self.cooccurrence_matrix5[i][j]+1
        
       #normalize
       for i in range(0,len(self.cooccurrence_matrix1)):
           count=0
           for j in range(0,len(self.cooccurrence_matrix1[i])):
               count=count + self.cooccurrence_matrix1[i][j]
           for j in range(0,len(self.cooccurrence_matrix1[i])):
               if(count>0):
                   self.cooccurrence_matrix1[i][j]=round(self.cooccurrence_matrix1[i][j]/count,5)
           count=0
           for j in range(0,len(self.cooccurrence_matrix2[i])):
               count=count + self.cooccurrence_matrix2[i][j]
           for j in range(0,len(self.cooccurrence_matrix2[i])):
               if(count>0):
                   self.cooccurrence_matrix2[i][j]=round(self.cooccurrence_matrix2[i][j]/count,5)
           count=0
           for j in range(0,len(self.cooccurrence_matrix3[i])):
               count=count + self.cooccurrence_matrix3[i][j]
           for j in range(0,len(self.cooccurrence_matrix3[i])):
               if(count>0):
                   self.cooccurrence_matrix3[i][j]=round(self.cooccurrence_matrix3[i][j]/count,5)
           count=0
           for j in range(0,len(self.cooccurrence_matrix4[i])):
               count=count + self.cooccurrence_matrix4[i][j]
           for j in range(0,len(self.cooccurrence_matrix4[i])):
               if(count>0):
                   self.cooccurrence_matrix4[i][j]=round(self.cooccurrence_matrix4[i][j]/count,5)
           count=0
           for j in range(0,len(self.cooccurrence_matrix5[i])):
               count=count + self.cooccurrence_matrix5[i][j]
           for j in range(0,len(self.cooccurrence_matrix5[i])):
               if(count>0):self.cooccurrence_matrix5[i][j]=round(self.cooccurrence_matrix5[i][j]/count,5)
       
       self.cooccurrence_matrix6=np.concatenate((self.cooccurrence_matrix1,self.cooccurrence_matrix2),1)         
       self.cooccurrence_matrix7=np.concatenate((self.cooccurrence_matrix6,self.cooccurrence_matrix3),1)
       self.cooccurrence_matrix8=np.concatenate((self.cooccurrence_matrix7,self.cooccurrence_matrix4),1)
       self.cooccurrence_matrix9=np.concatenate((self.cooccurrence_matrix8,self.cooccurrence_matrix5),1)
       
       
    def classification(self):
        #f1
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]

        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        
        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix1[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix1[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix1[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix1[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr=LogisticRegression()
            lr.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
                    
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f1")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
        
        #f2
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        precision=[]
        recall=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix2[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix2[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix2[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix2[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr2=LogisticRegression()
            lr2.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr2.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f2")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        

        #f3
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        precision=[]
        recall=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0

        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix3[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix3[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix3[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix3[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr3=LogisticRegression()
            lr3.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr3.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f3")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
##      
        #f4
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        precision=[]
        recall=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
       	TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0

        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix4[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix4[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix4[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix4[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr4=LogisticRegression()
            lr4.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr4.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
            
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f4")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
#       
        #f5
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        precision=[]
        recall=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        TPsum=0
        FPsum=0
        TNsum=0
        FNsum=0
        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix5[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix5[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix5[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix5[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr5=LogisticRegression()
            lr5.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr5.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
            #precision[i]=TP/(TP+FP)
            #print(TP/(TP+FP))
            #recall.append(TP/(TP+FN))
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10
        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f5")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
#       
#        
        #f1f2
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        precision=[]
        recall=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix6[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix6[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix6[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix6[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr6=LogisticRegression()
            lr6.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr6.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f1f2")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
#       
#       
        #f1f2f3
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]
        precision=[]
        recall=[]
        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0

        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix7[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix7[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix7[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix7[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr7=LogisticRegression()
            lr7.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr7.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f1f2f3")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
#       
#        
        #f1f2f3f4
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]

        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix8[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix8[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix8[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix8[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr8=LogisticRegression()
            lr8.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr8.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            TPsum=TPsum+TP
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f1f2f3f4")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        
#       
#        
        #f1f2f3f4f5
        folds=int(self.no_of_total_sentences/10)
        accuracy=[]

        TPR=[]
        TNR=[]
        FPR=[]
        FNR=[]
       	TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        for i in range(10):
            kfoldTest_data=self.cooccurrence_matrix9[(folds*i):(folds*(i+1))]
            kfoldTest_label=self.actual[(folds*i):(folds*(i+1))]
            if(i>0):
                kfoldTrain_Data=self.cooccurrence_matrix9[0:(folds*i)]
                kfoldTrain_label=self.actual[0:(folds*i)]
                if (i<9):
                    kfoldTrain_Data=np.append(kfoldTrain_Data,self.cooccurrence_matrix9[folds*(i+1):self.no_of_total_sentences],axis=0)    
                    kfoldTrain_label=np.append(kfoldTrain_label,self.actual[folds*(i+1):self.no_of_total_sentences],axis=0)    
            else:
                kfoldTrain_Data=self.cooccurrence_matrix9[folds*(i+1):self.no_of_total_sentences] 
                kfoldTrain_label=self.actual[folds*(i+1):self.no_of_total_sentences] 
    
            lr9=LogisticRegression()
            lr9.fit(kfoldTrain_Data, kfoldTrain_label)
            y_pred=lr9.predict(kfoldTest_data)
            accuracy.append(accuracy_score(kfoldTest_label, y_pred))
            cm = confusion_matrix(kfoldTest_label, y_pred)
            TP = cm[0][0]
            FN = cm[0][1]
            TN = cm[1][1]
            FP = cm[1][0]
        
            #true positive rate
            TPR.append(TP/(TP+FN))
            #true negative rate
            TNR.append(TN/(TN+FP)) 
            #false positive rate
            FPR.append(FP/(FP+TN))
            #False negative rate
            FNR.append(FN/(TP+FN))
            
            FPsum=FPsum+FP
            TNsum=TNsum+TN
            FNsum=FNsum+FN
        
        TPR_mean=sum(TPR)/len(TPR)
        TNR_mean=sum(TNR)/len(TNR)
        FNR_mean=sum(FNR)/len(FNR)
        FPR_mean=sum(FPR)/len(FPR)
        TP_mean=TPsum/10
        FN_mean=FNsum/10
        FP_mean=FPsum/10

        recall_mean=TP_mean/(TP_mean+FN_mean)
        precision_mean=TP_mean/(TP_mean+FP_mean)
        accuracy_mean=sum(accuracy)/len(accuracy)
        print("---------------------------------------------------------")    
        print("f1f2f3f4f5")
        print("---------------------------------------------------------")    
        print("False Positive Rate= ",FPR_mean)
        print("False Negative Rate= ",FNR_mean)
        print("True Positive Rate= ",TPR_mean)
        print("True Negative Rate= ",TNR_mean)
        print("Accuracy= ",accuracy_mean)  
        print("Precision= ",precision_mean)
        print("Recall= ",recall_mean)        

##--- RUN DATA --------------------------------------------------------------+

print("Dataset 1")
ngram = nGram("amazon_cells_labelled.txt")
ngram.findngram()
ngram.generate_cooccurrence_matrix()
ngram.classification()


print("Dataset 2")
ngram = nGram("imdb_labelled.txt")
ngram.findngram()
ngram.generate_cooccurrence_matrix()
ngram.classification()

print("Dataset 3")
ngram = nGram("yelp_labelled.txt")
ngram.findngram()
ngram.generate_cooccurrence_matrix()
ngram.classification()

##--- END ----------------------------------------------------------------------+