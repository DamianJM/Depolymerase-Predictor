#Depolymerase Predictor - A complete mahcine learning application for predicting phage depolymerases proteins with GUI
#Damian Magill & Timofey Skvortsov 2022

#Module Imports

#Tkinter

import tkinter as tk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

import sys, math

from itertools import product

from Bio import SeqIO
from Bio.SeqIO import FastaIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import numpy as np, pandas as pd

import sklearn
from sklearn.preprocessing import normalize, MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score

#Creation of main class incorporating major GUI elements and the machine learning program


class Application(tk.Frame, tk.Text):
    """ GUI application that Predicts Phage Depolymerase Proteins Using a Random Forest Machine Learning Approach. """
    #Class inheriting from tkinter to build GUI around ML portion

    def __init__(self, master):
        """ Initialize Frame. """
        super(Application, self).__init__(master)
        self.grid()
        self.create_widgets()       

    #Text output methods for writing concisely to text boxes

    def write(self, content):
        self.insert(tk.END, content)
 
    def writeln(self, content):
        self.insert(tk.END, content + '\n')

    #Creation of Widgets

    def create_widgets(self):
        """ Create widgets to bulid GUI Interaction. """

        # create instruction label and logo
        
        tk.Label(self,
        text = "DEPOLYMERASE PREDICT", font='Helvetica 18 bold'
        ).grid(row = 0, column = 0, columnspan = 2, rowspan = 2)

        #Creation of an Upload Button and associated method

        tk.Button(self,
        text = "Click to Upload Sequence",
        command = self.Upload
        ).grid(row = 10, column = 0)

        # create a submit button for parameter generation

        tk.Button(self,
        text = "Click to Generate Protein Parameters",
        command = self.Prot_Process
        ).grid(row = 12, column = 0)

        # create a submit button for prediction

        tk.Button(self,
                  text = "Click to Predict Potential DPs",
                  command = self.Predict
                  ).grid(row = 14, column = 0)

        #Creation of an View Results Button

        tk.Button(self,
        text = "View Results Following Prediction",
        command = self.ViewResults
        ).grid(row = 16, column = 0)
        
        #Creation of an about button

        tk.Button(self,
        text = "About",
        command = self.About
        ).grid(row = 18, column = 0)

        #Creation of a close button

        tk.Button(self,
        text = "Exit",
        command = self.Close
        ).grid(row = 20, column = 0)

    #More specific methods for the workflow
    #File upload of the protein multifasta and preprocessing into list - characters filtered out if forbidden (clean up)

    def Upload(self):
        """File upload of multifasta amino acids"""
        text_box.delete(1.0,tk.END)
        
        filename = filedialog.askopenfilename()
        text_box.insert(tk.END, '\n\nSelected: ' + str(filename))

        text_box.insert(tk.END, '\n\nPreprocessing...')
        global proteins
        global names

        proteins, names = [],[]
       
        with open(filename, "r") as infile:
            try:
                sequence = ''
                for line in infile:
                    if line.startswith('>'):
                        names.append(line)
                        proteins.append(sequence)
                        sequence = ''
                    else:
                        sequence += line.strip()
            except:
                text_box.insert(tk.END, '\n\nError in input. Please check the format of your file.')

        proteins = list(filter(("").__ne__, proteins))    
        names = list(filter(("").__ne__, [s.replace('>', '') for s in [t.replace('\n', '') for t in names]]))
        text_box.insert(tk.END, '\n\nPreprocessing Complete!')

    #Method which defines parameters used for comparing the sequences based on the amino acid physicochemical features

    def ProtParams(self, item):
            """Calculates protein paramaters for a multifasta protein input"""

            #output order - MW,Arom,Ins,Iso,Helices,Turns,Strands,Extred,ExtOx,GRAVY,Flexibility,AAcounts
            aa=[*'ACDEFGHIKLMNPQRSTVWY']

            x = ProteinAnalysis(item)

            xaaC,output = [],[]

            for i in aa:
                xaaC.append(x.get_amino_acids_percent()[i])

            a,b,c,d = x.molecular_weight(), x.aromaticity(), x.instability_index(), x.isoelectric_point()

            sec_struc = x.secondary_structure_fraction()
            e,f,g = sec_struc[0], sec_struc[1], sec_struc[2]

            epsilon_prot = x.molar_extinction_coefficient()
            h,i = epsilon_prot[0], epsilon_prot[1]
            j = x.gravy()

            try:
                k = sum(x.flexibility())/len(x.flexibility())
            except ArithmeticError:
                print("Flexibility calculation not allowed! Set to zero")
                k = 0
                   
            output.extend([a, b, c, d, e, f, g, h, i, j, k])
            output.extend(xaaC)

            return output

    #Method to count dipeptides and below tripeptides - categorises combinations into physicochemical types as all possible combinations is too large

    #Method to count dipeptides and below tripeptides - categorises combinations into physicochemical types as all possible combinations is too large

    def DiMer(self, seq):

        k=2 #i.e dipeptides
        
        groups={'A':'1','V':'1','G':'1','I':'1','L':'1','F':'2','T':'2','Y':'2',
                'M':'3','C':'3','H':'4','K':'4','R':'4','D':'5','E':'5',
                'S':'6','T':'6','N':'6','Q':'6','P':'7'}                        #Physicochemical classification of the AAs
                                                                                #1 = hydrocarbon R group
                                                                                #2 = Aromatic uncharged
                                                                                #3 = S containing
                                                                                #4 = Positive charged
                                                                                #5 = Negative charged
                                                                                #6 = Polar uncharged
                                                                                #7 = Proline...
        
        iteratives=[''.join (i) for i in product("1234567",repeat=k)] #All possible group combinations for subsequent vector creation

        for i in range (0,len(iteratives)):
            iteratives[i]=int(iteratives[i])

        ind=[]
        for i in range (0,len(iteratives)):
            ind.append(i)
            
        combinations=dict(zip(iteratives,ind))

        V=np.zeros(int((math.pow(7,k))))      #Establishing a vector account for the possible combinations of the AA groups
        try:
            for j in range (0,len(seq)-k+1):
                kmer=seq[j:j+k]
                c=''
                for l in range(0,k):
                    c+=groups[kmer[l]]
                    V[combinations[int(c)]]+=1
        except:
            count={'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
            for q in range(0,len(seq)):
                if seq[q]=='A' or seq[q]=='V' or seq[q]=='G':
                    count['1']+=1
                if seq[q]=='I' or seq[q]=='L'or seq[q]=='F' or seq[q]=='P':
                    count['2']+=1
                if seq[q]=='Y' or seq[q]=='M'or seq[q]=='T' or seq[q]=='S':
                    count['3']+=1
                if seq[q]=='H' or seq[q]=='N'or seq[q]=='Q' or seq[q]=='W':
                    count['4']+=1
                if seq[q]=='R' or seq[q]=='K':
                    count['5']+=1
                if seq[q]=='D' or seq[q]=='E':
                    count['6']+=1
                if seq[q]=='C':
                    count['7']+=1
            val=list(count.values())              #[ 0,0,0,0,0,0,0]
            key=list(count.keys())                #['1', '2', '3', '4', '5', '6', '7']
            m=0
            ind=0
            for t in range(0,len(val)):     #find maximum value from val
                if m<val[t]:
                    m=val[t]
                    ind=t
            m=key [ind]                     # m=group number of maximum occuring group alphabets in protein
            for j in range (0,len(seq)-k+1):
                kmer=seq[j:j+k]
                c=''
                for l in range(0,k):
                    if kmer[l] not in groups:
                        c+=m
                    else:
                        c+=groups[kmer[l]]
                V[combinations[int(c)]]+=1

        V=V/(len(seq)-1)
        return np.array(V)

    def TriMer(self, seq):

        k=3 #i.e tripeptides
        
        groups={'A':'1','V':'1','G':'1','I':'1','L':'1','F':'2','T':'2','Y':'2',
                'M':'3','C':'3','H':'4','K':'4','R':'4','D':'5','E':'5',
                'S':'6','T':'6','N':'6','Q':'6','P':'7'}                        #Physicochemical classification of the AAs
                                                                                #1 = hydrocarbon R group
                                                                                #2 = Aromatic uncharged
                                                                                #3 = S containing
                                                                                #4 = Positive charged
                                                                                #5 = Negative charged
                                                                                #6 = Polar uncharged
                                                                                #7 = Proline...
        
        iteratives=[''.join (i) for i in product("1234567",repeat=k)] #All possible group combinations for subsequent vector creation

        for i in range (0,len(iteratives)):
            iteratives[i]=int(iteratives[i])

        ind=[]
        for i in range (0,len(iteratives)):
            ind.append(i)
            
        combinations=dict(zip(iteratives,ind))

        V=np.zeros(int((math.pow(7,k))))      #Establishing a vector account for the possible combinations of the AA groups
        try:
            for j in range (0,len(seq)-k+1):
                kmer=seq[j:j+k]
                c=''
                for l in range(0,k):
                    c+=groups[kmer[l]]
                    V[combinations[int(c)]]+=1
        except:
            count={'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}
            for q in range(0,len(seq)):
                if seq[q]=='A' or seq[q]=='V' or seq[q]=='G':
                    count['1']+=1
                if seq[q]=='I' or seq[q]=='L'or seq[q]=='F' or seq[q]=='P':
                    count['2']+=1
                if seq[q]=='Y' or seq[q]=='M'or seq[q]=='T' or seq[q]=='S':
                    count['3']+=1
                if seq[q]=='H' or seq[q]=='N'or seq[q]=='Q' or seq[q]=='W':
                    count['4']+=1
                if seq[q]=='R' or seq[q]=='K':
                    count['5']+=1
                if seq[q]=='D' or seq[q]=='E':
                    count['6']+=1
                if seq[q]=='C':
                    count['7']+=1
            val=list(count.values())              #[ 0,0,0,0,0,0,0]
            key=list(count.keys())                #['1', '2', '3', '4', '5', '6', '7']
            m=0
            ind=0
            for t in range(0,len(val)):     #find maximum value from val
                if m<val[t]:
                    m=val[t]
                    ind=t
            m=key [ind]                     # m=group number of maximum occuring group alphabets in protein
            for j in range (0,len(seq)-k+1):
                kmer=seq[j:j+k]
                c=''
                for l in range(0,k):
                    if kmer[l] not in groups:
                        c+=m
                    else:
                        c+=groups[kmer[l]]
                V[combinations[int(c)]]+=1

        V2=V/(len(seq)-1)
        return np.array(V2)

    
    #Linked to the above methods - Formats protein parameters into pandas dataframe and provides intermediate output of parameters

    def Prot_Process(self):

        DiCol = ["di_" + str(i) for i in range(1,50)]
        TriCol = ["tri_" + str(j) for j in range(1,344)]

        output_columns = ["Name","MW","Aromaticity","Instability","Isoelectric","Helices","Turns",
                         "Strands","Extinction Red","Extinction Ox","GRAVY",'Flexibility','A', 'C', 'D',
                         'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
                         'Q', 'R', 'S', 'T', 'V', 'W', 'Y','Length'] + DiCol + TriCol

        print(len(output_columns))

        df = pd.DataFrame(columns=output_columns, index=None)

        for i, item in enumerate(proteins):
            di = self.DiMer(item)           #Calc Dimers from method above
            tri = self.TriMer(item)
            l = len(item)                   #Seq length
            x = self.ProtParams(item)
            x.insert(0,names[i])            #Inserting everything for output
            x.append(l)
            x += di.tolist()
            x += tri.tolist()
            df_length = len(df)
            df.loc[df_length] = x

        df.to_csv("ProteinParams.csv", index=False)

        text_box.insert(tk.END, '\n\nParameter Generation Complete!')
        text_box.insert(tk.END, '\n\nOutput File Written to: ProteinParams.csv') #Intermediate prot params file output.

    #Initialising model based on test set previously defined

    def ModelFit(self):
        """The optimised machine learning portion"""

        #Data Input and exploration/manipulation if necessary

        print("Importing and Processing Training Set and Input Data")

        acr_df = pd.read_csv('TrainingSet.csv')
        acr_df.index = acr_df['Name']
        acr_df = acr_df.drop(['Name'], axis=1)

        print("Finished!")

        #ML Preprocessing and train/test preparation

        print("Processing and Initialising Model...")

        X = acr_df.drop(['DP'], axis=1).values
        y = acr_df['DP'].values
        print(X.shape, y.shape)

        X_train, y_train = np.array(X),np.array(y)

        #Creation and fitting of model

        pipelineDP = Pipeline(steps=[
            ('PFeatures', PolynomialFeatures(2)),
            ('scaler', MinMaxScaler()),
            ('model', RandomForestClassifier(n_estimators=1500, criterion="entropy", max_features='auto',max_depth=30, bootstrap=True, min_samples_leaf=3,
                                             oob_score=False, min_samples_split=2))])

        global model_rf

        model_rf = pipelineDP.fit(X_train, y_train)

        print("Finished!")

    #Prediction of input data using the model

    def Predict(self):
        """Application of ML function of input data"""

        text_box.insert(tk.END, '\n\nPredicting...')

        predictions = pd.read_csv("ProteinParams.csv")
        predictions.index = predictions['Name']
        predictions = predictions.drop(['Name'], axis=1)


        newprob = model_rf.predict_proba(predictions)
        problist = newprob.tolist()
        global outputresults
        outputresults = []

        count = 1
        temp,temp2 = "",""
        ELIMINATE = ("[", "]")
        
        for i, item in enumerate(problist):
            item.insert(0, count)
            count += 1
            
        for j in problist:
            temp += str(j[0]) + "," + str(j[2])
            for k in temp:
                if k not in ELIMINATE:
                    temp2 += str(k)
                else:
                    pass
            outputresults.append(temp2)
            temp, temp2 = "",""
        
        with open("Predictions.csv", "w") as filename:
            filename.write("Gene,Probability_DePol,Sequence\n")
            for i, item in enumerate(outputresults):
                tempseq = str(proteins[i])
                filename.write(item)
                filename.write(",")
                filename.write(tempseq)
                filename.write("\n")
                tempseq = ""

        

        text_box.insert(tk.END, '\n\nComplete!')
        text_box.insert(tk.END, '\n\nOutput File Written to: Predictions.csv')

    #Allows output in separate window of predictions in addition to the output file

    def ViewResults(self):
        """Method for displaying Predictions Immediately to User"""

        text_box.insert(tk.END, "\n\nWorking...")

        g,y,view = "","",""
        x = 0
        
        for i in outputresults:
            for j, item in enumerate(i):
                if item == ",":
                    g = str(i[:j])
                    x = float(i[j+1:])
                if x < 0.25:
                    y = "Very Low - Very Unlikely to be Depolymerase"
                elif 0.25 <= x < 0.50:
                    y = "Low - Unlikey to be Depolymerase"
                elif 0.50 <= x < 0.75:
                    y = "Moderate - Potential Depolymerase Candidate with Low Confidence"
                elif 0.75 <= x < 0.9:
                    y = "High - Potential Depolymerase Candidate with Reasonable Confidence"
                else:
                    y = "Very High - Probable Depolymerase Candidate with Good Confidence"
      
            view += " " + str(g) + "\t      " + str("{:.3f}".format(x)) + "\t\t\t   " + str(y) + "\n"          
            g,y = "",""
            x = 0

        OutWindow = tk.Toplevel(self, bg="light sky blue")
        OutWindow
        self.W2 = tk.Text(OutWindow, width = 150, height = 20)
        self.W2.grid(row = 0, column = 0, columnspan = 5)
        self.W2.insert(tk.END, "Prediction Results\n\n")
        self.W2.insert(tk.END, "Gene\tProbability to be Depolymerase\t\t\t\t   Confidence\n\n")
        self.W2.insert(tk.END, view)

    #Information about the program in a separate window

    def About(self):
        """Generates Text Box with Info About Program"""
        newWindow = tk.Toplevel(self, bg="light sky blue")
        newWindow
        self.W1 = tk.Text(newWindow, width = 150, height = 20)
        self.W1.grid(row = 0, column = 0, columnspan = 5)
        self.W1.insert(tk.END, "Depolymerase Predictor - Damian Magill & Timofey Skvortsov 2022")
        self.W1.insert(tk.END, "\n\nMachine learning tool trained exclusively on experimentally proven depolymerase proteins followed by cross validation on")
        self.W1.insert(tk.END, "\nunseen data with an accuracy of 90% on this dataset.")
        self.W1.insert(tk.END, "\n\nInput is a multifasta file of amino acid sequences. Click to upload the file, to then generate parameters, and finally predictions.")
        self.W1.insert(tk.END, "\nVarious protein parameters are generated and output to a csv file.")
        self.W1.insert(tk.END, "\nThis includes abundance of each amino acid, hydrophobicity, secondary structure motifs etc")
        self.W1.insert(tk.END, "\nThese are compared to the trained model and probabilities extracted and output accordingly")
        self.W1.insert(tk.END, "\nProbabilities correspond to the likelihood of a given protein being a depolymerase.")
        self.W1.insert(tk.END, "\nTo be used for phages experiementally shown to have suspected depolymerase activity.")
        self.W1.insert(tk.END, "\n\nModel can be retrained/updated to a limited extent as new depolymerases are discovered.")
        self.W1.insert(tk.END, "\nTo do this, users can simply generate protein parameters using the tool for a new training ")
        self.W1.insert(tk.END, "\nset of sequences. This file can then be used to replace the existing one in the directory ")
        self.W1.insert(tk.END, "\nand the tool relaunched to take this into account. Expert users are free to modify model parameters ")
        self.W1.insert(tk.END, "\nas they see fit.")

        self.W1.insert(tk.END, "\n\nSupport/Requests/Questions: damianjmagill@gmail.com")

    def Close(self):
        """Closes the Program"""
        root.destroy()
        sys.exit()

#Main Portion of Program

def main():
    global root
    root = tk.Tk()
    root.title("Depolymerase Predictor")
    root.configure(background="SlateBlue1")
    app = Application(root)
    app.configure(bg="SlateBlue1")

    global text_box
    text_box = tk.Text(app, width = 130, height = 20)
    text_box.grid(row = 0, column = 5, columnspan = 4)
    text_box.insert(tk.END, "Welcome to Depolymerase Predict! Please Upload Your Sequence File Once the Program has Finished Initialisation.")

    text_box.insert(tk.END, "\n\nInitialising Modeller...")
    app.ModelFit()
    text_box.insert(tk.END, "\n\nInitialisation Complete!")

    root.mainloop()
       
 
if __name__ == '__main__':
    main()




