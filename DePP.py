#!/usr/bin/env python

#DePP (DePolymerase Predictor) - A complete machine learning application for predicting phage depolymerases proteins with GUI
#Damian Magill & Timofey Skvortsov 2022

#Module Imports

#Tkinter

import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage
import tkinter.scrolledtext as tkst

import math, sys
from itertools import product

from Bio import SeqIO
from Bio.SeqIO import FastaIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score


class Application(tk.Frame):
    """Class for main program"""
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        # Create widgets
        self.create_widgets()

        # Create text box
        self.text_box = tk.Text(self, width=130, height=20)
        scrollbar = tk.Scrollbar(self, command=self.text_box.yview)
        self.text_box.config(yscrollcommand=scrollbar.set)

        self.text_box.grid(row=0, column=5, columnspan=4)
        scrollbar.grid(row=0, column=9, sticky='ns')

        self.text_box.insert(tk.END, "Welcome to Depolymerase Predict! Please Upload Your Sequence File Once the Program has Finished Initialisation.")
        self.text_box.insert(tk.END, "\n\nInitialising Modeller...")

        # Initialise the model
        self.ModelFit()
        self.text_box.insert(tk.END, "\n\nInitialisation Complete!")

    # def invoke(self):
    #     super().invoke()
    #     self.autoscroll()

    def autoscroll(self):
        if self.text_box:
            self.text_box.see(tk.END)

    def write(self, content):
        self.insert(tk.END, content)

    def save_file_params_callback(self):
        if hasattr(self, 'save_file_params'):
            self.save_file_params()
        else:
            self.text_box.insert(tk.END, '\n\nError: You need to predict protein parameters first.')

    def save_file_predictions_callback(self):
        if hasattr(self, 'save_file_predictions'):
            self.save_file_predictions()
        else:
            self.text_box.insert(tk.END, '\n\nError: You need to predict DPs first.')

    def writeln(self, content):
        self.insert(tk.END, content + '\n')

    def create_widgets(self):
        """Create widgets for GUI interaction."""

        # Logo settings

        # Load the logo image and resize it
        logo_image = PhotoImage(file="./img/DePP_logo.png").subsample(1)
        # Replace "./img/DePP_logo.png" with the path to your logo file and adjust the subsample factor as needed

        logo_label = tk.Label(self, image=logo_image)
        logo_label.image = logo_image  # Keep a reference to the image to prevent garbage collection
        logo_label.grid(row=0, column=0, columnspan=2, rowspan=2)

        # Set the size of the image
        logo_label.config(width=200, height=200)
        # Replace 200 with the desired width and height in pixels
        
        # Button settings
        button_style = {"bg": "#ffffff", "font": "Helvetica 12", "width": 30, "anchor": "w"}

        # Create buttons and pack them to the parent widget
        button_texts = ["Upload Protein Sequences (FASTA)", "Generate Protein Parameters", "Save Protein Parameters",
                        "Predict Potential DPs", "View Results", "Save Predictions", "About", "Exit"]
        button_commands = [self.Upload, self.Prot_Process, self.save_file_params_callback,
                           self.Predict, self.ViewResults, self.save_file_predictions_callback, self.About, self.Close]
        self.buttons = []
        for i, text in enumerate(button_texts):
            button = tk.Button(self, text=text, command=button_commands[i], **button_style)
            button.grid(row=10+i*2, column=0, sticky="w", padx=(10, 0))
            self.buttons.append(button)

        # Bind the autoscroll() method to the <Button> event of all the buttons
        for button in self.buttons:
            button.bind("<Button>", self.autoscroll)
        
    def autoscroll(self, event):
        if self.text_box:
            self.text_box.see(tk.END)


    def About(self):
        """Generates a new window with information about the program"""
        # Create a new window with a light blue background
        newWindow = tk.Toplevel(self, bg="light sky blue")
        newWindow.title("About")
        
        # Create a text box with width 150 and height 20 inside the new window
        self.W1 = tk.Text(newWindow, width=150, height=20)
        self.W1.grid(row=0, column=0, columnspan=5)
        # Place the text box in the top row, starting from column 0 and spanning 5 columns
        
        # Add a header to the text box
        self.W1.insert(tk.END, "Depolymerase Predictor - Damian Magill & Timofey Skvortsov 2022")
        self.W1.insert(tk.END, "\n\nMachine learning tool trained exclusively on experimentally proven depolymerase proteins followed by cross validation on unseen data with an accuracy of 90% on this dataset.")
        # Add information about the input and output of the program
        self.W1.insert(tk.END, "\n\nInput is a multifasta file of amino acid sequences. Click to upload the file, to then generate parameters, and finally predictions.")
        self.W1.insert(tk.END, "\nVarious protein parameters are generated and output to a csv file.")
        self.W1.insert(tk.END, "\nThis includes abundance of each amino acid, hydrophobicity, secondary structure motifs, etc.")
        self.W1.insert(tk.END, "\nThese are compared to the trained model and probabilities extracted and output accordingly.")
        self.W1.insert(tk.END, "\nProbabilities correspond to the likelihood of a given protein being a depolymerase.")
        self.W1.insert(tk.END, "\nTo be used for phages experimentally shown to have suspected depolymerase activity.")
        # Add information on how to retrain the model
        self.W1.insert(tk.END, "\n\nModel can be retrained/updated to a limited extent as new depolymerases are discovered.")
        self.W1.insert(tk.END, "\nTo do this, users can generate protein parameters using the tool for a new training set of sequences.")
        self.W1.insert(tk.END, "\nThis file can then replace the existing one in the directory and the tool relaunched to take this into account.")
        self.W1.insert(tk.END, "\nExpert users are free to modify model parameters as they see fit.")
        # Add contact information
        self.W1.insert(tk.END, "\n\nSupport/Requests/Questions: damianjmagill@gmail.com")



    def Upload(self):
        """Handles the file upload of multifasta amino acids."""

        # Clears the text box before displaying the file selection
        self.text_box.delete(1.0, tk.END)

        # Opens a file dialog to choose a file
        filename = filedialog.askopenfilename()
        # Displays the selected filename in the text box
        self.text_box.insert(tk.END, '\n\nSelected: ' + str(filename))

        # Displays a message indicating the start of preprocessing
        self.text_box.insert(tk.END, '\n\nPreprocessing...')

        # Defines two empty global lists to store protein sequences and their names
        global proteins
        global names

        # Resets the lists
        proteins, names = [], []

        # Reads the selected file
        with open(filename, "r") as infile:
            try:
                # Reads the file line by line
                sequence = ''
                for line in infile:
                    # If the line starts with '>', it's a sequence name
                    if line.startswith('>'):
                        names.append(line)
                        proteins.append(sequence)
                        sequence = ''
                    # Otherwise, it's a sequence line
                    else:
                        sequence += line.strip()
            # If there's an error reading the file, display an error message
            except:
                self.text_box.insert(tk.END, '\n\nError in input. Please check the format of your file.')

        # Removes any empty strings from the protein list
        proteins = list(filter(("").__ne__, proteins))
        # Removes '>' and '\n' characters from the sequence names
        names = list(filter(("").__ne__, [s.replace('>', '') for s in [t.replace('\n', '') for t in names]]))
        # Displays a message indicating that preprocessing is complete
        self.text_box.insert(tk.END, '\n\nPreprocessing Complete!')

    def ProtParams(self, item):
        """Calculates protein parameters for a multifasta protein input."""

        # Define a list of amino acids
        aa = [*'ACDEFGHIKLMNPQRSTVWY']

        # Calculate protein parameters
        x = ProteinAnalysis(item)
        xaaC, output = [], []

        for i in aa:
            xaaC.append(x.get_amino_acids_percent()[i])

        # Calculate molecular weight, aromaticity, instability index, and isoelectric point
        a, b, c, d = x.molecular_weight(), x.aromaticity(), x.instability_index(), x.isoelectric_point()

        # Calculate secondary structure fractions
        sec_struc = x.secondary_structure_fraction()
        e, f, g = sec_struc[0], sec_struc[1], sec_struc[2]

        # Calculate molar extinction coefficient, GRAVY index, and flexibility
        epsilon_prot = x.molar_extinction_coefficient()
        h, i = epsilon_prot[0], epsilon_prot[1]
        j = x.gravy()
        try:
            k = sum(x.flexibility()) / len(x.flexibility())
        except ArithmeticError:
            print("Flexibility calculation not allowed! Set to zero")
            k = 0

        # Append calculated parameters to output list
        output.extend([a, b, c, d, e, f, g, h, i, j, k])
        output.extend(xaaC)

        return output


    def DiMer(self, seq):
        k = 2  # i.e dipeptides

        groups = {'A': '1', 'V': '1', 'G': '1', 'I': '1', 'L': '1', 'F': '2', 'T': '2', 'Y': '2',
                'M': '3', 'C': '3', 'H': '4', 'K': '4', 'R': '4', 'D': '5', 'E': '5',
                'S': '6', 'T': '6', 'N': '6', 'Q': '6', 'P': '7'}
        # Map each amino acid to a physicochemical group

        iteratives = [''.join(i) for i in product("1234567", repeat=k)]
        # Generate all possible combinations of physicochemical groups

        for i in range(0, len(iteratives)):
            iteratives[i] = int(iteratives[i])

        ind = []
        for i in range(0, len(iteratives)):
            ind.append(i)

        combinations = dict(zip(iteratives, ind))
        # Map each combination to an index in the vector representation of the dipeptide

        V = np.zeros(int((math.pow(7, k))))
        # Initialise the vector representation of the dipeptide as a vector of zeros

        try:
            for j in range(0, len(seq) - k + 1):
                kmer = seq[j:j + k]
                c = ''
                for l in range(0, k):
                    c += groups[kmer[l]]
                    # Map each amino acid in the kmer to its corresponding physicochemical group
                V[combinations[int(c)]] += 1
        except:
            count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
            for q in range(0, len(seq)):
                if seq[q] == 'A' or seq[q] == 'V' or seq[q] == 'G':
                    count['1'] += 1
                if seq[q] == 'I' or seq[q] == 'L' or seq[q] == 'F' or seq[q] == 'P':
                    count['2'] += 1
                if seq[q] == 'Y' or seq[q] == 'M' or seq[q] == 'T' or seq[q] == 'S':
                    count['3'] += 1
                if seq[q] == 'H' or seq[q] == 'N' or seq[q] == 'Q' or seq[q] == 'W':
                    count['4'] += 1
                if seq[q] == 'R' or seq[q] == 'K':
                    count['5'] += 1
                if seq[q] == 'D' or seq[q] == 'E':
                    count['6'] += 1
                if seq[q] == 'C':
                    count['7'] += 1
            val = list(count.values())  # [ 0,0,0,0,0,0,0]
            key = list(count.keys())  # ['1', '2', '3', '4', '5', '6', '7']
            m = 0
            ind = 0
            for t in range(0, len(val)):  # find maximum value from val
                if m < val[t]:
                    m = val[t]
                    ind = t
            m = key[ind]  # m=group number of maximum occurring group alphabets in protein
            for j in range(0, len(seq) - k + 1):
                kmer = seq[j:j + k]
                c = ''
                for l in range(0, k):
                    if kmer[l] not in groups:
                        c += m
                    else:
                        c += groups[kmer[l]]
                V[combinations[int(c)]] += 1

        V = V / (len(seq) - 1)
        return np.array(V)

    def TriMer(self, seq):
        k = 3  # Tripeptides

        # Physicochemical classification of the AAs
        groups = {'A': '1', 'V': '1', 'G': '1', 'I': '1', 'L': '1',
                'F': '2', 'T': '2', 'Y': '2', 'M': '3', 'C': '3',
                'H': '4', 'K': '4', 'R': '4', 'D': '5', 'E': '5',
                'S': '6', 'T': '6', 'N': '6', 'Q': '6', 'P': '7'}

        # Generate all possible group combinations for vector creation
        iteratives = [''.join(i) for i in product("1234567", repeat=k)]
        iteratives = [int(i) for i in iteratives]

        # Create a dictionary to map combinations to indices
        combinations = dict(zip(iteratives, range(len(iteratives))))

        # Initialise a vector to account for the possible combinations of AA groups
        V = np.zeros(int(math.pow(7, k)))

        # Count the occurrence of each group in the sequence
        count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        for aa in seq:
            if aa in groups:
                count[groups[aa]] += 1

        # Find the group number of the maximum occurring group alphabets in the protein
        max_group = max(count, key=count.get)

        # Iterate over the sequence and calculate tripeptide vectors
        for j in range(0, len(seq) - k + 1):
            kmer = seq[j:j + k]
            # Replace non-group AAs with the group number of the maximum occurring group alphabets
            c = ''.join(groups.get(aa, max_group) for aa in kmer)
            V[combinations[int(c)]] += 1

        # Normalize the vector
        V2 = V / (len(seq) - 1)
        return np.array(V2)

    def Prot_Process(self):
        # Define output column names
        DiCol = ["di_" + str(i) for i in range(1, 50)]
        TriCol = ["tri_" + str(j) for j in range(1, 344)]
        output_columns = ["Name", "MW", "Aromaticity", "Instability", "Isoelectric", "Helices", "Turns",
                            "Strands", "Extinction Red", "Extinction Ox", "GRAVY", 'Flexibility', 'A', 'C', 'D',
                            'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
                            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Length'] + DiCol + TriCol

        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame(columns=output_columns)

        # Loop through each protein sequence and calculate its parameters
        for i, item in enumerate(proteins):
            # Calculate dimers and trimers from the methods above
            di = self.DiMer(item)           
            tri = self.TriMer(item)

            # Calculate protein parameters
            l = len(item)                   # Sequence length
            x = self.ProtParams(item)
            x.insert(0, names[i])           # Insert the name of the protein at the beginning of the list
            x.append(l)                     # Append sequence length to the end of the list
            x += di.tolist()                # Append dimers to the end of the list
            x += tri.tolist()               # Append trimers to the end of the list

            # Add the protein parameters to the DataFrame
            df_length = len(df)
            df.loc[df_length] = x

        self.prot_params = df

        self.text_box.insert(tk.END, '\n\nParameter Generation Complete!')

        def save_file_params():
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
            if file_path:
                df.to_csv(file_path, index=False)
                self.text_box.insert(tk.END, f'\n\nProtein parameters saved to: {file_path}')
            else:
                self.text_box.insert(tk.END, '\n\nFile save operation was canceled.')
        
        self.save_file_params = save_file_params

        # # Save the DataFrame to a CSV file
        # df.to_csv("ProteinParams.csv", index=False)

        # Display output messages in the GUI

    def ModelFit(self):
        """Fit the optimised machine learning model"""

        # Import and process training set and input data
        print("Importing and Processing Training Set and Input Data")
        training_set_path = './TrainingSet/TrainingSet.csv'  # Define the path to the training set
        acr_df = pd.read_csv(training_set_path)  # Read the CSV file into a pandas dataframe
        acr_df.index = acr_df['Name']  # Set the index of the dataframe to the protein names
        acr_df = acr_df.drop(['Name'], axis=1)  # Remove the protein names column from the dataframe

        print("Finished!")

        # Machine learning preprocessing and train/test preparation
        print("Processing and Initialising Model...")
        X = acr_df.drop(['DP'], axis=1).values  # Define the input features (exclude the output class column)
        y = acr_df['DP'].values  # Define the output class column
        print(X.shape, y.shape)

        X_train, y_train = np.array(X), np.array(y)  # Convert the input and output to numpy arrays

        # Define the machine learning model pipeline
        pipelineDP = Pipeline(steps=[
            ('PFeatures', PolynomialFeatures(2)),  # Generate polynomial features
            ('scaler', MinMaxScaler()),  # Scale the features to a range of 0-1
            ('model', RandomForestClassifier(n_estimators=1500, criterion="entropy", max_features='auto', max_depth=30, bootstrap=True, min_samples_leaf=3, oob_score=False, min_samples_split=2))])

        # Fit the machine learning model
        self.model_rf = pipelineDP.fit(X_train, y_train)  # Set the machine learning model as a class attribute

        print("Finished!")

    def Predict(self):
        """Apply the ML function on input data and save the results to a CSV file."""
        self.text_box.insert(tk.END, '\n\nPredicting...')

        # Load protein data from file and set index to "Name"
        protein_data = self.prot_params
        protein_data.index = protein_data['Name']
        protein_data = protein_data.drop(['Name'], axis=1)

        # Check if the model has been initialised before proceeding
        if not hasattr(self, 'model_rf') or self.model_rf is None:
            self.text_box.insert(tk.END, '\n\nError: Model has not been initialised')
            return

        # Use the trained model to predict probabilities of DePol for each protein
        probabilities = self.model_rf.predict_proba(protein_data).tolist()

        # Create an empty list to store the results
        self.output_results = []

        # Iterate over the probabilities and format them as strings
        for i, prob in enumerate(probabilities):
            probability = "{:.6f}".format(prob[1])
            self.output_results.append(f"{i+1},{probability}")

        self.text_box.insert(tk.END, '\n\nComplete! You can save and view your predictions now')

        def save_file_predictions():
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, "w") as file:
                    file.write("Gene,Probability_DePol,Sequence\n")
                    for i, result in enumerate(self.output_results):
                        file.write(f"{result},{proteins[i]}\n")
                        
                self.text_box.insert(tk.END, f'\n\nDepolymerase predictions saved to: {file_path}')
            else:
                self.text_box.insert(tk.END, '\n\nFile save operation was canceled.')
        
        self.save_file_predictions = save_file_predictions



        # # Save the results to a CSV file
        # with open("Predictions.csv", "w") as file:
        #     file.write("Gene,Probability_DePol,Sequence\n")
        #     for i, result in enumerate(self.output_results):
        #         file.write(f"{result},{proteins[i]}\n")

        # # Display completion message in the text box
        # self.text_box.insert(tk.END, '\n\nComplete!')
        # self.text_box.insert(tk.END, '\n\nOutput File Written to: Predictions.csv')

    def ViewResults(self):
        """Display the prediction results in a separate window."""
        def get_confidence_level(probability):
            """Return a string indicating the confidence level based on the given probability."""
            if probability < 0.25:
                return "Very Low - Very Unlikely to be Depolymerase"
            elif probability < 0.50:
                return "Low - Unlikely to be Depolymerase"
            elif probability < 0.75:
                return "Moderate - Potential Depolymerase Candidate with Low Confidence"
            elif probability < 0.9:
                return "High - Potential Depolymerase Candidate with Reasonable Confidence"
            else:
                return "Very High - Probable Depolymerase Candidate with Good Confidence"

        self.text_box.insert(tk.END, "\n\nWorking...")

        # Create a string with the prediction results
        results_str = ""
        for i, result in enumerate(self.output_results):
            probability = float(result.split(',')[1])
            confidence = get_confidence_level(probability)
            results_str += f"{i+1}\t{probability:.3f}\t{confidence}\n"

        # Create a separate window to display the results
        OutWindow = tk.Toplevel(self, bg="light sky blue")
        result_text = tk.Text(OutWindow, width=150, height=20)
        result_text.grid(row=0, column=0, columnspan=5)
        result_text.insert(tk.END, "Prediction Results\n\n")
        result_text.insert(tk.END, "Gene\tProbability to be Depolymerase\tConfidence\n\n")
        result_text.insert(tk.END, results_str)

 
    def Close(self):
        """Close the program."""
        root.destroy()
        sys.exit()


def main():
    """Create the root window and initialise the Application class."""
    global root

    root = tk.Tk()
    root.title("DePP - DePolymerase Predictor")
    root.configure(background="SlateBlue1")


    app = Application(root)
    app.configure(bg="SlateBlue1")

    root.mainloop()


if __name__ == '__main__':
    main()