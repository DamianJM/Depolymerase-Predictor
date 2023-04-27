#!/usr/bin/env python

# A command-line implementation of DePolymerase Predictor (DePP) - a machine learning tool for predicting phage depolymerases
# Damian Magill & Timofey Skvortsov 2023

import argparse
from utils.data_processing import read_fasta, calculate_protein_parameters, generate_dataframe, save_dataframe
from utils.depolymerase_prediction import train_model, predict_depolymerases

def main():
    # Step 1: Parse arguments
    parser = argparse.ArgumentParser(description='DePolymerase Predictor (DePP)')
    parser.add_argument('-i', '--input', required=True, help='Protein FASTA file as input')
    parser.add_argument('-t', '--training_set', default='./TrainingSet/TrainingSet.csv', help='CSV file containing training set (default: ./TrainingSet/TrainingSet.csv)')
    parser.add_argument('-op', '--output_parameters', help='CSV file to save calculated protein parameters (optional)')
    parser.add_argument('-o', '--output_predictions', help='CSV file to save depolymerase predictions (optional)')

    args = parser.parse_args()

    # Step 2: Accept multi-FASTA files containing protein sequences
    fasta_file = args.input
    protein_records = read_fasta(fasta_file)

    # Step 3: Generate a pandas dataframe containing protein parameters
    # [print(protein_record.seq) for protein_record in protein_records]

    protein_parameters = [calculate_protein_parameters(protein_record) for protein_record in protein_records]
    protein_df = generate_dataframe(protein_parameters)
    if args.output_parameters:
        save_dataframe(protein_df, args.output_parameters)

    # Step 4: Train a random forest model using the training set
    training_set_path = args.training_set
    model = train_model(training_set_path)

    # Step 5: Predict probabilities of being depolymerases for each protein
    predictions = predict_depolymerases(protein_df, model)

    # Step 6: Save the predictions to a CSV file or print to the screen
    if args.output_predictions:
        save_dataframe(predictions, args.output_predictions)
    else:
        print(predictions)

if __name__ == '__main__':
    main()