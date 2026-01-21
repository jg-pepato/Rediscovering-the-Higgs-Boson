import numpy as np
import pandas as pd
import tensorflow as tf
import ROOT
from higgsdnnmodel import HiggsClassifier

def mlp_denoised_plot(model_path, threshold=0.95):
    # Open files
    f_dataG  = ROOT.TFile("final/sel_doublemu_4mu_2016G.root")
    f_dataH  = ROOT.TFile("final/sel_doublemu_4mu_2016H.root")
    f_noise  = ROOT.TFile("final/sel_doublemu_4mu_noise.root")
    f_signal = ROOT.TFile("final/sel_doublemu_4mu_signal.root")

    # Get hists
    h_dataG  = f_dataG.Get("M4L")
    h_dataH  = f_dataH.Get("M4L")
    h_noise  = f_noise.Get("M4L")
    h_signal = f_signal.Get("M4L")
    
    # Combine data from runs G and H
    h_data_total = h_dataG.Clone("h_data_total")
    h_data_total.Add(h_dataH)
    
    # Style noise histogram (Light blue)
    h_noise.SetFillColor(ROOT.kAzure - 9)
    h_noise.SetLineColor(ROOT.kBlack)
    
    # Style signal histogram (Light red)
    h_signal.SetFillColor(ROOT.kRed - 7)
    h_signal.SetLineColor(ROOT.kBlack)
    
    # Style data histogram (Black dots)
    h_data_total.SetMarkerStyle(20)
    h_data_total.SetMarkerSize(1.2)
    h_data_total.SetLineColor(ROOT.kBlack)
    
    # Create stack
    stack = ROOT.THStack("hs", "CMS Open Data 2016(G+H) - 4mu channel;M_{4l} (GeV);Events")
    stack.Add(h_noise)  
    stack.Add(h_signal)
    
    # ==================
    # DENOISED HISTOGRAM
    # ==================
    
    # Get DataFrames from data files
    dfG = pd.DataFrame(ROOT.RDataFrame("SelectedEvents", "final/sel_doublemu_4mu_2016G.root").AsNumpy())
    dfH = pd.DataFrame(ROOT.RDataFrame("SelectedEvents", "final/sel_doublemu_4mu_2016H.root").AsNumpy())
    
    # Combine data frames
    full_df = pd.concat([dfG, dfH], ignore_index=True)
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare features (excluding label, weight, and m4l columns)
    features = full_df.drop(columns=['label', 'weight', 'm4l'])
    
    # Convert to np array
    X = features.values.astype('float32')
    
    # Predict using the model
    predictions = model.predict(X)
    
    # Filter based on threshold value
    keep = predictions.flatten() >= threshold
    filtered_df = full_df[keep]
    
    # Create denoised histogram
    denoised_hist = ROOT.TH1F("m4l_denoised", "Denoised m4l; m4l (GeV); Events", 40, 70, 180)
    for m4l_val in filtered_df['m4l']:
        denoised_hist.Fill(m4l_val)
    
    # Style denoised histogram (Green with black outline)
    denoised_hist.SetFillColor(ROOT.kGreen - 3)
    denoised_hist.SetLineColor(ROOT.kBlack)

    # ==================
    # CREATE CANVAS AND PLOT
    # ==================

    # Create canvas
    canvas = ROOT.TCanvas("c1", "c1", 800, 700)
    ROOT.gStyle.SetOptStat(0)
    
    # Draw stack and data
    stack.Draw("HIST")
    h_data_total.Draw("SAME E1")
    denoised_hist.Draw("SAME HIST")
    stack.SetMaximum(h_data_total.GetMaximum() * 1.5)
    
    # Add legend
    legend = ROOT.TLegend(0.7, 0.7, 0.95, 0.95)
    legend.AddEntry(h_noise, "Noise", "f")
    legend.AddEntry(h_signal, "Signal", "f")
    legend.AddEntry(denoised_hist, "Denoised", "f")
    legend.AddEntry(h_data_total, "Data", "pe")
    legend.Draw()
    
    canvas.Update()
    canvas.SaveAs("4mu_higgs_final_plot.png")
    
    # Clean up
    f_dataG.Close()
    f_dataH.Close()
    f_noise.Close()
    f_signal.Close()

if __name__ == "__main__":
    model_path = "4mu_higgs_classifier_model.keras" 
    mlp_denoised_plot(model_path, threshold=0.7)