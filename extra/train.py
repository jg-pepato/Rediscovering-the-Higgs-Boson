import numpy as np
import tensorflow as tf
from higgsdnnmodel import HiggsClassifier
import ROOT
import pandas as pd

def train_higgs_classifier(signal_path, background_path, model_save_path, epochs=50, batch_size=32):
    # Load Data 
    df_sig = ROOT.RDataFrame("SelectedEvents", signal_path)
    df_bkg = ROOT.RDataFrame("SelectedEvents", background_path)

    # Convert to Pandas and Combine
    sig_pd = pd.DataFrame(df_sig.AsNumpy())
    bkg_pd = pd.DataFrame(df_bkg.AsNumpy())
    full_df = pd.concat([sig_pd, bkg_pd], ignore_index=True)

    # Calculate class weights
    sig_len = len(sig_pd)
    bkg_len = len(bkg_pd)
    total_len = sig_len + bkg_len

    sig_w = total_len / (2 * sig_len)
    bkg_w = total_len / (2 * bkg_len)
    class_weights = {1: float(sig_w), 0: float(bkg_w)}

    # Shuffle the data
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Prepare features and labels
    y = full_df['label'].values.copy()
    X_df = full_df.drop(columns=['label', 'weight', 'm4l'])
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    
    mask = ~X_df.isna().any(axis=1)
    X_df = X_df[mask]
    y = y[mask]
    
    X = X_df.values.astype('float32')
    y = y.astype('float32')

    # Setup Model
    model = HiggsClassifier(input_dim=X.shape[1])
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    # Train
    model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights
    )

    # Save
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_higgs_classifier(
        signal_path='sel_doublemu_4mu_signal.root', 
        background_path='sel_doublemu_4mu_noise.root', 
        model_save_path='4mu_higgs_classifier_model.keras'
    )