import ROOT
import os
import numpy as np

def skim_file_doublemu(input_file, out_file_index, output_dir):
    """
    Skim DoubleMuon events for H -> ZZ -> 4l analysis.
    Requirements: 
    - At least 4 muons OR (at least 2 muons AND at least 2 electrons)
    - pT > 5 GeV for all considered leptons
    - Charge configuration allows for a neutral system (sum = 0)
    """
    
    # === Open input file and get tree === #
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    f_in = ROOT.TFile.Open(input_file, "READ")
    if not f_in or f_in.IsZombie():
        print(f"Error opening file: {input_file}")
        return

    tree_in = f_in.Get("Events")
    if not tree_in:
        print("Tree 'Events' not found.")
        f_in.Close()
        return

    tree_in.SetBranchStatus("*", 0)
    branches_to_keep = [
        "run", "luminosityBlock", "event",
        "nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Muon_pfRelIso04_all", "Muon_sip3d", "Muon_dxy", "Muon_dxyErr", 
        "Muon_dz", "Muon_dzErr", "Muon_tightId", "Muon_softId", 
        "Muon_softMvaId", "Muon_fsrPhotonIdx",
        "nElectron", "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", 
        "Electron_charge", "Electron_pfRelIso03_all", "Electron_dxy", 
        "Electron_dxyErr", "Electron_dz", "Electron_dzErr", 
        "Electron_mvaFall17V2noIso_WP90", "Electron_mvaFall17V2noIso_WP80", 
        "Electron_cutBased", "Electron_photonIdx"
    ]
    for branch in branches_to_keep:
        tree_in.SetBranchStatus(branch, 1)

    # === Create Output Directory and File === #
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.join(output_dir, f"skimmedDoubleMu_{out_file_index}.root")
    f_out = ROOT.TFile(out_name, "RECREATE")
    tree_out = tree_in.CloneTree(0)

    nentries = tree_in.GetEntries()
    events_kept = 0

    # === Skimming Logic Loop === #
    for i in range(nentries):
        tree_in.GetEntry(i)

        nMu = tree_in.nMuon
        nEle = tree_in.nElectron

        # Get kinematics/charges
        mu_pt = np.array(tree_in.Muon_pt) if nMu > 0 else np.array([])
        ele_pt = np.array(tree_in.Electron_pt) if nEle > 0 else np.array([])
        mu_q = np.array(tree_in.Muon_charge) if nMu > 0 else np.array([])
        ele_q = np.array(tree_in.Electron_charge) if nEle > 0 else np.array([])

        # pT filter; condition for good leptons
        mask_mu = mu_pt > 5
        mask_ele = ele_pt > 5

        good_mu_q = mu_q[mask_mu]
        good_ele_q = ele_q[mask_ele]

        n_good_mu = len(good_mu_q)
        n_good_ele = len(good_ele_q)

        # Count positive and negative charges to ensure neutral combinations are possible
        n_pos_mu = np.sum(good_mu_q > 0)
        n_neg_mu = np.sum(good_mu_q < 0)
        n_pos_ele = np.sum(good_ele_q > 0)
        n_neg_ele = np.sum(good_ele_q < 0)

        pass_skim = False

        # 4Mu Channel
        # Needs at least 4 muons total and at least two of each charge
        if (n_good_mu >= 4) and (n_pos_mu >= 2 and n_neg_mu >= 2):
            pass_skim = True

        # 2Mu2e Channel
        # Needs at least 2 muons (1 pos, 1 neg) and 2 electrons (1 pos, 1 neg)
        elif (n_good_mu >= 2 and n_good_ele >= 2):
            if (n_pos_mu >= 1 and n_neg_mu >= 1) and (n_pos_ele >= 1 and n_neg_ele >= 1):
                pass_skim = True

        if pass_skim:
            tree_out.Fill()
            events_kept += 1

    # Write and close
    tree_out.Write()
    f_out.Close()
    f_in.Close()
    
    print(f"File {out_file_index}: Kept {events_kept}/{nentries}")