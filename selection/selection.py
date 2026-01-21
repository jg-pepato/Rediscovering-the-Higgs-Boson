import ROOT
import numpy as np
import array
import os
from itertools import combinations
import argparse

# ===== GLOBAL MASS VARIABLES =====
MASS_ELECTRON = 0.000511  # GeV
MASS_MUON = 0.106         # GeV

# ===== CALCULATING INVARIANT MASS =====
def get_4vector(pt, eta, phi, mass):
    """Create ROOT 4-vector from kinematic variables."""
    p4 = ROOT.TLorentzVector()
    p4.SetPtEtaPhiM(pt, eta, phi, mass)
    return p4


def calculate_4lepton_mass(tree, good_muons, good_electrons):
    """Calculate invariant mass of 4-lepton system."""
    
    four_vectors = []
    
    # Add muons
    for mu_idx in good_muons:
        pt = tree.Muon_pt[mu_idx]
        eta = tree.Muon_eta[mu_idx]
        phi = tree.Muon_phi[mu_idx]
        mass = MASS_MUON
        four_vectors.append(get_4vector(pt, eta, phi, mass))
    
    # Add electrons
    for e_idx in good_electrons:
        pt = tree.Electron_pt[e_idx]
        eta = tree.Electron_eta[e_idx]
        phi = tree.Electron_phi[e_idx]
        mass = MASS_ELECTRON
        four_vectors.append(get_4vector(pt, eta, phi, mass))
    
    # Sum all 4-vectors
    total_p = four_vectors[0]
    for vec in four_vectors[1:]:
        total_p += vec
    
    return total_p.M() 

# ===== CALCULATING ELECTRON SIP3D =====
def get_electron_sip3d(tree, index):
    """Safely get Electron_sip3d value."""
    electron_dxy = tree.Electron_dxy[index]
    electron_dxyErr = tree.Electron_dxyErr[index]
    electron_dz = tree.Electron_dz[index]
    electron_dzErr = tree.Electron_dzErr[index]

    sip3d = np.sqrt((electron_dxy / electron_dxyErr)**2 + (electron_dz / electron_dzErr)**2) if electron_dxyErr != 0 and electron_dzErr != 0 else 0.0
    return sip3d

# ===== SELECTION =====
ELECTRON_CUTS = {
    "pt": 7,
    "eta": 2.5,
    "isolation": 0.4,
    "sip": 4,
    "id": "WP80",  # Options: "WP80", "WP90", "cutBased"
}

MUON_CUTS = {
    "pt": 5,
    "eta": 2.4,
    "isolation": 0.4,
    "sip": 4,
}


def select_electrons(tree):
    """Apply electron selection cuts (quality cuts)."""
    n_ele = tree.nElectron
    if n_ele == 0:
        return np.array([], dtype=int)
    
    ele_pt = np.array(tree.Electron_pt)
    ele_eta = np.abs(np.array(tree.Electron_eta))
    ele_iso = np.array(tree.Electron_pfRelIso03_all)
    ele_dxy = np.array(tree.Electron_dxy)
    ele_dxyErr = np.array(tree.Electron_dxyErr)
    ele_dz = np.array(tree.Electron_dz)
    ele_dzErr = np.array(tree.Electron_dzErr)
    
    sip = np.sqrt((ele_dxy / ele_dxyErr)**2 + (ele_dz / ele_dzErr)**2)
    
    mask = (
        (ele_pt > ELECTRON_CUTS["pt"]) &
        (ele_eta < ELECTRON_CUTS["eta"]) &
        (ele_iso < ELECTRON_CUTS["isolation"]) &
        (sip < ELECTRON_CUTS["sip"])
    )
    
    # Apply ID cut
    if ELECTRON_CUTS["id"] == "WP80":
        ele_id = np.array(tree.Electron_mvaFall17V2noIso_WP80)
    elif ELECTRON_CUTS["id"] == "WP90":
        ele_id = np.array(tree.Electron_mvaFall17V2noIso_WP90)
    elif ELECTRON_CUTS["id"] == "cutBased":
        ele_cutbased = np.array(tree.Electron_cutBased)
        ele_id = ele_cutbased >= 3
    
    mask = mask & ele_id
    return np.where(mask)[0]


def select_muons(tree):
    """Apply muon selection cuts (quality cuts)."""
    n_mu = tree.nMuon
    if n_mu == 0:
        return np.array([], dtype=int)
    
    mu_pt = np.array(tree.Muon_pt)
    mu_eta = np.abs(np.array(tree.Muon_eta))
    mu_iso = np.array(tree.Muon_pfRelIso04_all)
    mu_sip = np.array(tree.Muon_sip3d)
    
    mask = (
        (mu_pt > MUON_CUTS["pt"]) &
        (mu_eta < MUON_CUTS["eta"]) &
        (mu_iso < MUON_CUTS["isolation"]) &
        (mu_sip < MUON_CUTS["sip"])
    )
    
    mu_tight = np.array(tree.Muon_tightId)
    mu_soft = np.array(tree.Muon_softId)

    low_pt = mu_pt < 10
    id_low = mu_tight | mu_soft
    mask_id = mu_tight
    mask_id[low_pt] = id_low[low_pt]

    mask = mask & mask_id

    return np.where(mask)[0]

def find_best_4lepton_candidate(tree, good_mu, good_e, channel):
    """
    Evaluates all possible OSSF pairings in the event.
    Returns: (bool pass_filter, float m4l)
    """
    Z_MASS = 91.1876
    candidates = []

    # 1. Collect all valid OSSF pairings (4mu channel)
    if channel == "4mu":
        if len(good_mu) >= 4:
            for idxs in combinations(good_mu, 4):
                q = [tree.Muon_charge[j] for j in idxs]
                if sum(q) != 0: continue
                p4s = [get_4vector(tree.Muon_pt[j], tree.Muon_eta[j], tree.Muon_phi[j], MASS_MUON) for j in idxs]
                for p1, p2 in [((0,1), (2,3)), ((0,2), (1,3)), ((0,3), (1,2))]:
                    if (q[p1[0]] + q[p1[1]] == 0) and (q[p2[0]] + q[p2[1]] == 0):
                        m_a = (p4s[p1[0]]+p4s[p1[1]]).M()
                        m_b = (p4s[p2[0]]+p4s[p2[1]]).M()
                        candidates.append({'p4s': [p4s[p1[0]], p4s[p1[1]], p4s[p2[0]], p4s[p2[1]]], 
                                        'm_pairs': (m_a, m_b), 
                                        'indices': {'mu': [idxs[p1[0]], idxs[p1[1]], idxs[p2[0]], idxs[p2[1]]],'el': []} })

    # 2. Collect all valid OSSF pairings (2mu2e channel)
    elif channel == "2mu2e":
        if len(good_mu) >= 2 and len(good_e) >= 2:
            for m_idx in combinations(good_mu, 2):
                for e_idx in combinations(good_e, 2):
                    if (tree.Muon_charge[m_idx[0]] + tree.Muon_charge[m_idx[1]] == 0) and \
                    (tree.Electron_charge[e_idx[0]] + tree.Electron_charge[e_idx[1]] == 0):
                        p4_mu = [get_4vector(tree.Muon_pt[j], tree.Muon_eta[j], tree.Muon_phi[j], MASS_MUON) for j in m_idx]
                        p4_el = [get_4vector(tree.Electron_pt[j], tree.Electron_eta[j], tree.Electron_phi[j], MASS_ELECTRON) for j in e_idx]
                        candidates.append({'p4s': p4_mu + p4_el, 
                                        'm_pairs': ((p4_mu[0]+p4_mu[1]).M(), (p4_el[0]+p4_el[1]).M()), 
                                        'indices': {'mu': [m_idx[0], m_idx[1]],'el': [e_idx[0], e_idx[1]]}})
    
    elif channel == "4e":
        if len(good_e) >= 4:
            for idxs in combinations(good_e, 4):
                q = [tree.Electron_charge[j] for j in idxs]
                if sum(q) != 0: continue
                p4s = [get_4vector(tree.Electron_pt[j], tree.Electron_eta[j], tree.Electron_phi[j], MASS_ELECTRON) for j in idxs]
                for p1, p2 in [((0,1), (2,3)), ((0,2), (1,3)), ((0,3), (1,2))]:
                    if (q[p1[0]] + q[p1[1]] == 0) and (q[p2[0]] + q[p2[1]] == 0):
                        m_a, m_b = (p4s[p1[0]]+p4s[p1[1]]).M(), (p4s[p2[0]]+p4s[p2[1]]).M()
                        candidates.append({'p4s': [p4s[p1[0]], p4s[p1[1]], p4s[p2[0]], p4s[p2[1]]], 'm_pairs': (m_a, m_b), 'indices': {'mu': [],'el': [idxs[p1[0]], idxs[p1[1]], idxs[p2[0]], idxs[p2[1]]]}})

    else:
        print("Invalid channel.")
        return False, 0.0, None

    # 3. Arbitrate between candidates
    best_m4l = -1.0
    min_z1_diff = 1e6
    best_cand = None
    for cand in candidates:
        m_a, m_b = cand['m_pairs']
        # Identify Z1 and Z2
        z1, z2 = (m_a, m_b) if abs(m_a - Z_MASS) < abs(m_b - Z_MASS) else (m_b, m_a)
        
        # Apply cuts from the paper
        if (40 < z1 < 120) and (12 < z2 < 120):
            if abs(z1 - Z_MASS) < min_z1_diff:
                min_z1_diff = abs(z1 - Z_MASS)
                total_p4 = cand['p4s'][0] + cand['p4s'][1] + cand['p4s'][2] + cand['p4s'][3]
                best_m4l = total_p4.M()
                best_cand = cand


    if best_m4l > 0:
        return True, best_m4l, best_cand['indices']
    return False, 0.0, None

## ===== GENERATE HISTOGRAM AND TREE =====
def analyze_and_plot(input_file, channel, output_filename="higgs_skim.root", weight=1.0, label=-1):
    # 1. Setup Output File and Tree
    out = ROOT.TFile(output_filename, "RECREATE")
    out_tree = ROOT.TTree("SelectedEvents", "Selected 4-Lepton Events")
    hist = ROOT.TH1F("M4L", "4-Lepton Invariant Mass;M_{4l} (GeV);Events", 40, 70, 180)

    tree_branches = [
        "lep1_pt", "lep1_costheta", "lep1_phi", "lep1_m", "lep1_iso", "lep1_sip",
        "lep2_pt", "lep2_costheta", "lep2_phi", "lep2_m", "lep2_iso", "lep2_sip",
        "lep3_pt", "lep3_costheta", "lep3_phi", "lep3_m", "lep3_iso", "lep3_sip",
        "lep4_pt", "lep4_costheta", "lep4_phi", "lep4_m", "lep4_iso", "lep4_sip",
        "m4l", "weight", "label"
    ]

    dict_4lepton = {branch: array.array('f', [0.0]) for branch in tree_branches}

    for branch in tree_branches:
        out_tree.Branch(branch, dict_4lepton[branch], f"{branch}/F")

    selected_events = 0
    
    # 2. Process File
    if not os.path.exists(input_file): return
    f_in = ROOT.TFile.Open(input_file, "READ")
    tree = f_in.Get("Events")
    if not tree: return
        
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        good_mu = select_muons(tree)
        good_e = select_electrons(tree) 

        passed, m4l, indices = find_best_4lepton_candidate(tree, good_mu, good_e, channel)

        if passed:
            temp_leps = [('mu', idx) for idx in indices['mu']] + [('el', idx) for idx in indices['el']]
            
            temp_leps.sort(key=lambda x: tree.Muon_pt[x[1]] if x[0]=='mu' else tree.Electron_pt[x[1]], reverse=True)

            for n, (l_type, idx) in enumerate(temp_leps, 1):
                if l_type == 'mu':
                    dict_4lepton[f"lep{n}_pt"][0] = tree.Muon_pt[idx] / m4l 
                    dict_4lepton[f"lep{n}_costheta"][0] = np.tanh(tree.Muon_eta[idx])
                    dict_4lepton[f"lep{n}_phi"][0] = tree.Muon_phi[idx]
                    dict_4lepton[f"lep{n}_m"][0] = MASS_MUON
                    dict_4lepton[f"lep{n}_iso"][0] = tree.Muon_pfRelIso04_all[idx]
                    dict_4lepton[f"lep{n}_sip"][0] = tree.Muon_sip3d[idx]
                else:
                    dict_4lepton[f"lep{n}_pt"][0] = tree.Electron_pt[idx] / m4l
                    dict_4lepton[f"lep{n}_costheta"][0] = np.tanh(tree.Electron_eta[idx])
                    dict_4lepton[f"lep{n}_phi"][0] = tree.Electron_phi[idx]
                    dict_4lepton[f"lep{n}_m"][0] = MASS_ELECTRON
                    dict_4lepton[f"lep{n}_iso"][0] = tree.Electron_pfRelIso03_all[idx]
                    dict_4lepton[f"lep{n}_sip"][0] = get_electron_sip3d(tree, idx)

            dict_4lepton["m4l"][0] = m4l
            dict_4lepton["weight"][0] = weight
            dict_4lepton["label"][0] = label
                
            out_tree.Fill()
            hist.Fill(m4l, weight)
            selected_events += 1
            
    f_in.Close()
    
    # 3. Write and Close 
    out.cd()
    out_tree.Write()
    hist.Write() 
    out.Close()
    print(f"Done! Selected {selected_events} events saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_name", default="higgs_mass.root")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--channel", choices=["4mu", "2mu2e", "4e"], required=True)
    parser.add_argument("--label", type=int, default=-1)
    args = parser.parse_args()

    analyze_and_plot(args.input_file, args.channel, args.output_name, args.weight, args.label)