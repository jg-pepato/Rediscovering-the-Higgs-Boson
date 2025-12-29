import ROOT
import numpy as np
import os
from itertools import combinations

# ===== CALCULATING INVARIANT MASS =====
def get_4vector(pt, eta, phi, mass):
    """Create ROOT 4-vector from kinematic variables."""
    p4 = ROOT.TLorentzVector()
    p4.SetPtEtaPhiM(pt, eta, phi, mass)
    return p4


def get_lepton_mass(lepton_type):
    """Get rest mass of lepton."""
    masses = {
        "electron": 0.000511,  # GeV
        "muon": 0.106,          # GeV
    }
    return masses.get(lepton_type, 0)

def calculate_4lepton_mass(tree, good_muons, good_electrons):
    """Calculate invariant mass of 4-lepton system."""
    
    four_vectors = []
    
    # Add muons
    for mu_idx in good_muons:
        pt = tree.Muon_pt[mu_idx]
        eta = tree.Muon_eta[mu_idx]
        phi = tree.Muon_phi[mu_idx]
        mass = get_lepton_mass("muon")
        four_vectors.append(get_4vector(pt, eta, phi, mass))
    
    # Add electrons
    for e_idx in good_electrons:
        pt = tree.Electron_pt[e_idx]
        eta = tree.Electron_eta[e_idx]
        phi = tree.Electron_phi[e_idx]
        mass = get_lepton_mass("electron")
        four_vectors.append(get_4vector(pt, eta, phi, mass))
    
    # Sum all 4-vectors
    total_p = four_vectors[0]
    for vec in four_vectors[1:]:
        total_p += vec
    
    return total_p.M()  # Return invariant mass


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

def find_best_4lepton_candidate(tree, good_mu, good_e):
    """
    Evaluates all possible OSSF pairings in the event.
    Returns: (bool pass_filter, float m4l)
    """
    Z_MASS = 91.1876
    candidates = []

    # 1. Collect all valid OSSF pairings (4mu channel)
    if len(good_mu) >= 4:
        for idxs in combinations(good_mu, 4):
            q = [tree.Muon_charge[j] for j in idxs]
            if sum(q) != 0: continue
            p4s = [get_4vector(tree.Muon_pt[j], tree.Muon_eta[j], tree.Muon_phi[j],get_lepton_mass("muon")) for j in idxs]
            for p1, p2 in [((0,1), (2,3)), ((0,2), (1,3)), ((0,3), (1,2))]:
                if (q[p1[0]] + q[p1[1]] == 0) and (q[p2[0]] + q[p2[1]] == 0):
                    m_a = (p4s[p1[0]]+p4s[p1[1]]).M()
                    m_b = (p4s[p2[0]]+p4s[p2[1]]).M()
                    candidates.append({'p4s': [p4s[p1[0]], p4s[p1[1]], p4s[p2[0]], p4s[p2[1]]], 'm_pairs': (m_a, m_b)})

    # 2. Collect all valid OSSF pairings (2mu2e channel)
    if len(good_mu) >= 2 and len(good_e) >= 2:
        for m_idx in combinations(good_mu, 2):
            for e_idx in combinations(good_e, 2):
                if (tree.Muon_charge[m_idx[0]] + tree.Muon_charge[m_idx[1]] == 0) and \
                   (tree.Electron_charge[e_idx[0]] + tree.Electron_charge[e_idx[1]] == 0):
                    p4_mu = [get_4vector(tree.Muon_pt[j], tree.Muon_eta[j], tree.Muon_phi[j],get_lepton_mass("muon")) for j in m_idx]
                    p4_el = [get_4vector(tree.Electron_pt[j], tree.Electron_eta[j], tree.Electron_phi[j],get_lepton_mass("electron")) for j in e_idx]
                    candidates.append({'p4s': p4_mu + p4_el, 'm_pairs': ((p4_mu[0]+p4_mu[1]).M(), (p4_el[0]+p4_el[1]).M())})

    # 3. Arbitrate between candidates
    best_m4l = -1.0
    min_z1_diff = 1e6
    
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

    if best_m4l > 0:
        return True, best_m4l
    return False, 0.0

# ===== MAIN ANALYSIS =====
def analyze_and_plot(input_files, output_filename="higgs_mass4.root"):
    hist = ROOT.TH1F("M4L", "4-Lepton Invariant Mass;M_{4l} (GeV);Events", 40, 70, 180)
    selected_events = 0
    
    for input_file in input_files:
        if not os.path.exists(input_file): continue
        f_in = ROOT.TFile.Open(input_file, "READ")
        tree = f_in.Get("Events")
        if not tree: continue
        
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            
            good_e = select_electrons(tree)
            good_mu = select_muons(tree)
            
            # Use the separate logic function
            passed, m4l = find_best_4lepton_candidate(tree, good_mu, good_e)
            
            if passed:
                hist.Fill(m4l)
                selected_events += 1
            
        f_in.Close()
    
    out = ROOT.TFile(output_filename, "RECREATE")
    hist.Write()
    out.Close()
    print(f"Analysis Complete. Selected {selected_events} events.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", required=True, help="Local path to the directory containing the ROOT files.")
    parser.add_argument("--output_name", required=False)
    args = parser.parse_args()
    
    path = args.input_files
    files = os.listdir(path)
    abs_path = [os.path.abspath(os.path.join(path, f)) for f in files if f.endswith(".root")]
    analyze_and_plot(abs_path, args.output_name)