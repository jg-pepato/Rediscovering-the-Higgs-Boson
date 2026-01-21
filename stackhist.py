import ROOT

def make_final_plot():
    # Open files
    f_dataG  = ROOT.TFile("sel_doublemu_2mu2e_2016G.root")
    f_dataH  = ROOT.TFile("sel_doublemu_2mu2e_2016H.root")
    f_noise  = ROOT.TFile("sel_doublemu_2mu2e_noise.root")
    f_signal = ROOT.TFile("sel_doublemu_2mu2e_signal.root")

    # Get hists
    h_dataG  = f_dataG.Get("M4L")
    h_dataH  = f_dataH.Get("M4L")
    h_noise  = f_noise.Get("M4L")
    h_signal = f_signal.Get("M4L")

    # Combine data from runs G and H
    h_data_total = h_dataG.Clone("h_data_total")
    h_data_total.Add(h_dataH)

    # 4. STYLING
    h_noise.SetFillColor(ROOT.kAzure - 9)
    h_noise.SetLineColor(ROOT.kBlack)

    # Signal (Solid Red)
    h_signal.SetFillColor(ROOT.kRed - 7)
    h_signal.SetLineColor(ROOT.kBlack)

    # Data (Black dots)
    h_data_total.SetMarkerStyle(20)
    h_data_total.SetMarkerSize(1.2)
    h_data_total.SetLineColor(ROOT.kBlack)

    # Create stack
    stack = ROOT.THStack("hs", "CMS Open Data 2016(G+H) - 2mu2e channel;M_{4l} 125 (GeV);Events")
    stack.Add(h_noise)  # Bottom of stack
    stack.Add(h_signal) # Top of stack

    # Create canvas
    canvas = ROOT.TCanvas("c1", "c1", 800, 700)
    ROOT.gStyle.SetOptStat(0) 

    stack.Draw("HIST")
    stack.SetMaximum(h_data_total.GetMaximum() * 1.5) 

    h_data_total.Draw("SAME E1")

    # Create legend
    legend = ROOT.TLegend(0.55, 0.7, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.AddEntry(h_data_total, "Data (Run G+H)", "lep")
    legend.AddEntry(h_noise, "ZZ Background", "f")
    legend.AddEntry(h_signal, "H #rightarrow ZZ #rightarrow 4L", "f")
    legend.Draw()

    canvas.Update()
    canvas.SaveAs("2mu2e_higgs_final_plot.png")
    print("Meaningful plot created: 2mu2e_higgs_final_plot.png")

if __name__ == "__main__":
    make_final_plot()