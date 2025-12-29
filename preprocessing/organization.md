# Rediscovering the Higgs Boson
## Directory Organization
Rediscovering the Higgs Boson/
│
├── data/                       # (created by the script)
│   ├── raw/                    # (Temporary raw files are downloaded here, then get deleted)
│   └── skimmed/ 
│       ├── DoubleEG/    
│       │   ├── Run2016G/       
│       │   │   ├── skimmedDoubleEG_1.root
│       │   │   ├── skimmedDoubleEG_2.root
│       │   │   └── ... (hundreds of small files)
│       │   │
│       │   └── Run2016H/       
│       │       ├── skimmedDoubleEG_1.root
│       │       └── ...
│       │
│       └── DoubleMuon/ 
│           ├── Run2016G/       
│           │   ├── skimmedDoubleMu_1.root
│           │   ├── skimmedDoubleMu_2.root
│           │   └── ...
│           │
│           └── Run2016H/       
│               └── ...
│
├── preprocessing/             
│   ├── inputs/                 
│   │   ├── DoubleMu_Run2016G.txt   
│   │   ├── DoubleMu_Run2016H.txt   
│   │   └── ...
│   │
│   ├── logs/                   # (created by the script)
│   │   ├── index_DoubleMuon_Run2016G.txt
│   │   └── ...
│   │
│   ├── main_pipeline.py       
│   ├── skimming_Mu.py          
│   └── skimming_EG.py          
│
└── analysis/                   # (Empty for now)                