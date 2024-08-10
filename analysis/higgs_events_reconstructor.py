import os
import numpy as np
import pandas as pd
import uproot
import math
from enum import Enum
import requests

Lepton = Enum("Lepton", ["ELECTRON", "MUON"])


def calculate_momenergy(pt, eta, phi, **kwargs):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2)
    return {"px": px, "py": py, "pz": pz, "E": E}


def calculate_invariant_mass(E, px, py, pz, **kwargs):
    return np.sqrt(E**2 - px**2 - py**2 - pz**2)


def calculate_sip3d(dxy, dz, dxyErr, dzErr, **kwargs):
    return {"sip3d": np.sqrt(dxy**2 + dz**2) / np.sqrt(dxyErr**2 + dzErr**2)}


def reconstruct_higgs_events(df):
    # Filter events out with less then 4 leptons
    filtered_df = df[df["nMuon"] + df["nElectron"] > 4]

    # Declare final data
    combinations = []

    # Create final data
    for index, row in filtered_df.iterrows():
        # Combine muon and electron data
        pt_array = np.append(row["Muon_pt"], row["Electron_pt"])
        eta_array = np.append(row["Muon_eta"], row["Electron_eta"])
        phi_array = np.append(row["Muon_phi"], row["Electron_phi"])
        charge_array = np.append(row["Muon_charge"], row["Electron_charge"])
        isolation_array = np.append(
            row["Muon_pfRelIso03_all"], row["Electron_pfRelIso03_all"]
        )
        dxy_array = np.append(row["Muon_dxy"], row["Electron_dxy"])
        dz_array = np.append(row["Muon_dz"], row["Electron_dz"])
        dxyErr_array = np.append(row["Muon_dxyErr"], row["Electron_dxyErr"])
        dzErr_array = np.append(row["Muon_dzErr"], row["Electron_dzErr"])

        # Couple data arrays into lepton array
        leptons = [
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "charge": charge,
                "isolation": isolation,
                "dxy": dxy,
                "dz": dz,
                "dxyErr": dxyErr,
                "dzErr": dzErr,
                "type": Lepton.MUON if i < row["nMuon"] else Lepton.ELECTRON,
                "i": i,
            }
            for i, (
                pt,
                eta,
                phi,
                charge,
                isolation,
                dxy,
                dz,
                dxyErr,
                dzErr,
            ) in enumerate(
                zip(
                    pt_array,
                    eta_array,
                    phi_array,
                    charge_array,
                    isolation_array,
                    dxy_array,
                    dz_array,
                    dxyErr_array,
                    dzErr_array,
                )
            )
        ]

        # Split leptons into: positive and negative, remove unknown isolation (-999)
        positive_leptons = [
            lepton
            for lepton in leptons
            if lepton["charge"] > 0 and lepton["isolation"] != -999
        ]
        negative_leptons = [
            lepton
            for lepton in leptons
            if lepton["charge"] < 0 and lepton["isolation"] != -999
        ]

        # Skip if there are not at least 2 negative and 2 positive leptons
        if len(positive_leptons) < 2 or len(negative_leptons) < 2:
            continue

        # Go through every combination of first lepton pair
        for positive_lepton_1 in positive_leptons:
            for negative_lepton_1 in negative_leptons:
                # Skip if type of leptons are not the same
                if positive_lepton_1["type"] != negative_lepton_1["type"]:
                    continue

                # Calculate momenergy for positive lepton 1
                momenergy_positive_lepton_1 = calculate_momenergy(**positive_lepton_1)
                # Add momenergy
                positive_lepton_1 |= momenergy_positive_lepton_1

                # Calculate SIP3D for positive lepton 1
                sip3d_positive_lepton_1 = calculate_sip3d(**positive_lepton_1)
                # Add SIP3D
                positive_lepton_1 |= sip3d_positive_lepton_1

                # Calculate momenergy for negative lepton 1
                momenergy_negative_lepton_1 = calculate_momenergy(**negative_lepton_1)
                # Add momenergy
                negative_lepton_1 |= momenergy_negative_lepton_1

                # Calculate SIP3D for negative lepton 1
                sip3d_negative_lepton_1 = calculate_sip3d(**negative_lepton_1)
                # Add SIP3D
                negative_lepton_1 |= sip3d_negative_lepton_1

                # Add momenergy of positive lepton 1 with momenergy of negative lepton 1
                momenergy_sum_1 = {
                    i: momenergy_positive_lepton_1[i] + momenergy_negative_lepton_1[i]
                    for i in momenergy_positive_lepton_1.keys()
                }

                # Calculate invariant mass of positive lepton 1 and negative lepton 1 together
                invariant_mass_1 = calculate_invariant_mass(**momenergy_sum_1)

                # Go through every combination of second lepton pair
                for positive_lepton_2 in positive_leptons:
                    for negative_lepton_2 in negative_leptons:
                        # Skip if leptons are already used in first lepton pair
                        if (
                            positive_lepton_2["i"] == positive_lepton_1["i"]
                            or negative_lepton_2["i"] == negative_lepton_1["i"]
                        ):
                            continue

                        # Skip if type of leptons are not the same
                        if positive_lepton_2["type"] != negative_lepton_2["type"]:
                            continue

                        # Calculate momenergy for positive lepton 2
                        momenergy_positive_lepton_2 = calculate_momenergy(
                            **positive_lepton_2
                        )
                        # Add momenergy
                        positive_lepton_2 |= momenergy_positive_lepton_2

                        # Calculate SIP3D for positive lepton 2
                        sip3d_positive_lepton_2 = calculate_sip3d(**positive_lepton_2)
                        # Add SIP3D
                        positive_lepton_2 |= sip3d_positive_lepton_2

                        # Calculate momenergy for negative lepton 2
                        momenergy_negative_lepton_2 = calculate_momenergy(
                            **negative_lepton_2
                        )
                        # Add momenergy
                        negative_lepton_2 |= momenergy_negative_lepton_2

                        # Calculate SIP3D for negative lepton 2
                        sip3d_negative_lepton_2 = calculate_sip3d(**negative_lepton_2)
                        # Add SIP3D
                        negative_lepton_2 |= sip3d_negative_lepton_2

                        # Add momenergy of positive lepton 1 with momenergy of negative lepton 2
                        momenergy_sum_2 = {
                            i: momenergy_positive_lepton_2[i]
                            + momenergy_negative_lepton_2[i]
                            for i in momenergy_positive_lepton_2.keys()
                        }

                        # Skip if leptons have higher energy than leptons from Z_1
                        if (
                            positive_lepton_2["E"] > positive_lepton_1["E"]
                            or positive_lepton_2["E"] > negative_lepton_1["E"]
                            or negative_lepton_2["E"] > positive_lepton_1["E"]
                            or negative_lepton_2["E"] > negative_lepton_1["E"]
                        ):
                            continue

                        # Calculate invariant mass of positive lepton 2 and negative lepton 2 together
                        invariant_mass_2 = calculate_invariant_mass(**momenergy_sum_2)

                        # Calculate invariant mass of total system
                        momenergy_sum_total = {
                            i: momenergy_sum_1[i] + momenergy_sum_2[i]
                            for i in momenergy_sum_1.keys()
                        }
                        invariant_mass_total = calculate_invariant_mass(
                            **momenergy_sum_total
                        )

                        # Sort leptons descending of energy
                        lepton_1 = (
                            positive_lepton_1
                            if positive_lepton_1["E"] >= negative_lepton_1["E"]
                            else negative_lepton_1
                        )
                        lepton_2 = (
                            positive_lepton_1
                            if positive_lepton_1["E"] < negative_lepton_1["E"]
                            else negative_lepton_1
                        )
                        lepton_3 = (
                            positive_lepton_2
                            if positive_lepton_2["E"] >= negative_lepton_2["E"]
                            else negative_lepton_2
                        )
                        lepton_4 = (
                            positive_lepton_2
                            if positive_lepton_2["E"] < negative_lepton_2["E"]
                            else negative_lepton_2
                        )

                        # Add combination to final combinations
                        combination = {
                            "mass_Z_1": invariant_mass_1,
                            "mass_Z_2": invariant_mass_2,
                            "mass_H": invariant_mass_total,
                            "type_1_1": lepton_1["type"].name,
                            "px_1_1": lepton_1["px"],
                            "py_1_1": lepton_1["py"],
                            "pz_1_1": lepton_1["pz"],
                            "energy_1_1": lepton_1["E"],
                            "charge_1_1": lepton_1["charge"],
                            "isolation_1_1": lepton_1["isolation"],
                            "sip3d_1_1": lepton_1["sip3d"],
                            "type_1_2": lepton_2["type"].name,
                            "px_1_2": lepton_2["px"],
                            "py_1_2": lepton_2["py"],
                            "pz_1_2": lepton_2["pz"],
                            "energy_1_2": lepton_2["E"],
                            "charge_1_2": lepton_2["charge"],
                            "isolation_1_2": lepton_2["isolation"],
                            "sip3d_1_2": lepton_2["sip3d"],
                            "type_2_1": lepton_3["type"].name,
                            "px_2_1": lepton_3["px"],
                            "py_2_1": lepton_3["py"],
                            "pz_2_1": lepton_3["pz"],
                            "energy_2_1": lepton_3["E"],
                            "charge_2_1": lepton_3["charge"],
                            "isolation_2_1": lepton_3["isolation"],
                            "sip3d_2_1": lepton_3["sip3d"],
                            "type_2_2": lepton_4["type"].name,
                            "px_2_2": lepton_4["px"],
                            "py_2_2": lepton_4["py"],
                            "pz_2_2": lepton_4["pz"],
                            "energy_2_2": lepton_4["E"],
                            "charge_2_2": lepton_4["charge"],
                            "isolation_2_2": lepton_4["isolation"],
                            "sip3d_2_2": lepton_4["sip3d"],
                        }
                        combinations.append(combination)

    final_df = pd.DataFrame.from_dict(combinations)
    return final_df


def download_dataset(dataset, datasets_dir, datasets):
    print(f"Starting to download dataset {dataset}...")
    url = datasets[dataset]

    chunk_size = 1024 * 1024  # 1 MB chunk size
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Create directory if it doesn't exist
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)

        file_path = os.path.join(
            datasets_dir, dataset
        )  # File path within the datasets_dir
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print(f"Dataset '{dataset}' downloaded successfully into '{datasets_dir}'.")
    else:
        print(f"Failed to download the dataset. Status code: {response.status_code}")


datasets_dir = "./datasets"
datasets = {
    "SMHiggsToZZTo4L.root": "https://opendata.cern.ch/record/12361/files/SMHiggsToZZTo4L.root",
    "ZZTo4mu.root": "https://opendata.cern.ch/record/12362/files/ZZTo4mu.root",
    "ZZTo4e.root": "https://opendata.cern.ch/record/12363/files/ZZTo4e.root",
    "ZZTo2e2mu.root": "https://opendata.cern.ch/record/12364/files/ZZTo2e2mu.root",
    "Run2012B_DoubleMuParked.root": "https://opendata.cern.ch/record/12365/files/Run2012B_DoubleMuParked.root",
    "Run2012C_DoubleMuParked.root": "https://opendata.cern.ch/record/12366/files/Run2012C_DoubleMuParked.root",
    "Run2012B_DoubleElectron.root": "https://opendata.cern.ch/record/12367/files/Run2012B_DoubleElectron.root",
    "Run2012C_DoubleElectron.root": "https://opendata.cern.ch/record/12368/files/Run2012C_DoubleElectron.root",
}
num_entries_per_iteration = 5_000_000
output_dir = "./reconstructed_events"
columns = [
    "nMuon",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_charge",
    "Muon_pfRelIso03_all",
    "Muon_dxy",
    "Muon_dxyErr",
    "Muon_dz",
    "Muon_dzErr",
    "nElectron",
    "Electron_pt",
    "Electron_eta",
    "Electron_phi",
    "Electron_charge",
    "Electron_pfRelIso03_all",
    "Electron_dxy",
    "Electron_dxyErr",
    "Electron_dz",
    "Electron_dzErr",
]

if __name__ == "__main__":
    print("Starting reconstruction of higgs events...")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Number of entries per iteration: {num_entries_per_iteration}")

    for dataset in datasets.keys():
        if os.path.exists(f"{datasets_dir}/{dataset}"):
            print(f"Found dataset {dataset}")
        else:
            print(f"Could not find dataset {dataset} in {datasets_dir} directory!")
            download_dataset(dataset, datasets_dir, datasets)

        print(f"Analysing dataset {dataset} now.")

        tree = uproot.open(os.path.join(datasets_dir, dataset))["Events"]
        num_iterations = math.ceil(tree.num_entries / num_entries_per_iteration)
        for i in range(num_iterations):
            print(f"Analysing slice {i+1} of {num_iterations}...")

            df = tree.arrays(
                columns,
                entry_start=i * num_entries_per_iteration,
                entry_stop=min(tree.num_entries, (i + 1) * num_entries_per_iteration),
                library="pd",
            )
            rec_df = reconstruct_higgs_events(df)
            output_filename = os.path.join(
                output_dir, f"higgs_events_{'.'.join(dataset.split('.')[:-1])}_{i}.csv"
            )
            print(
                f"Analysis of dataset {dataset} finished. Saving to {output_filename}."
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            rec_df.to_csv(output_filename, index=False)

    print("Analysis finished successfully.")
