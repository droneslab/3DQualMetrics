import json
import numpy as np
import os
import argparse
import pandas as pd
import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to json files directory")
    parser.add_argument("--save_path", type=str, required=True, help="Save path for csv file")
    args = parser.parse_args()

    path = args.input_dir

    output_path = args.save_path
    # Candidates = ["DAVIS", "VILLAGE", "HILTI", "MAI", "Warehouse"]

    Candidates = [pos_json for pos_json in os.listdir(path) if os.path.isdir(os.path.join(path, pos_json))]
    #Candidates = ["DAVIS"]
    print ("Candidates: {}".format(Candidates))

    data_dict = {}
    output_dict = {}

    SLAM_METHODS = ["FAST", "LeGO", "SHINE"]

    for candidate in Candidates:
        print ("Processing Candidate: {}".format(candidate))
        data_dict[candidate] = {}
        candidate_path = os.path.join(path, candidate)

        for SM in SLAM_METHODS:
            data_dict[candidate][SM] = {}

            # SLAM_METHOD = candidate_path.split("/")[-1].split("_")[0]        

            try:
                # ref_json = [pos_json for pos_json in os.listdir(candidate_path) if pos_json.endswith('_ref_metrics.json')][0]
                # Get reference json file which ends with _ref_metrics.json and starts with SLAM_METHOD
                ref_json = [pos_json for pos_json in os.listdir(candidate_path) if pos_json.endswith('_ref_metrics.json') and pos_json.startswith(SM)][0]
                print ("Reference json file: {}".format(ref_json))
                with open(os.path.join(candidate_path, ref_json)) as f:
                    data_ref = json.load(f)

                data_dict[candidate][SM]["REF"] = data_ref
                data_dict[candidate][SM]["REF"]["Path"] = os.path.join(candidate_path, ref_json)
            except:
                print("Reference json file not found for {}".format(candidate))
                data_dict[candidate][SM]["REF"] = None

            data_dict[candidate][SM]["PQM"] = {}        
            candidate_jsons = [pos_json for pos_json in os.listdir(candidate_path) if pos_json.endswith('.json') and not pos_json.endswith('_ref_metrics.json') and pos_json.startswith(SM)]
            for candidate_json in candidate_jsons:
                eps = candidate_json.split("_eps_")[1].split("_")[0][:-5]
                size = candidate_json.split("_size_")[1].split("_")[0]
                data_dict[candidate][SM]["PQM"][eps, size] = {}

                data_dict[candidate][SM]["PQM"][eps, size]["Path"] = os.path.join(candidate_path, candidate_json)

                # Load json file
                with open(data_dict[candidate][SM]["PQM"][eps, size]["Path"]) as f:
                    data = json.load(f)
                data_dict[candidate][SM]["PQM"][eps, size]["Metrics"] = data["average"]

    # pprint.pprint(data_dict)
    print ("Generating csv file...")

    # Generate dataframe from dictionary
    # Rows are chamfer, hausforff from REF and completeness, accuracy, artifacts, resolution from avergae from PQM
    # Columns are candidate, eps, size
    rows = []
    for candidate in Candidates:
        for SM in SLAM_METHODS:
            for eps, size in data_dict[candidate][SM]["PQM"].keys():
                row = []
                row.append(candidate)
                row.append(SM)
                row.append(eps)
                row.append(size)
                try:
                    row.append(data_dict[candidate][SM]["REF"]["chamfer"])
                    row.append(data_dict[candidate][SM]["REF"]["hausdorff"])
                except:
                    row.append(None)
                    row.append(None)    
                row.append(data_dict[candidate][SM]["PQM"][eps, size]["Metrics"]["completeness"])
                row.append(data_dict[candidate][SM]["PQM"][eps, size]["Metrics"]["accuracy"])
                row.append(data_dict[candidate][SM]["PQM"][eps, size]["Metrics"]["artifacts"])
                row.append(data_dict[candidate][SM]["PQM"][eps, size]["Metrics"]["resolution"])
                rows.append(row)

    df = pd.DataFrame(rows, columns=["Candidate","SM", "eps", "size", "chamfer", "hausdorff", "completeness", "accuracy", "artifacts", "resolution"])
    print (df)
    # Save dataframe to csv file
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(output_path + timestr + ".csv")
            
if __name__ == "__main__":
    main()


