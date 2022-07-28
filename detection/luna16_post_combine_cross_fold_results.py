import json
import csv
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-i",
        "--input",
        nargs='+', default=[],
        help="input json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output csv",
    )

    args = parser.parse_args()

    in_json_list = args.input
    out_csv = args.output

    with open(out_csv, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['seriesuid','coordX','coordY','coordZ','probability'])
        for in_json in in_json_list:
            result = json.load(open(in_json, "r"))
            for subj in result["validation"]:
                seriesuid = os.path.split(subj["image"])[-1][:-7]
                for b in range(len(subj["box"])):
                    spamwriter.writerow([seriesuid]+subj["box"][b][0:3]+[subj["score"][b]])

if __name__ == '__main__':
    main()
