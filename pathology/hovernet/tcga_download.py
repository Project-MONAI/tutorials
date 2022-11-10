import requests
import json
import re

BASE_URL = "https://api.gdc.cancer.gov"


def extract_info(filename):
    """Extract wsi and patch info from filename

    Args:
        filename: name of the rgb or mask file in NuCLS dataset
    """
    wsi_name = filename.split("_id")[0]
    case_name, dx = wsi_name.rsplit("-", 1)
    matches = re.search("left-([0-9]+).*top-([0-9]+).*bottom-([0-9]+).*right-([0-9]+)", filename)
    left, top, bottom, right = [int(m) for m in matches.groups()]
    location = (top, left)
    size = max(right - left, bottom - top)
    print(location, size)
    file_name_wild = f"{case_name}-*{dx}*"
    return file_name_wild, location, size


def get_file_id(file_name_wild):
    """Retrieve file_id from partial filenames with wildcard

    Args:
        file_name_wild: partial filename of a file on TCGA with wildcard
    """
    file_filters = {
        "op": "=",
        "content": {
            "field": "files.file_name",
            "value": [file_name_wild],
        },
    }
    file_endpoint = f"{BASE_URL}/files"
    params = {"filters": json.dumps(file_filters)}
    response = requests.get(file_endpoint, params=params)
    print(json.dumps(response.json(), indent=2))
    return response.json()["data"]["hits"][0]["file_id"]


def download_file(file_id):
    """Download a file based on its file_id

    Args:
        file_id: UUID of a file on TCGA
    """
    query = f"{BASE_URL}/data/{file_id}"
    print(f"Fetching {file_id} ...")
    response = requests.get(query, headers={"Content-Type": "application/json"})
    response_head_cd = response.headers["Content-Disposition"]
    file_name = re.findall("filename=(.+)", response_head_cd)[0]

    with open(file_name, "wb") as output_file:
        output_file.write(response.content)
    print(f"{file_id} is saved in {file_name}.")


def create_file_info_list(filenames):
    """Create information records for each of NuCLS files

    Args:
        filenames: list of NuCLS filenames for images or masks
    """
    info_list = []
    for filename in filenames:
        file_name_wild, location, size = extract_info(filename)
        uid = get_file_id(file_name_wild)
        info_list.append((uid, filename, location, size))
    return info_list


if __name__ == "__main__":
    filenames = [
        "TCGA-A1-A0SP-DX1_id-5ea4095addda5f8398977ebc_left-7053_top-53967_bottom-54231_right-7311",
        "TCGA-A2-A0D0-DX1_id-5ea40b17ddda5f839899849a_left-69243_top-41106_bottom-41400_right-69527",
    ]
    info_list = create_file_info_list(filenames)
    print(f"{info_list=}")

    file_ids = list(set([r[0] for r in info_list]))
    print(f"{file_ids=}")

    for uid in file_ids:
        download_file(uid)
